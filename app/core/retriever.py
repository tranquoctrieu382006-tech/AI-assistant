"""Vector and BM25 retrieval, score gating, and optional cross-encoder reranking."""
from __future__ import annotations

import copy
import logging
import re
import threading
import unicodedata
from collections import defaultdict

from langchain_core.documents import Document

from app.core import vector_store
from app.core.config import settings

logger = logging.getLogger(__name__)

_BM25_INSTANCE = None
_BM25_REVISION: tuple[int, int] | None = None
_BM25_LOCK = threading.RLock()
_RRF_CONSTANT = 60
_RRF_WEIGHTS = (0.5, 0.5)  # BM25, vector


def _bm25_tokens(text: str) -> list[str]:
    """Normalize Unicode/case and discard punctuation around Vietnamese tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def get_bm25_retriever(k: int = 5):
    """Return a per-call BM25 retriever over a revision-aware cached corpus index."""
    global _BM25_INSTANCE, _BM25_REVISION
    revision = vector_store.get_corpus_revision()
    if _BM25_INSTANCE is None or _BM25_REVISION != revision:
        with _BM25_LOCK:
            # The cache may have been populated/reindexed while this caller waited.
            revision = vector_store.get_corpus_revision()
            if _BM25_INSTANCE is None or _BM25_REVISION != revision:
                from langchain_community.retrievers import BM25Retriever

                chunks = vector_store.get_all_chunks()
                docs = [
                    Document(
                        page_content=chunk["content"],
                        metadata={
                            "source": chunk["source"],
                            "chunk_index": chunk["chunk_index"],
                            "page_number": chunk.get("page_number"),
                            "section": chunk.get("section"),
                        },
                    )
                    for chunk in chunks
                ]
                if not docs:
                    _BM25_INSTANCE = None
                    _BM25_REVISION = revision
                    raise vector_store.VectorStoreNotReady("RAG vector store contains no indexed chunks.")
                _BM25_INSTANCE = BM25Retriever.from_documents(
                    docs,
                    k=max(k, 20),
                    preprocess_func=_bm25_tokens,
                )
                _BM25_REVISION = revision
    # The cached model shares its immutable token index; request-specific k is
    # applied to a shallow copy so concurrent calls cannot overwrite each other.
    per_call = copy.copy(_BM25_INSTANCE)
    per_call.k = k
    return per_call


def clear_bm25_cache() -> None:
    """Clear cached BM25 state, for tests and explicit in-process reindexing."""
    global _BM25_INSTANCE, _BM25_REVISION
    with _BM25_LOCK:
        _BM25_INSTANCE = None
        _BM25_REVISION = None


def _hybrid_fusion(bm25_docs: list[Document], vector_matches: list[dict], top_k: int) -> list[Document]:
    """Fuse two ranked lists with weighted RRF while keeping source/chunk identity."""
    scores: dict[tuple[str, int], float] = defaultdict(float)
    documents: dict[tuple[str, int], Document] = {}

    for weight, ranked_items, is_bm25 in (
        (_RRF_WEIGHTS[0], bm25_docs, True),
        (_RRF_WEIGHTS[1], vector_matches, False),
    ):
        for rank, item in enumerate(ranked_items, start=1):
            if is_bm25:
                source = item.metadata["source"]
                chunk_index = item.metadata["chunk_index"]
                doc = item
            else:
                source = item["source"]
                chunk_index = item["chunk_index"]
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": source,
                        "chunk_index": chunk_index,
                        "page_number": item.get("page_number"),
                        "section": item.get("section"),
                    },
                )
            key = (source, chunk_index)
            scores[key] += weight / (_RRF_CONSTANT + rank)
            documents.setdefault(key, doc)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [documents[key] for key in ranked_keys[:top_k]]


def retrieve_and_gate(
    query: str,
    top_k: int = 5,
    mode: str = "vector",
    use_reranker: bool = False,
    reranker_top_n: int = 5,
    reranker_pool_size: int = 20,
) -> tuple[list[dict], bool]:
    """Retrieve evidence, enforce per-chunk cosine gating, and optionally rerank."""
    if mode not in {"vector", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    candidate_k = max(top_k, reranker_pool_size) if use_reranker else top_k
    if mode == "hybrid":
        query_vector = vector_store.embed_query(query)
        vector_matches = vector_store.similarity_search(
            query, k=candidate_k, embedding=query_vector
        )
    else:
        query_vector = None
        vector_matches = vector_store.similarity_search(query, k=candidate_k)
    gated_vector_matches = [
        match for match in vector_matches
        if match["score"] >= settings.rag_min_relevance_score
    ]
    if mode == "vector" and not gated_vector_matches:
        return [], True

    if mode == "vector":
        candidates = gated_vector_matches
    else:
        bm25_docs = get_bm25_retriever(k=candidate_k).invoke(query)
        # Gate after fusion, so keep both lists' candidates until each has a
        # real cosine score; otherwise low-score BM25 hits can consume top-k.
        fused_docs = _hybrid_fusion(bm25_docs, vector_matches, candidate_k * 2)
        score_lookup = {
            (match["source"], match["chunk_index"]): match["score"]
            for match in vector_matches
        }
        missing_keys = [
            (doc.metadata["source"], doc.metadata["chunk_index"])
            for doc in fused_docs
            if (doc.metadata["source"], doc.metadata["chunk_index"]) not in score_lookup
        ]
        score_lookup.update(
            vector_store.score_chunks(query, missing_keys, embedding=query_vector)
        )
        candidates = []
        for doc in fused_docs:
            key = (doc.metadata["source"], doc.metadata["chunk_index"])
            score = score_lookup.get(key)
            if score is None or score < settings.rag_min_relevance_score:
                continue
            candidates.append({
                "content": doc.page_content,
                "source": key[0],
                "chunk_index": key[1],
                "score": float(score),
                "page_number": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
            })
        if not candidates:
            return [], True
        if not use_reranker:
            candidates = candidates[:top_k]

    if not use_reranker or len(candidates) <= 1:
        return candidates[:reranker_top_n if use_reranker else top_k], False

    from app.core.reranker import rerank_documents

    docs = [
        Document(
            page_content=candidate["content"],
            metadata={
                "source": candidate["source"],
                "chunk_index": candidate["chunk_index"],
                "score": candidate["score"],
                "page_number": candidate.get("page_number"),
                "section": candidate.get("section"),
            },
        )
        for candidate in candidates
    ]
    reranked_docs = rerank_documents(
        query,
        docs,
        top_n=reranker_top_n,
        model_name=settings.reranker_model,
    )
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata["source"],
            "chunk_index": doc.metadata["chunk_index"],
            "score": doc.metadata["score"],
            "rerank_score": doc.metadata.get("rerank_score"),
            "page_number": doc.metadata.get("page_number"),
            "section": doc.metadata.get("section"),
        }
        for doc in reranked_docs
    ], False
