"""
Unified retrieval abstraction supporting Vector, Hybrid (BM25 + PGVector), and Reranking.
"""
from __future__ import annotations

import logging
from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.core.config import settings
from app.core import vector_store

logger = logging.getLogger(__name__)

_BM25_INSTANCE = None
_BM25_REVISION: tuple[int, int] | None = None


class VectorStoreRetriever(BaseRetriever):
    """LangChain BaseRetriever adapter around vector_store.similarity_search."""
    top_k: int = Field(default=5)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        matches = vector_store.similarity_search(query, k=self.top_k)
        return [
            Document(
                page_content=m["content"],
                metadata={
                    "chunk_id": f"{m['source']}::{m['chunk_index']}",
                    "source": m["source"],
                    "chunk_index": m["chunk_index"],
                    "score": m["score"],
                    "page_number": m.get("page_number"),
                    "section": m.get("section"),
                },
            )
            for m in matches
        ]


def get_bm25_retriever(k: int = 5):
    """Build or retrieve cached BM25Retriever over all corpus chunks."""
    global _BM25_INSTANCE, _BM25_REVISION
    revision = vector_store.get_corpus_revision()
    if _BM25_INSTANCE is None or _BM25_REVISION != revision:
        from langchain_community.retrievers import BM25Retriever
        raw_chunks = vector_store.get_all_chunks()
        docs = [
            Document(
                page_content=c["content"],
                metadata={
                    "chunk_id": f"{c['source']}::{c['chunk_index']}",
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "page_number": c.get("page_number"),
                    "section": c.get("section"),
                },
            )
            for c in raw_chunks
        ]
        if not docs:
            _BM25_INSTANCE = None
            _BM25_REVISION = revision
            raise vector_store.VectorStoreNotReady("RAG vector store contains no indexed chunks.")
        _BM25_INSTANCE = BM25Retriever.from_documents(docs, k=k)
        _BM25_REVISION = revision
    else:
        _BM25_INSTANCE.k = k
    return _BM25_INSTANCE


def clear_bm25_cache() -> None:
    """Clear cached BM25 instance (e.g. after re-ingestion)."""
    global _BM25_INSTANCE, _BM25_REVISION
    _BM25_INSTANCE = None
    _BM25_REVISION = None


def get_hybrid_retriever(k: int = 5, weights: list[float] | None = None):
    """
    EnsembleRetriever combining BM25 and PGVector via Reciprocal Rank Fusion (RRF).
    Uses id_key='chunk_id' to prevent deduplicating chunks with identical text across files.
    """
    from langchain_classic.retrievers import EnsembleRetriever
    vector_retriever = VectorStoreRetriever(top_k=k)
    bm25_retriever = get_bm25_retriever(k=k)
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=weights or [0.5, 0.5],
        id_key="chunk_id",
    )


def get_retriever(
    mode: str = "vector",
    top_k: int = 5,
    use_reranker: bool = False,
    reranker_top_n: int = 5,
) -> BaseRetriever:
    """Factory function for retrievers."""
    if mode == "hybrid":
        base = get_hybrid_retriever(k=top_k)
    else:
        base = VectorStoreRetriever(top_k=top_k)

    if use_reranker:
        from app.core.reranker import get_reranked_retriever
        return get_reranked_retriever(base, top_n=reranker_top_n)
    return base


def retrieve_and_gate(
    query: str,
    top_k: int = 5,
    mode: str = "vector",
    use_reranker: bool = False,
    reranker_top_n: int = 5,
) -> tuple[list[dict], bool]:
    """
    Retrieves documents according to mode ('vector' or 'hybrid') and enforces cosine gate.
    Optionally applies cross-encoder reranking down to reranker_top_n.

    Returns:
        (matches, is_refused):
            - matches: list of chunk dicts {content, source, chunk_index, score, page_number, section}
            - is_refused: True if top vector similarity score is below rag_min_relevance_score
    """
    vector_matches = vector_store.similarity_search(query, k=top_k)
    gated_vector_matches = [
        m for m in vector_matches
        if m["score"] >= settings.rag_min_relevance_score
    ]

    # Gating rule: cosine score from vector branch determines if there is sufficient evidence.
    if not gated_vector_matches:
        return [], True

    if mode == "vector":
        candidates = gated_vector_matches
    elif mode == "hybrid":
        hybrid_retriever = get_hybrid_retriever(k=top_k)
        docs = hybrid_retriever.invoke(query)
        docs = docs[:top_k]

        score_lookup = {
            (m["source"], m["chunk_index"]): m["score"]
            for m in vector_matches
        }
        candidate_keys = [
            (doc.metadata["source"], doc.metadata["chunk_index"])
            for doc in docs
        ]
        missing_scores = vector_store.score_chunks(
            query,
            [key for key in candidate_keys if key not in score_lookup],
        )
        score_lookup.update(missing_scores)

        candidates = []
        for doc in docs:
            chunk_key = (doc.metadata["source"], doc.metadata["chunk_index"])
            cosine_score = score_lookup.get(chunk_key)
            if cosine_score is None or cosine_score < settings.rag_min_relevance_score:
                continue
            candidates.append({
                "content": doc.page_content,
                "source": doc.metadata["source"],
                "chunk_index": doc.metadata["chunk_index"],
                "score": float(cosine_score),
                "page_number": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
            })
        if not candidates:
            return [], True
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    if not use_reranker or len(candidates) <= 1:
        return candidates[:reranker_top_n if use_reranker else top_k], False

    # Apply cross-encoder reranking
    from app.core.reranker import rerank_documents
    doc_candidates = [
        Document(
            page_content=c["content"],
            metadata={
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
                "page_number": c.get("page_number"),
                "section": c.get("section"),
            },
        )
        for c in candidates
    ]
    reranked_docs = rerank_documents(query, doc_candidates, top_n=reranker_top_n)

    reranked_matches = [
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
    ]
    return reranked_matches, False
