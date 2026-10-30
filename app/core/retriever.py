"""Vector and BM25 retrieval, score gating, and optional cross-encoder reranking."""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import OrderedDict, defaultdict

from langchain_core.documents import Document

from app.core import vector_store
from app.core.config import settings
from app.core.text_utils import tokenize_with_unaccented

logger = logging.getLogger(__name__)

_BM25_INSTANCE = None
_BM25_REVISION: tuple[int, int] | None = None
# Corpus-specific indexes are keyed by the normalised corpus selector. The
# None key keeps the legacy all-corpus index in the two globals above.
_BM25_INSTANCES: OrderedDict[tuple[str, ...], object] = OrderedDict()
_BM25_REVISIONS: dict[tuple[str, ...], tuple[int, int]] = {}
# A per-session selector embeds copies of demo/user documents. Bound the number
# of selectors kept warm so a long-lived process cannot grow without limit.
_BM25_MAX_CACHED_SELECTORS = 32
_BM25_LOCK = threading.RLock()
_REVISION_LOCK = threading.Lock()
_REVISION_CACHE_TTL_SECONDS = settings.bm25_revision_ttl_seconds
_REVISION_CACHE: dict[tuple[str, ...] | None, tuple[tuple[int, int], float]] = {}
_RRF_CONSTANT = 60
# Weighted RRF. BM25 tokens are unaccented (homophones collapse), so lexical
# ranks are noisier than before; vector-heavy fusion preserves Hit@1 while
# keeping BM25's recall contribution. Measured on the 65-case dev set:
# eval/rag_variants_dev.json (0.2/0.8 -> 35/41 Hit@1, 0.5/0.5 -> 26/41).
_RRF_WEIGHTS = (0.2, 0.8)  # BM25, vector


def _corpus_key(corpus: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Return a stable cache key for a corpus selector."""
    if corpus is None:
        return None
    if isinstance(corpus, str):
        return (corpus,)
    return tuple(sorted({str(item) for item in corpus}))


def _cached_bm25(
    key: tuple[str, ...] | None,
) -> tuple[object | None, tuple[int, int] | None]:
    if key is None:
        return _BM25_INSTANCE, _BM25_REVISION
    with _BM25_LOCK:
        instance = _BM25_INSTANCES.get(key)
        if instance is not None:
            _BM25_INSTANCES.move_to_end(key)
        return instance, _BM25_REVISIONS.get(key)


def _store_bm25(
    key: tuple[str, ...] | None,
    instance: object | None,
    revision: tuple[int, int],
) -> None:
    global _BM25_INSTANCE, _BM25_REVISION
    if key is None:
        _BM25_INSTANCE = instance
        _BM25_REVISION = revision
        return

    if instance is None:
        _BM25_INSTANCES.pop(key, None)
    else:
        _BM25_INSTANCES[key] = instance
        _BM25_INSTANCES.move_to_end(key)
        while len(_BM25_INSTANCES) > _BM25_MAX_CACHED_SELECTORS:
            evicted_key, _ = _BM25_INSTANCES.popitem(last=False)
            _BM25_REVISIONS.pop(evicted_key, None)
    _BM25_REVISIONS[key] = revision


def _bm25_tokens(text: str) -> list[str]:
    """Normalize Unicode/case and emit unaccented tokens for BM25.

    Delegates to the shared tokenizer so documents and queries use exactly the
    same normalization; this is what makes unaccented queries match accented
    documents in hybrid mode.
    """
    return tokenize_with_unaccented(text)


def get_bm25_retriever(
    k: int = 5,
    corpus: str | list[str] | tuple[str, ...] | None = None,
):
    """Return a per-call BM25 retriever over the requested corpus only."""
    key = _corpus_key(corpus)
    # A DB revision check is needed for each hybrid query, but it must not hold
    # the process-wide cache lock on the warm path.
    revision = _get_corpus_revision(corpus)
    instance, cached_revision = _cached_bm25(key)
    if instance is None or cached_revision != revision:
        with _BM25_LOCK:
            # Recheck after acquiring the lock; another request may have built
            # the current revision while this request waited.
            revision = _get_corpus_revision(corpus)
            instance, cached_revision = _cached_bm25(key)
            if instance is None or cached_revision != revision:
                from langchain_community.retrievers import BM25Retriever

                chunks = (
                    vector_store.get_all_chunks()
                    if key is None
                    else vector_store.get_all_chunks(corpus=corpus)
                )
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
                    _store_bm25(key, None, revision)
                    raise vector_store.VectorStoreNotReady(
                        "RAG vector store contains no indexed chunks for the requested corpus."
                    )
                instance = BM25Retriever.from_documents(
                    docs,
                    k=max(k, 1),
                    preprocess_func=_bm25_tokens,
                )
                _store_bm25(key, instance, revision)
    # Hold a local strong reference before copying. A concurrent cache clear
    # may replace the global reference, but cannot invalidate this request's
    # immutable index or make copy.copy(None).
    if instance is None:
        raise vector_store.VectorStoreNotReady("BM25 index is unavailable.")
    per_call = copy.copy(instance)
    per_call.k = k
    return per_call


def clear_bm25_cache(corpus: str | list[str] | tuple[str, ...] | None = None) -> None:
    """Clear cached BM25 state.

    With no argument every cached selector is dropped (tests and explicit full
    reindexing). When a corpus is supplied, only selectors containing that
    corpus are removed, so uploading one document does not invalidate every
    unrelated session index.
    """
    global _BM25_INSTANCE, _BM25_REVISION
    targets = set(_corpus_key(corpus) or ())
    with _BM25_LOCK:
        if corpus is None:
            _BM25_INSTANCE = None
            _BM25_REVISION = None
            _BM25_INSTANCES.clear()
            _BM25_REVISIONS.clear()
        else:
            # The legacy all-corpus index includes every corpus.
            _BM25_INSTANCE = None
            _BM25_REVISION = None
            for cached_key in list(_BM25_INSTANCES):
                if targets & set(cached_key):
                    _BM25_INSTANCES.pop(cached_key, None)
            for cached_key in list(_BM25_REVISIONS):
                if targets & set(cached_key):
                    _BM25_REVISIONS.pop(cached_key, None)
    with _REVISION_LOCK:
        _REVISION_CACHE.clear()


def _get_corpus_revision(
    corpus: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[int, int]:
    """Cache DB revision checks briefly; external ingestion appears within 5 seconds."""
    key = _corpus_key(corpus)
    now = time.monotonic()
    cached = _REVISION_CACHE.get(key)
    if cached is not None and now - cached[1] < _REVISION_CACHE_TTL_SECONDS:
        return cached[0]
    with _REVISION_LOCK:
        now = time.monotonic()
        cached = _REVISION_CACHE.get(key)
        if cached is None or now - cached[1] >= _REVISION_CACHE_TTL_SECONDS:
            revision = (
                vector_store.get_corpus_revision()
                if key is None
                else vector_store.get_corpus_revision(corpus=corpus)
            )
            cached = (revision, time.monotonic())
            _REVISION_CACHE[key] = cached
        return cached[0]


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
                corpus = item.metadata.get("corpus")
                doc = item
            else:
                source = item["source"]
                chunk_index = item["chunk_index"]
                corpus = item.get("corpus")
                doc = Document(
                    page_content=item["content"],
                    metadata={
                        "source": source,
                        "chunk_index": chunk_index,
                        "corpus": corpus,
                        "page_number": item.get("page_number"),
                        "section": item.get("section"),
                    },
                )
            key = (corpus, source, chunk_index)
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
    reranker_pool_size: int | None = None,
    corpus: str | list[str] | tuple[str, ...] | None = None,
    raw_query: str | None = None,
) -> tuple[list[dict], bool]:
    """Retrieve evidence, enforce per-chunk cosine gating, and optionally rerank."""
    if mode not in {"vector", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")

    configured_pool_size = reranker_pool_size or settings.reranker_pool_size
    candidate_k = max(top_k, configured_pool_size) if use_reranker else top_k
    search_kwargs = {"corpus": corpus} if corpus is not None else {}
    if mode == "hybrid":
        query_vector = vector_store.embed_query(query)
        vector_matches = vector_store.similarity_search(
            query, k=candidate_k, embedding=query_vector, **search_kwargs
        )
    else:
        query_vector = None
        vector_matches = vector_store.similarity_search(query, k=candidate_k, **search_kwargs)

    # Dual Retrieval: If raw_query differs from rewritten query, also retrieve using raw_query
    # and take max cosine score per chunk to prevent entity distortion or diacritic misinterpretation.
    if raw_query and raw_query.strip().casefold() != query.strip().casefold():
        try:
            raw_matches = vector_store.similarity_search(raw_query, k=candidate_k, **search_kwargs)
            merged: dict[tuple[str, int], dict] = {}
            for m in vector_matches:
                merged[(m["source"], m["chunk_index"])] = copy.deepcopy(m)
            for m in raw_matches:
                key = (m["source"], m["chunk_index"])
                if key not in merged:
                    merged[key] = copy.deepcopy(m)
                else:
                    merged[key]["score"] = max(merged[key]["score"], m["score"])
            vector_matches = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:candidate_k]
        except Exception as exc:
            logger.warning("Dual retrieval for raw query failed; falling back to primary query (%s)", exc)

    gated_vector_matches = [
        match for match in vector_matches
        if match["score"] >= settings.rag_min_relevance_score
    ]
    if mode == "vector" and not gated_vector_matches:
        return [], True

    if mode == "vector":
        candidates = gated_vector_matches
    else:
        bm25_docs = get_bm25_retriever(k=candidate_k, corpus=corpus).invoke(query)
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

    # Hybrid fusion can contain the union of both ranked lists (up to 2*k).
    # Bound cross-encoder work by the explicitly configured candidate pool.
    if use_reranker:
        candidates = candidates[:candidate_k]

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
