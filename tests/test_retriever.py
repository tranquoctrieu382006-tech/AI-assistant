"""Unit tests for the unified retrieval abstraction."""
import pytest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.core.config import settings
import app.core.retriever as retriever_module
from app.core.retriever import (
    _bm25_tokens,
    _hybrid_fusion,
    get_bm25_retriever,
    retrieve_and_gate,
    clear_bm25_cache,
)


def test_bm25_tokens_ignore_case_and_normalize_unicode():
    assert _bm25_tokens("NghỈ PHÉP") == _bm25_tokens("nghỉ phép")
    assert _bm25_tokens("ngày, lỗi?") == ["ngày", "lỗi"]


def test_retrieve_and_gate_abstains_when_vector_below_cutoff(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)
    fake_matches = [
        {
            "content": "Chế độ bảo mật",
            "source": "02_quy_che_bao_mat.md",
            "chunk_index": 0,
            "score": 0.42,
        }
    ]
    with patch("app.core.vector_store.similarity_search", return_value=fake_matches):
        matches, is_refused = retrieve_and_gate("thai sản", top_k=5, mode="vector")

    assert is_refused is True
    assert matches == []


def test_retrieve_and_gate_passes_when_vector_above_cutoff(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)
    fake_matches = [
        {
            "content": "Thời gian làm việc từ 08:30 đến 17:30",
            "source": "01_quy_che_lam_viec.md",
            "chunk_index": 0,
            "score": 0.78,
            "section": "Giờ làm việc",
        }
    ]
    with patch("app.core.vector_store.similarity_search", return_value=fake_matches):
        matches, is_refused = retrieve_and_gate("giờ làm việc", top_k=5, mode="vector")

    assert is_refused is False
    assert len(matches) == 1
    assert matches[0]["score"] == 0.78


def test_hybrid_fusion_keeps_same_text_from_distinct_sources():
    text = "Trách nhiệm thực hiện: Toàn thể cán bộ nhân viên."
    bm25 = [
        Document(page_content=text, metadata={"source": "a.md", "chunk_index": 1}),
        Document(page_content=text, metadata={"source": "b.md", "chunk_index": 1}),
    ]
    vector = [
        {"content": text, "source": "a.md", "chunk_index": 1},
        {"content": text, "source": "b.md", "chunk_index": 1},
    ]
    docs = _hybrid_fusion(bm25, vector, top_k=2)
    assert {(doc.metadata["source"], doc.metadata["chunk_index"]) for doc in docs} == {
        ("a.md", 1), ("b.md", 1)
    }


def test_hybrid_gate_scores_and_filters_bm25_only_candidates(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)
    bm25_doc = Document(
        page_content="lexical-only weak result",
        metadata={"source": "bm25.md", "chunk_index": 4, "page_number": None, "section": None},
    )
    bm25 = MagicMock()
    bm25.invoke.return_value = [bm25_doc]

    with (
        patch("app.core.vector_store.embed_query", return_value=[0.1, 0.2]) as embed,
            patch("app.core.vector_store.similarity_search", return_value=[]),
        patch("app.core.vector_store.score_chunks", return_value={("bm25.md", 4): 0.31}) as score_chunks,
        patch("app.core.retriever.get_bm25_retriever", return_value=bm25),
    ):
        matches, refused = retrieve_and_gate("query", top_k=1, mode="hybrid")

    score_chunks.assert_called_once_with("query", [("bm25.md", 4)], embedding=[0.1, 0.2])
    embed.assert_called_once_with("query")
    assert refused is True
    assert matches == []


def test_hybrid_gate_can_keep_bm25_candidate_when_vector_top_k_is_empty(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)
    bm25_doc = Document(
        page_content="lexical result with strong cosine score",
        metadata={"source": "bm25.md", "chunk_index": 4, "page_number": None, "section": None},
    )
    bm25 = MagicMock()
    bm25.invoke.return_value = [bm25_doc]
    with (
        patch("app.core.vector_store.embed_query", return_value=[0.1, 0.2]),
        patch("app.core.vector_store.similarity_search", return_value=[]),
        patch("app.core.vector_store.score_chunks", return_value={("bm25.md", 4): 0.8}),
        patch("app.core.retriever.get_bm25_retriever", return_value=bm25),
    ):
        matches, refused = retrieve_and_gate("query", top_k=1, mode="hybrid")

    assert refused is False
    assert len(matches) == 1
    assert matches[0]["source"] == "bm25.md"
    assert matches[0]["score"] == 0.8


def test_bm25_cache_refreshes_when_database_corpus_revision_changes(monkeypatch):
    clear_bm25_cache()
    monkeypatch.setattr(retriever_module, "_REVISION_CACHE_TTL_SECONDS", 0)
    revisions = iter([(2, 10), (3, 13), (4, 16), (4, 16)])
    chunks = [
        {"content": "first indexed chunk", "source": "a.md", "chunk_index": 0},
        {"content": "second indexed chunk", "source": "a.md", "chunk_index": 1},
    ]
    created = []

    class FakeBM25Instance:
        def __init__(self, docs, k):
            self.docs = docs
            self.k = k

        def invoke(self, query):
            return self.docs

    class FakeBM25:
        @classmethod
        def from_documents(cls, docs, k, preprocess_func):
            instance = FakeBM25Instance(docs, k)
            instance.preprocess_func = preprocess_func
            created.append(instance)
            return instance

    fake_module = __import__("types").SimpleNamespace(BM25Retriever=FakeBM25)
    with (
        patch("app.core.vector_store.get_corpus_revision", side_effect=lambda: next(revisions)),
        patch("app.core.vector_store.get_all_chunks", return_value=chunks) as get_chunks,
        patch.dict("sys.modules", {"langchain_community.retrievers": fake_module}),
    ):
        first = get_bm25_retriever(k=3)
        second = get_bm25_retriever(k=4)

    assert first is not second
    assert len(created) == 2
    assert second.k == 4
    assert created[0].k == 3
    assert get_chunks.call_count == 2
    clear_bm25_cache()


def test_corpus_revision_probe_is_cached_for_short_ttl():
    clear_bm25_cache()
    with patch("app.core.vector_store.get_corpus_revision", return_value=(9, 12)) as revision:
        assert retriever_module._get_corpus_revision() == (9, 12)
        assert retriever_module._get_corpus_revision() == (9, 12)
    assert revision.call_count == 1
    clear_bm25_cache()


def test_concurrent_first_bm25_requests_build_index_once():
    clear_bm25_cache()
    created = []

    class FakeBM25Instance:
        def __init__(self, docs, k):
            self.docs, self.k = docs, k

    class FakeBM25:
        @classmethod
        def from_documents(cls, docs, k, preprocess_func):
            instance = FakeBM25Instance(docs, k)
            created.append(instance)
            return instance

    fake_module = __import__("types").SimpleNamespace(BM25Retriever=FakeBM25)
    chunks = [{"content": "one", "source": "a.md", "chunk_index": 0}]
    with (
        patch("app.core.vector_store.get_corpus_revision", return_value=(1, 1)),
        patch("app.core.vector_store.get_all_chunks", return_value=chunks) as get_chunks,
        patch.dict("sys.modules", {"langchain_community.retrievers": fake_module}),
    ):
        with ThreadPoolExecutor(max_workers=8) as executor:
            retrievers = list(executor.map(lambda _: get_bm25_retriever(k=5), range(8)))

    assert len(created) == 1
    assert get_chunks.call_count == 1
    assert len({id(retriever) for retriever in retrievers}) == 8
    clear_bm25_cache()


def test_bm25_local_reference_survives_concurrent_cache_clear():
    import copy

    clear_bm25_cache()
    copying = Event()
    allow_copy = Event()
    original_copy = copy.copy

    class FakeBM25Instance:
        def __init__(self, docs, k):
            self.docs, self.k = docs, k

    class FakeBM25:
        @classmethod
        def from_documents(cls, docs, k, preprocess_func):
            return FakeBM25Instance(docs, k)

    fake_module = __import__("types").SimpleNamespace(BM25Retriever=FakeBM25)

    def slow_copy(value):
        copying.set()
        assert allow_copy.wait(timeout=3)
        return original_copy(value)

    with (
        patch("app.core.vector_store.get_corpus_revision", return_value=(1, 1)),
        patch("app.core.vector_store.get_all_chunks", return_value=[{
            "content": "one", "source": "a.md", "chunk_index": 0,
        }]),
        patch("app.core.retriever.copy.copy", side_effect=slow_copy),
        patch.dict("sys.modules", {"langchain_community.retrievers": fake_module}),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            retrieval = executor.submit(get_bm25_retriever, 5)
            assert copying.wait(timeout=3)
            clearing = executor.submit(clear_bm25_cache)
            clearing.result(timeout=1)
            allow_copy.set()
            assert retrieval.result(timeout=3).k == 5


def test_warm_bm25_cache_path_does_not_acquire_global_lock():
    class LockSpy:
        entered = 0

        def __enter__(self):
            self.entered += 1

        def __exit__(self, exc_type, exc, traceback):
            return False

    class CachedIndex:
        k = 20

    lock = LockSpy()
    with (
        patch.object(retriever_module, "_BM25_LOCK", lock),
        patch.object(retriever_module, "_BM25_INSTANCE", CachedIndex()),
        patch.object(retriever_module, "_BM25_REVISION", (1, 1)),
        patch("app.core.retriever._get_corpus_revision", return_value=(1, 1)),
    ):
        per_call = get_bm25_retriever(k=7)

    assert lock.entered == 0
    assert per_call.k == 7



def test_bm25_cache_is_partitioned_by_corpus():
    """Mỗi corpus phải có index BM25 riêng, không dùng lẫn toàn bộ corpus."""
    clear_bm25_cache()
    created = []

    class FakeBM25Instance:
        def __init__(self, docs, k):
            self.docs, self.k = docs, k

    class FakeBM25:
        @classmethod
        def from_documents(cls, docs, k, preprocess_func):
            instance = FakeBM25Instance(docs, k)
            created.append(instance)
            return instance

    fake_module = __import__("types").SimpleNamespace(BM25Retriever=FakeBM25)
    chunks_by_corpus = {
        "demo": [{"content": "demo chunk", "source": "demo.md", "chunk_index": 0}],
        "user": [{"content": "user chunk", "source": "user.md", "chunk_index": 0}],
    }

    def fake_chunks(corpus=None):
        return chunks_by_corpus[corpus]

    def fake_revision(corpus=None):
        return (1, 1)

    with (
        patch("app.core.vector_store.get_corpus_revision", side_effect=fake_revision),
        patch("app.core.vector_store.get_all_chunks", side_effect=fake_chunks) as get_chunks,
        patch.dict("sys.modules", {"langchain_community.retrievers": fake_module}),
    ):
        demo = get_bm25_retriever(k=3, corpus="demo")
        user = get_bm25_retriever(k=3, corpus="user")
        demo_warm = get_bm25_retriever(k=5, corpus="demo")

    assert len(created) == 2
    assert [call.kwargs for call in get_chunks.call_args_list] == [
        {"corpus": "demo"},
        {"corpus": "user"},
    ]
    assert created[0].docs[0].page_content == "demo chunk"
    assert created[1].docs[0].page_content == "user chunk"
    assert demo is not user
    assert demo_warm.k == 5
    clear_bm25_cache()


def test_bm25_cache_is_bounded_and_invalidates_only_matching_corpora():
    """Per-session selectors are capped and a session delete keeps other indexes."""
    retriever_module.clear_bm25_cache()
    original_limit = retriever_module._BM25_MAX_CACHED_SELECTORS
    retriever_module._BM25_MAX_CACHED_SELECTORS = 2
    try:
        with retriever_module._BM25_LOCK:
            for key in (("demo",), ("demo", "user"), ("demo", "user", "session_x")):
                retriever_module._store_bm25(key, object(), (1, 1))

        assert list(retriever_module._BM25_INSTANCES) == [
            ("demo", "user"),
            ("demo", "user", "session_x"),
        ]

        retriever_module.clear_bm25_cache(corpus="session_x")

        assert list(retriever_module._BM25_INSTANCES) == [("demo", "user")]
        assert ("demo", "user", "session_x") not in retriever_module._BM25_REVISIONS
    finally:
        retriever_module._BM25_MAX_CACHED_SELECTORS = original_limit
        retriever_module.clear_bm25_cache()


def test_type_error_from_similarity_search_is_not_swallowed():
    """Lỗi lập trình (sai chữ ký) phải nổi lên thay vì âm thầm gọi lại API cũ."""
    with patch(
        "app.core.vector_store.similarity_search",
        side_effect=TypeError("unexpected keyword argument"),
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            retrieve_and_gate("query", top_k=1, mode="vector")


def test_rerank_documents_reorders_by_score():
    from app.core.reranker import rerank_documents

    docs = [
        Document(page_content="Tài liệu A", metadata={"source": "a.md", "chunk_index": 0, "score": 0.6}),
        Document(page_content="Tài liệu B", metadata={"source": "b.md", "chunk_index": 1, "score": 0.7}),
        Document(page_content="Tài liệu C", metadata={"source": "c.md", "chunk_index": 2, "score": 0.8}),
    ]

    mock_model = MagicMock()
    # Mock predict: gán điểm sao cho Tài liệu A cao nhất (0.95), C (0.5), B (0.2)
    mock_model.predict.return_value = [0.95, 0.2, 0.5]

    with patch("app.core.reranker.get_cross_encoder", return_value=mock_model):
        reranked = rerank_documents("câu hỏi", docs, top_n=2)

    assert len(reranked) == 2
    assert reranked[0].page_content == "Tài liệu A"
    assert reranked[0].metadata["rerank_score"] == 0.95
    assert reranked[1].page_content == "Tài liệu C"
    assert reranked[1].metadata["rerank_score"] == 0.5


def test_retrieve_and_gate_with_reranker():
    fake_matches = [
        {"content": "Doc 1", "source": "1.md", "chunk_index": 0, "score": 0.7},
        {"content": "Doc 2", "source": "2.md", "chunk_index": 1, "score": 0.8},
        {"content": "Doc 3", "source": "3.md", "chunk_index": 2, "score": 0.9},
    ]

    mock_model = MagicMock()
    # Reranker đảo thứ tự: Doc 1 thành điểm cao nhất
    mock_model.predict.return_value = [0.99, 0.5, 0.1]

    with patch("app.core.vector_store.similarity_search", return_value=fake_matches), \
         patch("app.core.reranker.get_cross_encoder", return_value=mock_model):
        matches, is_refused = retrieve_and_gate(
            "test query", top_k=3, mode="vector", use_reranker=True, reranker_top_n=2
        )

    assert is_refused is False
    assert len(matches) == 2
    assert matches[0]["content"] == "Doc 1"
    assert matches[0]["rerank_score"] == 0.99
    assert matches[1]["content"] == "Doc 2"
    assert matches[1]["rerank_score"] == 0.5


def test_reranker_uses_larger_candidate_pool_then_returns_requested_top_k(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.0)
    fake_matches = [
        {"content": f"Doc {i}", "source": f"{i}.md", "chunk_index": i, "score": 0.8}
        for i in range(20)
    ]
    model = MagicMock()
    model.predict.return_value = list(range(20))
    with patch("app.core.vector_store.similarity_search", return_value=fake_matches) as search, \
         patch("app.core.reranker.get_cross_encoder", return_value=model):
        matches, refused = retrieve_and_gate(
            "question", top_k=5, mode="vector", use_reranker=True,
            reranker_top_n=5, reranker_pool_size=20,
        )
    search.assert_called_once_with("question", k=20)
    assert refused is False
    assert len(matches) == 5


def test_hybrid_reranker_candidate_pairs_are_limited_to_configured_pool(monkeypatch):
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.0)
    vector_matches = [
        {"content": f"vector {i}", "source": f"v{i}.md", "chunk_index": 0, "score": 0.8}
        for i in range(3)
    ]
    bm25 = MagicMock()
    bm25.invoke.return_value = [
        Document(page_content=f"lexical {i}", metadata={
            "source": f"b{i}.md", "chunk_index": 0,
        })
        for i in range(3)
    ]
    model = MagicMock()
    model.predict.return_value = [0.9] * 6
    with (
        patch("app.core.vector_store.embed_query", return_value=[0.1, 0.2]),
        patch("app.core.vector_store.similarity_search", return_value=vector_matches),
        patch("app.core.vector_store.score_chunks", return_value={
            (f"b{i}.md", 0): 0.8 for i in range(3)
        }),
        patch("app.core.retriever.get_bm25_retriever", return_value=bm25),
        patch("app.core.reranker.get_cross_encoder", return_value=model),
    ):
        matches, refused = retrieve_and_gate(
            "question", top_k=2, mode="hybrid", use_reranker=True,
            reranker_top_n=2, reranker_pool_size=3,
        )

    assert refused is False
    assert len(matches) == 2
    assert len(model.predict.call_args.args[0]) == 3


def test_dual_retrieval_merges_matches_taking_max_score(monkeypatch):
    """Khi raw_query được cung cấp và khác với rewritten query, Dual Retrieval lấy điểm cosine cao nhất."""
    monkeypatch.setattr(settings, "rag_min_relevance_score", 0.5)

    matches_query = [
        {"content": "Chunk A", "source": "a.md", "chunk_index": 0, "score": 0.55},
        {"content": "Chunk B", "source": "b.md", "chunk_index": 0, "score": 0.40},
    ]
    matches_raw = [
        {"content": "Chunk A", "source": "a.md", "chunk_index": 0, "score": 0.65},
        {"content": "Chunk C", "source": "c.md", "chunk_index": 0, "score": 0.70},
    ]

    def fake_search(q, **kwargs):
        if "khong dau" in q:
            return matches_raw
        return matches_query

    with patch("app.core.vector_store.similarity_search", side_effect=fake_search):
        matches, refused = retrieve_and_gate(
            "câu hỏi có dấu", 5, "vector", False, 5, None, None, "cau hoi khong dau"
        )

    assert refused is False
    scores = {(m["source"], m["chunk_index"]): m["score"] for m in matches}
    assert scores[("a.md", 0)] == 0.65
    assert ("c.md", 0) in scores
    assert ("b.md", 0) not in scores
