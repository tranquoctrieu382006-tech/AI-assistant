"""Unit tests for the unified retrieval abstraction."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.core.config import settings
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


def test_bm25_cache_refreshes_when_database_corpus_revision_changes():
    clear_bm25_cache()
    revisions = iter([(2, 10), (3, 13)])
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
    assert created[0].k == 20
    assert get_chunks.call_count == 2
    clear_bm25_cache()



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
