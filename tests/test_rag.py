import sys
import types
from unittest.mock import patch

from app.api import rag
from app.schemas.rag import RagAskRequest


class FakeModel:
    async def ainvoke(self, prompt, config):
        return types.SimpleNamespace(content="Nhân viên được nghỉ 12 ngày.")


def run_inline(coroutine):
    """Run these no-I/O async paths without opening a Windows event-loop socket."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Test unexpectedly suspended on asynchronous I/O")


async def inline_threadpool(function, *args):
    return function(*args)


def test_rag_abstains_when_retrieval_has_no_evidence():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: []
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "run_in_threadpool", inline_threadpool)):
        response = run_inline(rag.ask_rag(RagAskRequest(question="Chính sách nghỉ phép?")))
    assert response.answer == rag.NO_ANSWER
    assert response.sources == []


def test_rag_abstains_when_top_match_is_below_configured_threshold():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "Nội dung không liên quan", "source": "other.md", "chunk_index": 0, "score": 0.44
    }]
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_llm", side_effect=AssertionError("LLM must not run without evidence"))):
        response = run_inline(rag.ask_rag(RagAskRequest(question="Thông tin ngoài tài liệu?")))
    assert response.answer == rag.NO_ANSWER
    assert response.sources == []


def test_rag_returns_answer_and_retrieved_source():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "12 ngày nghỉ phép mỗi năm", "source": "policy.md", "chunk_index": 0, "score": 0.8
    }]
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "get_llm", return_value=FakeModel()),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        response = run_inline(rag.ask_rag(RagAskRequest(question="Chính sách nghỉ phép?")))
    assert response.answer == "Nhân viên được nghỉ 12 ngày."
    assert response.sources[0].source == "policy.md"
    assert response.sources[0].score == 0.8
