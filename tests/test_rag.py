import asyncio
import json
import sys
import types
import httpx
import pytest
from contextlib import contextmanager
from unittest.mock import patch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.api import rag
from app.schemas.rag import RagAskRequest
from fastapi import HTTPException
from app.main import app


class FakeModel(RunnableLambda):
    def __init__(self, content="Nhân viên được nghỉ 12 ngày."):
        self.content = content
        self.prompt = None
        self.config = None

        async def invoke(prompt_value, config):
            self.prompt = prompt_value.to_messages()
            self.config = config
            return AIMessage(content=json.dumps({
                "answer": self.content,
                "abstain": self.content == rag.NO_ANSWER,
            }, ensure_ascii=False))

        super().__init__(invoke)


def run_inline(coroutine):
    """Run mocked no-I/O endpoint paths on an event loop."""
    return asyncio.run(coroutine)


async def inline_threadpool(function, *args):
    return function(*args)


def test_rag_abstains_when_retrieval_has_no_evidence():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: []
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "run_in_threadpool", inline_threadpool)):
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
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
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
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
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
            response = run_inline(rag.ask_rag(RagAskRequest(question="Chính sách nghỉ phép?")))
    assert response.answer == "Nhân viên được nghỉ 12 ngày."
    assert response.sources[0].source == "policy.md"
    assert response.sources[0].score == 0.8


def test_rag_filters_each_low_score_chunk_and_uses_separate_messages():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [
        {"content": "good", "source": "good.md", "chunk_index": 0, "score": 0.8},
        {"content": "poor", "source": "poor.md", "chunk_index": 1, "score": 0.2},
    ]
    model = FakeModel()
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "get_llm", return_value=model),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
            response = run_inline(rag.ask_rag(RagAskRequest(question="Câu hỏi phù hợp?")))
    assert [source.source for source in response.sources] == ["good.md"]
    assert len(model.prompt) == 2
    assert "poor" not in model.prompt[1].content


def test_rag_clears_sources_when_model_abstains():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "unrelated", "source": "policy.md", "chunk_index": 0, "score": 0.8
    }]
    model = FakeModel(rag.NO_ANSWER)
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "get_llm", return_value=model),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
            response = run_inline(rag.ask_rag(RagAskRequest(question="Câu hỏi?")))
    assert response.answer == rag.NO_ANSWER
    assert response.sources == []


def test_rag_returns_pydantic_parsed_answer_and_honors_abstain_flag():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "Evidence", "source": "policy.md", "chunk_index": 0, "score": 0.8
    }]
    model = FakeModel("The generated field is valid.")
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "get_llm", return_value=model),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
            response = run_inline(rag.ask_rag(RagAskRequest(question="Question?")))

    assert response.answer == "The generated field is valid."
    assert '"answer"' in model.prompt[1].content
    assert '"abstain"' in model.prompt[1].content


def test_rag_rejects_invalid_structured_generation():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "Evidence", "source": "policy.md", "chunk_index": 0, "score": 0.8
    }]

    class InvalidModel(RunnableLambda):
        def __init__(self):
            async def invoke(prompt_value, config):
                return AIMessage(content="not valid structured output")

            super().__init__(invoke)

    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "get_llm", return_value=InvalidModel()),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
            try:
                run_inline(rag.ask_rag(RagAskRequest(question="Question?")))
            except HTTPException as exc:
                assert exc.status_code == 502
                assert exc.detail == "RAG answer generation failed"
            else:
                raise AssertionError("Expected invalid structured output to fail")


@pytest.mark.asyncio
async def test_stream_disconnect_persists_partial_turn_for_followup():
    from contextlib import contextmanager
    from types import SimpleNamespace

    session_store = rag.session_store
    session_store.clear_all_sessions()

    class FakeStreamChain:
        async def astream(self, *args, **kwargs):
            yield SimpleNamespace(content="Partial answer")

    class FakePrompt:
        def __or__(self, llm):
            return FakeStreamChain()

    async def inline_threadpool(function, *args):
        return function(*args)

    async def no_rewrite(question, history):
        return question

    @contextmanager
    def no_trace(*args, **kwargs):
        yield

    request = RagAskRequest(question="Question?", session_id="disconnected-stream")
    with (
        patch.object(rag, "rewrite_question_if_needed", no_rewrite),
        patch.object(rag, "run_in_threadpool", inline_threadpool),
        patch.object(rag, "retrieve_and_gate", return_value=([{
            "content": "Evidence", "source": "policy.md", "chunk_index": 0,
            "score": 0.8, "section": None, "page_number": None,
        }], False)),
        patch.object(rag, "build_rag_text_prompt", return_value=FakePrompt()),
        patch.object(rag, "get_rag_llm", return_value=object()),
        patch.object(rag, "get_langfuse_handler", return_value=None),
        patch.object(rag, "langfuse_observation", no_trace),
        patch.object(rag, "update_langfuse_observation"),
        patch.object(rag, "langfuse_context", no_trace),
    ):
        response = await rag.ask_rag_stream(request)
        stream = response.body_iterator
        first_event = await anext(stream)
        assert '"type": "sources"' in first_event
        try:
            await stream.athrow(asyncio.CancelledError())
        except asyncio.CancelledError:
            pass

    history = session_store.get_history("disconnected-stream")
    assert history[-2]["content"] == "Question?"
    assert history[-1]["content"] == "Partial answer"
    session_store.clear_all_sessions()


def test_rag_reports_retrieval_failure_without_leaking_exception_details():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: (_ for _ in ()).throw(ConnectionError("private detail"))
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "run_in_threadpool", inline_threadpool)):
        try:
            with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
                run_inline(rag.ask_rag(RagAskRequest(question="Chính sách nghỉ phép?")))
        except HTTPException as exc:
            assert exc.status_code == 503
            assert exc.detail == "RAG retrieval unavailable"
        else:
            raise AssertionError("Expected retrieval failure")


def test_rag_reports_generation_failure_separately():
    fake_store = types.ModuleType("app.core.vector_store")
    fake_store.similarity_search = lambda question, k: [{
        "content": "12 ngày nghỉ phép mỗi năm", "source": "policy.md", "chunk_index": 0, "score": 0.8
    }]
    with (patch.dict(sys.modules, {"app.core.vector_store": fake_store}),
          patch.object(rag, "run_in_threadpool", inline_threadpool),
          patch.object(rag, "get_llm", side_effect=RuntimeError("private provider detail")),
          patch.object(rag, "get_langfuse_handler", return_value=None)):
        try:
            with patch.object(rag.vector_store, "similarity_search", fake_store.similarity_search):
                run_inline(rag.ask_rag(RagAskRequest(question="Chính sách nghỉ phép?")))
        except HTTPException as exc:
            assert exc.status_code == 502
            assert exc.detail == "RAG answer generation failed"
        else:
            raise AssertionError("Expected generation failure")


@pytest.mark.asyncio
async def test_rag_request_validation_uses_http_boundary():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-API-Key": "test-api-key-with-more-than-32-characters"}
        short_question = await client.post(
            "/api/rag/ask", json={"question": "x"}, headers=headers
        )
        invalid_top_k = await client.post(
            "/api/rag/ask", json={"question": "Câu hỏi hợp lệ?", "top_k": 21}, headers=headers
        )
    assert short_question.status_code == 422
    assert invalid_top_k.status_code == 422


@pytest.mark.asyncio
async def test_rag_http_request_records_request_and_retrieval_spans():
    observations = []
    contexts = []
    handler = BaseCallbackHandler()
    model = FakeModel()

    @contextmanager
    def observation(name, **kwargs):
        observations.append(name)
        yield object()

    @contextmanager
    def trace_context(**kwargs):
        contexts.append(kwargs)
        yield

    matches = [{
        "content": "Thời hạn đổi trả là bảy ngày.",
        "source": "returns.md",
        "chunk_index": 1,
        "page_number": None,
        "section": "Đổi trả",
        "score": 0.9,
    }]
    with (patch.object(rag.vector_store, "similarity_search", return_value=matches),
          patch.object(rag, "get_llm", return_value=model),
          patch.object(rag, "get_langfuse_handler", return_value=handler),
          patch.object(rag, "langfuse_observation", observation),
          patch.object(rag, "langfuse_context", trace_context),
          patch.object(rag, "update_langfuse_observation")):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/rag/ask",
                json={"question": "Thời hạn đổi trả là bao lâu?", "session_id": "trace-test"},
                headers={"X-API-Key": "test-api-key-with-more-than-32-characters"},
            )

    assert response.status_code == 200
    assert response.json()["sources"][0]["section"] == "Đổi trả"
    assert observations == ["rag-request", "rag-retrieval"]
    assert contexts == [{"session_id": "trace-test", "tags": ["rag", "ai07"]}]
    assert model.prompt is not None
    assert "Đổi trả" in model.prompt[1].content
    assert model.config["callbacks"].handlers == [handler]
