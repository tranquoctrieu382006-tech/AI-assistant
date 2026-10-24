"""Unit tests for Langfuse Prompt Management fallback and SSE Streaming (TASK-AI12)."""
import json
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.core.config import settings
from app.core.rag_prompt import (
    build_rag_prompt,
    build_hardcoded_rag_prompt,
    build_rag_text_prompt,
    NO_ANSWER,
)
from app.main import app


def test_build_rag_prompt_fallback_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")

    prompt = build_rag_prompt(prompt_name="rag-answer", label="production")
    hardcoded = build_hardcoded_rag_prompt()

    # Verify template structures match fallback
    assert len(prompt.messages) == len(hardcoded.messages)
    assert prompt.messages[0].prompt.template == hardcoded.messages[0].prompt.template
    assert "dữ liệu không đáng tin cậy" in build_rag_text_prompt().messages[0].prompt.template


def test_build_rag_prompt_converts_langfuse_chat_prompt(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")

    class FakePrompt:
        is_fallback = False

        def get_langchain_prompt(self):
            return [
                ("system", "Ground answers in the context."),
                ("human", "{context}\n{question}\n{format_instructions}"),
            ]

    client = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    client.get_prompt.return_value = FakePrompt()
    with patch("app.core.tracing.get_langfuse_client", return_value=client):
        prompt = build_rag_prompt()

    client.get_prompt.assert_called_once()
    assert client.get_prompt.call_args.kwargs["type"] == "chat"
    rendered = prompt.format_messages(context="evidence", question="question")
    assert "evidence" in rendered[1].content
    assert "question" in rendered[1].content


def test_structured_rag_chain_is_configured_to_retry_parser_errors():
    from langchain_core.runnables import RunnableLambda
    from app.core.rag_prompt import build_rag_chain

    chain = build_rag_chain(
        RunnableLambda(lambda _prompt: "{}"),
        prompt_template=build_hardcoded_rag_prompt(),
    )
    assert type(chain).__name__ == "RunnableRetry"


@pytest.mark.asyncio
async def test_ask_rag_stream_no_answer_event():
    fake_matches = []
    transport = httpx.ASGITransport(app=app)

    with patch("app.api.rag.rewrite_question_if_needed", new=AsyncMock(side_effect=lambda q, _h: q)), \
         patch("app.api.rag.run_in_threadpool", new=AsyncMock(return_value=(fake_matches, True))):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rag/ask/stream",
                headers={"X-API-Key": settings.app_api_key},
                json={"question": "Câu hỏi không có thông tin?"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")

            body = resp.text
            assert 'data: {"type": "no_answer"' in body
            assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_ask_rag_stream_emits_sources_and_tokens():
    fake_matches = [
        {"content": "Nội dung tài liệu", "source": "01_quy_che_lam_viec.md", "chunk_index": 0, "score": 0.8}
    ]

    async def fake_astream(*args, **kwargs):
        for word in ["Nhân ", "viên ", "nghỉ ", "12 ngày."]:
            mock_chunk = AsyncMock()
            mock_chunk.content = word
            yield mock_chunk

    transport = httpx.ASGITransport(app=app)

    with patch("app.api.rag.rewrite_question_if_needed", new=AsyncMock(side_effect=lambda q, _h: q)), \
         patch("app.api.rag.run_in_threadpool", new=AsyncMock(return_value=(fake_matches, False))), \
         patch("langchain_core.runnables.base.RunnableSequence.astream", side_effect=fake_astream):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rag/ask/stream",
                headers={"X-API-Key": settings.app_api_key},
                json={"question": "Nhân viên nghỉ mấy ngày?"},
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = resp.text

            assert 'data: {"type": "sources"' in body
            assert 'data: {"type": "token", "token": "Nhân "}' in body
            assert 'data: {"type": "token", "token": "12 ngày."}' in body
            assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_streaming_model_refusal_does_not_emit_sources():
    fake_matches = [
        {"content": "Gần chủ đề nhưng thiếu dữ kiện", "source": "policy.md", "chunk_index": 0, "score": 0.8}
    ]

    async def fake_astream(*args, **kwargs):
        chunk = AsyncMock()
        chunk.content = NO_ANSWER
        yield chunk

    transport = httpx.ASGITransport(app=app)
    with patch("app.api.rag.rewrite_question_if_needed", new=AsyncMock(side_effect=lambda q, _h: q)), \
         patch("app.api.rag.run_in_threadpool", new=AsyncMock(return_value=(fake_matches, False))), \
         patch("langchain_core.runnables.base.RunnableSequence.astream", side_effect=fake_astream):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/rag/ask/stream",
                headers={"X-API-Key": settings.app_api_key},
                json={"question": "Thông tin còn thiếu là gì?"},
            )
    assert '"type": "sources"' not in resp.text
    assert '"type": "no_answer"' in resp.text
