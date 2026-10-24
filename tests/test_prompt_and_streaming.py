"""Unit tests for Langfuse Prompt Management fallback and SSE Streaming (TASK-AI12)."""
import json
import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.core.config import settings
from app.core.rag_prompt import build_rag_prompt, build_hardcoded_rag_prompt, NO_ANSWER
from app.main import app


def test_build_rag_prompt_fallback_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")

    prompt = build_rag_prompt(prompt_name="rag-answer", label="production")
    hardcoded = build_hardcoded_rag_prompt()

    # Verify template structures match fallback
    assert len(prompt.messages) == len(hardcoded.messages)
    assert prompt.messages[0].prompt.template == hardcoded.messages[0].prompt.template


@pytest.mark.asyncio
async def test_ask_rag_stream_no_answer_event():
    fake_matches = []
    transport = httpx.ASGITransport(app=app)

    with patch("app.api.rag.retrieve_and_gate", return_value=(fake_matches, True)):
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

    with patch("app.api.rag.retrieve_and_gate", return_value=(fake_matches, False)), \
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
