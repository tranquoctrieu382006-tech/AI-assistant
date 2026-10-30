"""
Unit tests for app.core.orchestrator and /api/ask endpoint.
Kiểm tra:
- Tính độc lập kiến trúc (app/core không import app/api).
- Chốt chặn PII cá nhân con người ngay tại cửa ngõ cho mọi mode.
- Chốt chặn lệnh ghi phá hoại (kể cả có dấu chấm hỏi cuối câu).
- Cơ chế fallback an toàn 1 bước từ NLQ sang RAG khi unanswerable.
- Tuyệt đối không fallback sang RAG khi vi phạm chính sách / PII.
- Chuẩn hóa lỗi hạ tầng (503/504) kèm header X-Trace-Id, không nuốt thành 200.
"""

from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
import psycopg
import pytest

from app.core.config import settings
from app.core.nlq_graph import NlqRequestTimeout
from app.core.orchestrator import (
    AskUnifiedRequest,
    ask_orchestrated,
)
from app.main import app
from app.schemas.rag import RagAskResponse, RagSource

MOCK_TRACE_ID = "a" * 32


def test_core_orchestrator_does_not_import_api_modules():
    """Kiểm tra app/core/orchestrator.py tuân thủ Clean Architecture, không import app.api."""
    with open("app/core/orchestrator.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "app.api" not in content, "Phát hiện app/core/orchestrator.py phụ thuộc ngược vào app.api!"


@pytest.mark.asyncio
async def test_orchestrator_blocks_human_pii_upfront_in_all_modes():
    """Chặn câu hỏi PII cá nhân con người ngay tại cửa ngõ cho auto, rag và nlq modes."""
    with patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID):
        for mode in ["auto", "rag", "nlq"]:
            req = AskUnifiedRequest(
                question="Email của khách hàng VIP là gì?",
                mode=mode,
            )
            res = await ask_orchestrated(req)
            assert res.status == "restricted"
            assert res.error_code == "policy_denied"
            assert res.trace_id == MOCK_TRACE_ID
            assert "hạn chế để bảo vệ quyền riêng tư" in res.answer


@pytest.mark.asyncio
async def test_orchestrator_blocks_imperative_write_actions():
    """Chặn các câu lệnh ghi/xóa CSDL kể cả khi kết thúc bằng dấu chấm hỏi."""
    write_queries = [
        "Xóa toàn bộ đơn hàng?",
        "Hãy xóa toàn bộ đơn hàng đi?",
        "Delete from orders?",
        "DROP TABLE customers",
    ]
    for q in write_queries:
        req = AskUnifiedRequest(question=q, mode="auto")
        res = await ask_orchestrated(req)
        assert res.status == "unsupported_action"
        assert res.route == "unsupported_action"
        assert "chỉ đọc" in res.answer


@pytest.mark.asyncio
async def test_orchestrator_explicit_rag_mode():
    """Kiểm tra chế độ ép RAG gọi answer_rag_request thành công."""
    mock_rag_res = RagAskResponse(
        answer="Quy định bảo hành là 12 tháng.",
        sources=[RagSource(source="chinh_sach.md", chunk_index=0, score=0.88)],
    )
    with patch("app.core.orchestrator.answer_rag_request", new_callable=AsyncMock, return_value=mock_rag_res):
        req = AskUnifiedRequest(question="Chính sách bảo hành", mode="rag")
        res = await ask_orchestrated(req)
        assert res.route == "rag"
        assert res.status == "ok"
        assert "12 tháng" in res.answer
        assert len(res.sources) == 1


@pytest.mark.asyncio
async def test_orchestrator_explicit_nlq_mode():
    """Kiểm tra chế độ ép NLQ gọi ask_nlq qua threadpool và trả kết quả chuẩn."""
    mock_nlq_res = {
        "status": "ok",
        "answer": "Có 150 đơn hàng.",
        "sql": "SELECT COUNT(*) FROM orders;",
        "columns": ["count"],
        "rows": [{"count": 150}],
    }
    with patch("app.core.orchestrator.ask_nlq", return_value=mock_nlq_res):
        req = AskUnifiedRequest(question="Có bao nhiêu đơn hàng?", mode="nlq")
        res = await ask_orchestrated(req)
        assert res.route == "nlq"
        assert res.status == "ok"
        assert res.sql == "SELECT COUNT(*) FROM orders;"
        assert res.rows == [{"count": 150}]


@pytest.mark.asyncio
async def test_orchestrator_auto_nlq_out_of_scope_falls_back_to_rag_safely():
    """Khi NLQ out_of_scope và không chứa PII (liên hệ cơ sở), tự động fallback sang RAG."""
    mock_nlq_res = {
        "status": "out_of_scope",
        "error_code": "unanswerable",
        "answer": "Không tìm thấy dữ liệu.",
    }
    mock_rag_res = RagAskResponse(
        answer="Số điện thoại kho hàng là 028-12345678.",
        sources=[RagSource(source="kho_hang.md", chunk_index=0, score=0.82)],
    )
    with (
        patch("app.core.orchestrator.ask_nlq", return_value=mock_nlq_res),
        patch("app.core.orchestrator.answer_rag_request", new_callable=AsyncMock, return_value=mock_rag_res),
    ):
        req = AskUnifiedRequest(question="Số điện thoại kho hàng là bao nhiêu?", mode="auto")
        res = await ask_orchestrated(req)
        assert res.route == "rag"
        assert res.fallback_from == "nlq"
        assert res.status == "ok"
        assert "028-12345678" in res.answer


@pytest.mark.asyncio
async def test_orchestrator_auto_nlq_restricted_does_not_fallback():
    """Khi NLQ trả về restricted do vi phạm chính sách, KHÔNG BAO GIỜ fallback sang RAG."""
    mock_nlq_res = {
        "status": "restricted",
        "error_code": "policy_denied",
        "answer": "Thông tin nhạy cảm bị hạn chế.",
    }
    with (
        patch("app.core.orchestrator.ask_nlq", return_value=mock_nlq_res),
        patch("app.core.orchestrator.answer_rag_request", new_callable=AsyncMock) as mock_rag,
    ):
        req = AskUnifiedRequest(question="Danh sách số điện thoại khách hàng", mode="nlq")
        res = await ask_orchestrated(req)
        assert res.status == "restricted"
        assert res.error_code == "policy_denied"
        mock_rag.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_nlq_timeout_raises_504_with_trace_id():
    """Lỗi quá hạn thời gian (NlqRequestTimeout) được nâng thành HTTP 504 kèm X-Trace-Id."""
    with (
        patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.core.orchestrator.ask_nlq", side_effect=NlqRequestTimeout("Timeout")),
    ):
        req = AskUnifiedRequest(question="Có bao nhiêu đơn hàng?", mode="nlq")
        with pytest.raises(HTTPException) as exc_info:
            await ask_orchestrated(req)
        assert exc_info.value.status_code == 504
        assert exc_info.value.headers is not None
        assert exc_info.value.headers.get("X-Trace-Id") == MOCK_TRACE_ID


@pytest.mark.asyncio
async def test_orchestrator_nlq_db_failure_raises_503_with_trace_id():
    """Lỗi mất kết nối CSDL (psycopg.OperationalError) được nâng thành HTTP 503 kèm X-Trace-Id."""
    with (
        patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.core.orchestrator.ask_nlq", side_effect=psycopg.OperationalError("DB down")),
    ):
        req = AskUnifiedRequest(question="Có bao nhiêu đơn hàng?", mode="nlq")
        with pytest.raises(HTTPException) as exc_info:
            await ask_orchestrated(req)
        assert exc_info.value.status_code == 503
        assert exc_info.value.headers is not None
        assert exc_info.value.headers.get("X-Trace-Id") == MOCK_TRACE_ID


@pytest.mark.asyncio
async def test_orchestrator_nlq_unconfigured_db_raises_503():
    """Lỗi thiếu cấu hình NLQ_DATABASE_URL được nâng thành HTTP 503 kèm hướng dẫn."""
    with (
        patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.core.orchestrator.ask_nlq", side_effect=RuntimeError("NLQ_DATABASE_URL not set")),
    ):
        req = AskUnifiedRequest(question="Có bao nhiêu đơn hàng?", mode="nlq")
        with pytest.raises(HTTPException) as exc_info:
            await ask_orchestrated(req)
        assert exc_info.value.status_code == 503
        assert "PostgreSQL chỉ đọc" in exc_info.value.detail
        assert exc_info.value.headers is not None
        assert exc_info.value.headers.get("X-Trace-Id") == MOCK_TRACE_ID


def test_api_ask_endpoint_roundtrip_with_headers():
    """Kiểm tra gọi endpoint /api/ask qua HTTP TestClient trả mã 200 và header X-Trace-Id."""
    client = TestClient(app)
    with patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID):
        resp = client.post(
            "/api/ask",
            headers={"X-API-Key": settings.app_api_key},
            json={"question": "Xóa toàn bộ đơn hàng?", "mode": "auto"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unsupported_action"
    assert resp.headers.get("X-Trace-Id") == MOCK_TRACE_ID
    assert data["trace_id"] == MOCK_TRACE_ID


def test_api_ask_endpoint_propagates_504_with_trace_id():
    """Kiểm tra endpoint /api/ask khi gặp timeout trả về 504 và bảo toàn X-Trace-Id trong header."""
    client = TestClient(app)
    with (
        patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.core.orchestrator.ask_nlq", side_effect=NlqRequestTimeout("Timeout")),
    ):
        resp = client.post(
            "/api/ask",
            headers={"X-API-Key": settings.app_api_key},
            json={"question": "Có bao nhiêu đơn hàng?", "mode": "nlq"},
        )
    assert resp.status_code == 504
    assert resp.headers.get("X-Trace-Id") == MOCK_TRACE_ID
    assert "timed out" in resp.json()["detail"].lower()
