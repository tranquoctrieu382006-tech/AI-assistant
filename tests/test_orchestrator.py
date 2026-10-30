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
from app.core.router import RouteCategory, RouteDecision
from app.main import app
from app.schemas.rag import RagAskResponse, RagSource

MOCK_TRACE_ID = "a" * 32


def test_core_orchestrator_does_not_import_api_modules():
    """Kiểm tra app/core/orchestrator.py tuân thủ Clean Architecture, không import app.api."""
    with open("app/core/orchestrator.py", "r", encoding="utf-8") as f:
        content = f.read()
    assert "app.api" not in content, "Phát hiện app/core/orchestrator.py phụ thuộc ngược vào app.api!"


@pytest.mark.asyncio
async def test_orchestrator_blocks_human_pii_on_nlq_route():
    """Câu PII định tuyến vào NLQ bị chặn, không gọi ask_nlq và không fallback RAG."""
    with (
        patch("app.core.orchestrator.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.core.orchestrator.ask_nlq") as mock_nlq,
        patch("app.core.orchestrator.answer_rag_request", new_callable=AsyncMock) as mock_rag,
    ):
        res = await ask_orchestrated(
            AskUnifiedRequest(
                question="Số điện thoại của khách hàng là bao nhiêu?", mode="auto"
            )
        )
    assert res.route == "nlq"
    assert res.status == "restricted"
    assert res.error_code == "policy_denied"
    assert res.trace_id == MOCK_TRACE_ID
    assert "không có trong dữ liệu được phép" in res.answer
    mock_nlq.assert_not_called()
    mock_rag.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_forced_nlq_blocks_pii_before_calling_graph():
    """Chế độ ép NLQ cũng phải chặn PII trước khi gọi graph."""
    with patch("app.core.orchestrator.ask_nlq") as mock_nlq:
        res = await ask_orchestrated(
            AskUnifiedRequest(question="Email của khách hàng VIP là gì?", mode="nlq")
        )
    assert res.status == "restricted"
    assert res.error_code == "policy_denied"
    mock_nlq.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_rag_mode_not_blocked_by_pii_gate():
    """Cổng PII chỉ áp cho NLQ: câu chính sách nhắc số điện thoại vẫn đi RAG."""
    mock_rag_res = RagAskResponse(
        answer="Chính sách bảo mật quy định không chia sẻ thông tin khách hàng.",
        sources=[RagSource(source="bao_mat.md", chunk_index=0, score=0.9)],
    )
    with patch(
        "app.core.orchestrator.answer_rag_request",
        new_callable=AsyncMock,
        return_value=mock_rag_res,
    ) as mock_rag:
        res = await ask_orchestrated(
            AskUnifiedRequest(
                question="Chính sách bảo mật số điện thoại của khách hàng là gì?",
                mode="rag",
            )
        )
    assert res.route == "rag"
    assert res.status == "ok"
    assert res.error_code is None
    mock_rag.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_auto_hotline_goes_to_rag():
    """Câu hỏi hotline/CSKH của công ty phải đi RAG, không vào NLQ."""
    mock_rag_res = RagAskResponse(
        answer="Tổng đài chăm sóc khách hàng: 1900 1234.",
        sources=[RagSource(source="lien_he.md", chunk_index=0, score=0.87)],
    )
    with (
        patch(
            "app.core.orchestrator.answer_rag_request",
            new_callable=AsyncMock,
            return_value=mock_rag_res,
        ),
        patch("app.core.orchestrator.ask_nlq") as mock_nlq,
    ):
        res = await ask_orchestrated(
            AskUnifiedRequest(
                question="Số điện thoại chăm sóc khách hàng của công ty là gì?",
                mode="auto",
            )
        )
    assert res.route == "rag"
    assert res.status == "ok"
    mock_nlq.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_forced_nlq_hotline_is_restricted():
    """Nếu hotline bị đẩy vào NLQ (chế độ ép), vẫn restricted, không lộ dữ liệu."""
    with patch("app.core.orchestrator.ask_nlq") as mock_nlq:
        res = await ask_orchestrated(
            AskUnifiedRequest(
                question="Số điện thoại chăm sóc khách hàng của công ty là gì?",
                mode="nlq",
            )
        )
    assert res.route == "nlq"
    assert res.status == "restricted"
    assert res.error_code == "policy_denied"
    mock_nlq.assert_not_called()


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
        patch("app.api.ask.create_trace_id", return_value=MOCK_TRACE_ID),
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


SUPPORT_OR_POLICY_QUESTIONS = [
    "Số điện thoại tổng đài chăm sóc khách hàng là bao nhiêu?",
    "Số điện thoại chăm sóc khách hàng của công ty là gì?",
    "Chính sách bảo mật số điện thoại của khách hàng là gì?",
    "Quy trình xác minh địa chỉ nhân viên mới là gì?",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("question", SUPPORT_OR_POLICY_QUESTIONS)
async def test_support_and_policy_questions_always_reach_rag(question):
    """Câu hotline/CSKH và câu chính sách có chữ "khách hàng"/"nhân viên" phải đi RAG.

    Đây là các câu từng bị cổng PII chặn oan khi cổng còn chạy trước router:
    "chăm sóc khách hàng" khớp _HUMAN_ENTITY_RE, còn "địa chỉ nhân viên" khớp
    cả trường PII lẫn thực thể người. Cổng PII chỉ được chạy SAU khi router đã
    quyết định đường NLQ, nên các câu này phải nguyên vẹn tới RAG.
    """
    mock_rag_res = RagAskResponse(
        answer="Thông tin từ tài liệu.",
        sources=[RagSource(source="lien_he.md", chunk_index=0, score=0.9)],
    )
    with (
        patch(
            "app.core.orchestrator.answer_rag_request",
            new_callable=AsyncMock,
            return_value=mock_rag_res,
        ) as mock_rag,
        patch("app.core.orchestrator.ask_nlq") as mock_nlq,
    ):
        res = await ask_orchestrated(AskUnifiedRequest(question=question, mode="auto"))

    assert res.route == "rag"
    assert res.status == "ok"
    assert res.error_code is None
    mock_rag.assert_awaited_once()
    mock_nlq.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Số điện thoại tổng đài chăm sóc khách hàng là bao nhiêu?",
        "Số điện thoại chăm sóc khách hàng của công ty là gì?",
    ],
)
async def test_misrouted_support_contact_in_nlq_mode_is_still_restricted(question):
    """Nếu câu hotline/CSKH bị đẩy vào NLQ, cổng PII vẫn chặn (không lộ dữ liệu)."""
    with patch("app.core.orchestrator.ask_nlq") as mock_nlq:
        res = await ask_orchestrated(AskUnifiedRequest(question=question, mode="nlq"))
    assert res.route == "nlq"
    assert res.status == "restricted"
    assert res.error_code == "policy_denied"
    assert "không có trong dữ liệu được phép" in res.answer
    mock_nlq.assert_not_called()


@pytest.mark.asyncio
async def test_greeting_uses_reason_code_not_reasoning_text():
    """Lời chào phải dựa trên `reason_code`, không so khớp chuỗi `reasoning` hiển thị."""
    decision = RouteDecision(
        category=RouteCategory.OUT_OF_SCOPE,
        reasoning="Chuỗi hiển thị đã được đổi hoàn toàn",
        reason_code="greeting",
    )
    with patch("app.core.orchestrator.route_question", return_value=decision):
        res = await ask_orchestrated(AskUnifiedRequest(question="Xin chào", mode="auto"))
    assert res.route == "out_of_scope"
    assert res.route_reason == "greeting"
    assert res.answer.startswith("Xin chào!")


def test_api_ask_unexpected_failure_returns_502_with_trace_id():
    """Lỗi không lường trước (provider LLM/hạ tầng) phải ra 502 kèm X-Trace-Id."""

    async def boom(request, trace_id=None):
        raise RuntimeError("provider exploded")

    client = TestClient(app)
    with (
        patch("app.api.ask.create_trace_id", return_value=MOCK_TRACE_ID),
        patch("app.api.ask.ask_orchestrated", side_effect=boom),
    ):
        resp = client.post(
            "/api/ask",
            headers={"X-API-Key": settings.app_api_key},
            json={"question": "Có bao nhiêu đơn hàng?", "mode": "nlq"},
        )
    assert resp.status_code == 502
    assert resp.headers.get("X-Trace-Id") == MOCK_TRACE_ID
    assert "hạ tầng" in resp.json()["detail"]
