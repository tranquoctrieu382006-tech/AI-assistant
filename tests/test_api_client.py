"""
Unit tests for FastAPI endpoints using async HTTP client with in-process ASGI transport.
Đảm bảo kiểm tra toàn diện API mà không gọi LLM hoặc database thật.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import pytest

from app.main import app
from app.core.config import settings
from app.schemas.rag import RagAskResponse, RagSource


@pytest.fixture(autouse=True)
def bypass_app_lifespan(monkeypatch):
    """
    Bypass FastAPI lifespan during unit tests.
    Tránh việc gọi check_ollama_status_async hoặc kết nối ngoài gây treo khi test chạy trong môi trường cô lập.
    """
    @asynccontextmanager
    async def _dummy_lifespan(_app):
        yield

    monkeypatch.setattr(app.router, "lifespan_context", _dummy_lifespan)


async def api_request(
    method: str,
    path: str,
    *,
    json: dict | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Gửi HTTP request trực tiếp vào app qua ASGI Transport (không mở socket thật)."""
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=json, headers=headers)


@pytest.mark.asyncio
async def test_api_requires_api_key():
    """Kiểm tra các endpoint bảo mật từ chối request thiếu X-API-Key."""
    res = await api_request("POST", "/api/rag/ask", json={"question": "Chính sách đổi trả?"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_api_rejects_invalid_api_key():
    """Kiểm tra request với X-API-Key sai bị từ chối."""
    res = await api_request(
        "POST",
        "/api/rag/ask",
        json={"question": "Chính sách đổi trả?"},
        headers={"X-API-Key": "wrong-key-less-than-required-length"},
    )
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_health_live_endpoint():
    """Kiểm tra liveness probe không yêu cầu auth."""
    res = await api_request("GET", "/api/health/live")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_rag_ask_endpoint_success():
    """Kiểm tra POST /api/rag/ask trả về đúng schema khi mock thành công."""
    mock_response = RagAskResponse(
        answer="Thời gian đổi trả là 7 ngày kể từ khi nhận hàng.",
        sources=[
            RagSource(
                source="04_chinh_sach_doi_tra.docx",
                chunk_index=0,
                score=0.92,
                section="Thời hạn",
                page_number=1,
            )
        ],
    )
    with patch("app.api.rag._answer_rag", new_callable=AsyncMock, return_value=mock_response):
        res = await api_request(
            "POST",
            "/api/rag/ask",
            json={"question": "Chính sách đổi trả hàng như thế nào?"},
            headers={"X-API-Key": settings.app_api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Thời gian đổi trả là 7 ngày kể từ khi nhận hàng."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "04_chinh_sach_doi_tra.docx"


@pytest.mark.asyncio
async def test_nlq_ask_endpoint_success(monkeypatch):
    """Kiểm tra POST /api/nlq/ask trả về đúng schema kết quả truy vấn."""
    monkeypatch.setattr(settings, "nlq_database_url", "postgresql://ai_nlq_reader:secret@localhost:5432/demo")
    mock_nlq_result = {
        "sql": "SELECT COUNT(*) FROM orders;",
        "columns": ["count"],
        "rows": [{"count": 150}],
        "answer": "Tổng số đơn hàng hiện tại là 150 đơn.",
        "repaired": False,
        "truncated": False,
    }
    with patch("app.api.nlq.ask_nlq", return_value=mock_nlq_result):
        res = await api_request(
            "POST",
            "/api/nlq/ask",
            json={"question": "Có bao nhiêu đơn hàng?"},
            headers={"X-API-Key": settings.app_api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["sql"] == "SELECT COUNT(*) FROM orders;"
        assert data["columns"] == ["count"]
        assert len(data["rows"]) == 1
        assert data["repaired"] is False


@pytest.mark.asyncio
async def test_rag_ask_validation_error():
    """Kiểm tra validation error khi truyền question rỗng."""
    res = await api_request(
        "POST",
        "/api/rag/ask",
        json={"question": ""},
        headers={"X-API-Key": settings.app_api_key},
    )
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_feedback_endpoint_success():
    """Kiểm tra POST /api/feedback tiếp nhận phản hồi hợp lệ."""
    with patch("app.api.feedback.get_langfuse_client", return_value=None):
        res = await api_request(
            "POST",
            "/api/feedback",
            json={"score": 1.0, "trace_id": "1bc12f0391e0773c3458cc76c295d8c6", "comment": "Tốt"},
            headers={"X-API-Key": settings.app_api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["score"] == 1.0
        assert data["trace_id"] == "1bc12f0391e0773c3458cc76c295d8c6"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_feedback_endpoint_missing_identifiers_returns_422():
    """Kiểm tra POST /api/feedback trả về HTTP 422 khi thiếu cả trace_id lẫn session_id."""
    res = await api_request(
        "POST",
        "/api/feedback",
        json={"score": 0.5, "comment": "Không có id"},
        headers={"X-API-Key": settings.app_api_key},
    )
    assert res.status_code == 422
    assert "trace_id hoặc session_id" in res.json()["detail"]


@pytest.mark.asyncio
async def test_feedback_endpoint_invalid_trace_id_format_returns_422():
    """Kiểm tra POST /api/feedback trả về HTTP 422 khi trace_id không phải 32 ký tự hex."""
    res = await api_request(
        "POST",
        "/api/feedback",
        json={"score": 1.0, "trace_id": "invalid-non-hex-id"},
        headers={"X-API-Key": settings.app_api_key},
    )
    assert res.status_code == 422
    assert "32 ký tự" in res.json()["detail"]


@pytest.mark.asyncio
async def test_feedback_endpoint_with_session_id():
    """Kiểm tra POST /api/feedback tiếp nhận phản hồi hợp lệ khi chỉ có session_id."""
    with patch("app.api.feedback.get_langfuse_client", return_value=None):
        res = await api_request(
            "POST",
            "/api/feedback",
            json={"score": 1.0, "session_id": "sess-xyz-789", "comment": "Rất hài lòng"},
            headers={"X-API-Key": settings.app_api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["session_id"] == "sess-xyz-789"
        assert data["recorded"] is False


def test_session_only_feedback_uses_unique_score_ids():
    """Feedback chỉ có session_id không được ghi đè lên lần gửi trước."""
    from app.api.feedback import FeedbackRequest, record_feedback

    client = MagicMock()
    with (
        patch("app.api.feedback.get_langfuse_client", return_value=client),
        patch("app.api.feedback.flush_langfuse"),
    ):
        record_feedback(FeedbackRequest(score=1.0, session_id="sess-1"))
        record_feedback(FeedbackRequest(score=0.0, session_id="sess-1"))

    score_ids = [call.kwargs["score_id"] for call in client.create_score.call_args_list]
    assert len(score_ids) == 2
    assert len(set(score_ids)) == 2
    assert all(score_id.startswith("feedback-session-sess-1-") for score_id in score_ids)


def test_trace_feedback_keeps_deterministic_score_id():
    """Feedback theo trace vẫn idempotent để không tạo nhiều score trùng."""
    from app.api.feedback import FeedbackRequest, record_feedback

    trace_id = "1bc12f0391e0773c3458cc76c295d8c6"
    client = MagicMock()
    with (
        patch("app.api.feedback.get_langfuse_client", return_value=client),
        patch("app.api.feedback.flush_langfuse"),
    ):
        record_feedback(FeedbackRequest(score=1.0, trace_id=trace_id))
        record_feedback(FeedbackRequest(score=1.0, trace_id=trace_id))

    score_ids = [call.kwargs["score_id"] for call in client.create_score.call_args_list]
    assert score_ids == [f"feedback-{trace_id}", f"feedback-{trace_id}"]


async def test_router_classify_endpoint():
    """Kiểm tra POST /api/router/classify phân loại intent qua HTTP mà không gọi LLM ngoài."""
    from app.core.router import RouteDecision, RouteCategory
    mock_decision = RouteDecision(category=RouteCategory.RAG, reasoning="Khớp từ khóa tài liệu")
    with patch("app.api.router_api.route_question", return_value=mock_decision):
        res = await api_request(
            "POST",
            "/api/router/classify",
            json={"question": "Chính sách đổi trả hàng như thế nào?"},
            headers={"X-API-Key": settings.app_api_key},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["category"] == "rag"
        assert data["reasoning"] == "Khớp từ khóa tài liệu"
