import asyncio
import logging
from typing import Any, Literal
from fastapi import HTTPException
from pydantic import BaseModel, Field
import psycopg

from app.core.nlq_graph import NlqRequestTimeout, ask_nlq
from app.core.nlq_policy import (
    InvalidSqlError,
    PrivacyViolation,
    SchemaMismatch,
    has_sensitive_mention,
    is_human_pii_query,
    restricted_answer,
)
from app.core.rag_service import answer_rag_request
from app.core.router import RouteCategory, route_question
from app.core.tracing import create_trace_id, langfuse_observation, update_langfuse_observation
from app.schemas.rag import RagAskRequest

logger = logging.getLogger(__name__)


class AskUnifiedRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    mode: Literal["auto", "rag", "nlq"] = "auto"
    top_k: int = Field(default=5, ge=1, le=20)


class AskUnifiedResponse(BaseModel):
    route: Literal["rag", "nlq", "out_of_scope", "unsupported_action"]
    status: Literal["ok", "restricted", "out_of_scope", "unsupported_action", "error"] = "ok"
    answer: str
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    reasoning: str | None = None
    route_reason: str | None = None
    fallback_from: str | None = None
    error_code: str | None = None
    trace_id: str | None = None


async def _run_nlq_step(
    question: str,
    session_id: str | None,
    user_id: str | None,
    trace_id: str,
) -> dict[str, Any]:
    """Chạy NLQ trong thread riêng, bọc tracing và chuẩn hóa lỗi hạ tầng (503/504)."""
    headers = {"X-Trace-Id": trace_id} if trace_id else None
    try:
        with langfuse_observation(name="nlq-request", trace_id=trace_id) as observation:
            result = await asyncio.to_thread(
                ask_nlq,
                question,
                session_id=session_id,
                user_id=user_id,
            )
            update_langfuse_observation(
                observation,
                output={
                    "sql": result.get("sql"),
                    "row_count": len(result.get("rows", [])),
                    "status": result.get("status"),
                },
            )
            return result
    except NlqRequestTimeout as exc:
        raise HTTPException(status_code=504, detail="NLQ request timed out.", headers=headers) from exc
    except (psycopg.OperationalError, psycopg.Error) as exc:
        logger.exception("NLQ database connection failure in orchestrator")
        raise HTTPException(status_code=503, detail="NLQ database is unavailable.", headers=headers) from exc
    except RuntimeError as exc:
        if "NLQ_DATABASE_URL" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.",
                headers=headers,
            ) from exc
        logger.exception("NLQ runtime failure in orchestrator")
        raise HTTPException(status_code=503, detail="NLQ hiện chưa sẵn sàng.", headers=headers) from exc
    except PrivacyViolation as exc:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": restricted_answer(),
            "repaired": False,
            "truncated": False,
            "status": "restricted",
            "error_code": "policy_denied",
        }
    except SchemaMismatch as exc:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": f"Không tìm thấy dữ liệu phù hợp trong CSDL: {exc}",
            "repaired": False,
            "truncated": False,
            "status": "out_of_scope",
            "error_code": "unanswerable",
        }
    except InvalidSqlError as exc:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "answer": f"Không thể tạo hoặc chạy truy vấn an toàn: {exc}",
            "repaired": False,
            "truncated": False,
            "status": "error",
            "error_code": "exec_error",
        }


def _policy_denied(trace_id: str | None, reasoning: str) -> AskUnifiedResponse:
    """Câu hỏi trên đường NLQ cần một trường nhạy cảm không được cấp quyền."""
    return AskUnifiedResponse(
        route="nlq",
        status="restricted",
        answer=restricted_answer(),
        columns=[],
        rows=[],
        reasoning=reasoning,
        error_code="policy_denied",
        trace_id=trace_id,
    )


async def ask_orchestrated(
    request: AskUnifiedRequest, trace_id: str | None = None
) -> AskUnifiedResponse:
    """
    Điều phối truy vấn hợp nhất tại tầng Backend:
    Hành vi:
    - Tiếp nhận câu hỏi, tự động định tuyến nếu ở chế độ 'auto'.
    - Cổng PII chỉ áp cho đường NLQ (CSDL): dữ liệu cá nhân khách hàng chỉ tồn
      tại trong CSDL, nên câu hỏi đi RAG không bị chặn bởi cổng này. Nhờ vậy
      câu hỏi hotline/liên hệ công ty hoặc chính sách bảo mật vẫn tới RAG.
    - Chặn các câu lệnh ghi/xóa (unsupported_action) ngay lập tức.
    - Chặn câu hỏi nhạy cảm và không bao giờ fallback sang RAG để chống lộ dữ liệu.
    - Nếu NLQ trả về out_of_scope (unanswerable) mà không nhạy cảm -> tự động fallback sang RAG đúng 1 lần.
    """
    q = request.question.strip()
    trace_id = trace_id or create_trace_id()

    # 1. Chế độ ép RAG
    if request.mode == "rag":
        rag_req = RagAskRequest(
            question=q,
            session_id=request.session_id,
            user_id=request.user_id,
            top_k=request.top_k,
        )
        rag_res = await answer_rag_request(rag_req, trace_id=trace_id)
        return AskUnifiedResponse(
            route="rag",
            status="ok" if rag_res.sources else "out_of_scope",
            answer=rag_res.answer,
            sources=[s.model_dump() for s in rag_res.sources],
            trace_id=trace_id,
        )

    # 2. Chế độ ép NLQ
    if request.mode == "nlq":
        if is_human_pii_query(q):
            return _policy_denied(
                trace_id, "Chặn truy cập dữ liệu nhạy cảm trên đường NLQ"
            )
        nlq_res = await _run_nlq_step(q, request.session_id, request.user_id, trace_id)
        return AskUnifiedResponse(
            route="nlq",
            status=nlq_res.get("status", "ok"),
            answer=nlq_res.get("answer", ""),
            sql=nlq_res.get("sql"),
            columns=nlq_res.get("columns", []),
            rows=nlq_res.get("rows", []),
            error_code=nlq_res.get("error_code"),
            trace_id=trace_id,
        )

    # 3. Chế độ Tự động thông minh (auto)
    decision = route_question(q)

    # A. Chặn thao tác phá hoại / ghi dữ liệu
    if decision.category == RouteCategory.UNSUPPORTED_ACTION:
        return AskUnifiedResponse(
            route="unsupported_action",
            status="unsupported_action",
            answer="Hệ thống chỉ hỗ trợ tra cứu dữ liệu (chế độ chỉ đọc), không thực hiện thay đổi hoặc xóa dữ liệu.",
            reasoning=decision.reasoning,
            trace_id=trace_id,
        )

    # B. Ngoài phạm vi hỗ trợ (Chào hỏi xã giao hoặc lạc đề)
    if decision.category == RouteCategory.OUT_OF_SCOPE:
        is_greeting = decision.reason_code == "greeting"
        if is_greeting:
            greeting_answer = (
                "Xin chào! Tôi là trợ lý ảo hỗ trợ tra cứu tài liệu quy định và số liệu kinh doanh. "
                "Bạn có thể đặt câu hỏi về quy chế công ty, chính sách bán hàng, đổi trả, bảo hành hoặc tra cứu đơn hàng, doanh thu, sản phẩm. "
                "Tôi có thể giúp gì cho bạn hôm nay?"
            )
            return AskUnifiedResponse(
                route="out_of_scope",
                status="out_of_scope",
                answer=greeting_answer,
                reasoning=decision.reasoning,
                route_reason=decision.reason_code,
                trace_id=trace_id,
            )

        return AskUnifiedResponse(
            route="out_of_scope",
            status="out_of_scope",
            answer="Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của hệ thống (hệ thống chỉ hỗ trợ tra cứu tài liệu quy định và dữ liệu bán hàng).",
            reasoning=decision.reasoning,
            route_reason=decision.reason_code or None,
            trace_id=trace_id,
        )

    # C. Định tuyến CSDL (NLQ)
    if decision.category == RouteCategory.NLQ:
        if is_human_pii_query(q):
            return _policy_denied(trace_id, decision.reasoning)
        nlq_res = await _run_nlq_step(q, request.session_id, request.user_id, trace_id)
        nlq_status = nlq_res.get("status", "ok")

        # Vi phạm chính sách / cột nhạy cảm: Tuyệt đối KHÔNG fallback sang RAG!
        if nlq_status == "restricted":
            return AskUnifiedResponse(
                route="nlq",
                status="restricted",
                answer=nlq_res.get("answer", ""),
                sql=nlq_res.get("sql"),
                columns=[],
                rows=[],
                reasoning=decision.reasoning,
                error_code="policy_denied",
                trace_id=trace_id,
            )

        # CSDL không trả lời được (out_of_scope / unanswerable)
        if nlq_status == "out_of_scope":
            # Nếu câu hỏi có cờ nhạy cảm: Chặn đứng, không fallback!
            if has_sensitive_mention(q):
                return AskUnifiedResponse(
                    route="nlq",
                    status="restricted",
                    answer=nlq_res.get("answer", "Thông tin bị hạn chế để bảo vệ quyền riêng tư."),
                    sql="",
                    columns=[],
                    rows=[],
                    reasoning=decision.reasoning,
                    error_code="policy_denied",
                    trace_id=trace_id,
                )

            # An toàn để fallback sang RAG đúng 1 lần
            logger.info("NLQ out_of_scope (unanswerable); falling back to RAG for question: %s", q)
            rag_req = RagAskRequest(
                question=q,
                session_id=request.session_id,
                user_id=request.user_id,
                top_k=request.top_k,
            )
            rag_res = await answer_rag_request(rag_req, trace_id=trace_id)
            return AskUnifiedResponse(
                route="rag",
                fallback_from="nlq",
                status="ok" if rag_res.sources else "out_of_scope",
                answer=rag_res.answer,
                sources=[s.model_dump() for s in rag_res.sources],
                reasoning=f"Tự động chuyển từ NLQ sang RAG: {decision.reasoning}",
                trace_id=trace_id,
            )

        # Trả về kết quả bình thường hoặc lỗi SQL
        return AskUnifiedResponse(
            route="nlq",
            status=nlq_status,
            answer=nlq_res.get("answer", ""),
            sql=nlq_res.get("sql"),
            columns=nlq_res.get("columns", []),
            rows=nlq_res.get("rows", []),
            reasoning=decision.reasoning,
            error_code=nlq_res.get("error_code"),
            trace_id=trace_id,
        )

    # D. Định tuyến Tài liệu (RAG)
    rag_req = RagAskRequest(
        question=q,
        session_id=request.session_id,
        user_id=request.user_id,
        top_k=request.top_k,
    )
    rag_res = await answer_rag_request(rag_req, trace_id=trace_id)
    return AskUnifiedResponse(
        route="rag",
        status="ok" if rag_res.sources else "out_of_scope",
        answer=rag_res.answer,
        sources=[s.model_dump() for s in rag_res.sources],
        reasoning=decision.reasoning,
        trace_id=trace_id,
    )
