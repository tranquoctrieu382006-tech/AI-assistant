import logging
from typing import Any, Literal
from pydantic import BaseModel, Field

from app.api.rag import _answer_rag
from app.core.nlq_graph import ask_nlq
from app.core.nlq_policy import has_sensitive_mention
from app.core.router import RouteCategory, route_question
from app.core.tracing import create_trace_id
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
    fallback_from: str | None = None
    error_code: str | None = None
    trace_id: str | None = None


async def ask_orchestrated(request: AskUnifiedRequest) -> AskUnifiedResponse:
    """
    Điều phối truy vấn hợp nhất tại tầng Backend:
    - Tiếp nhận câu hỏi, tự động định tuyến nếu ở chế độ 'auto'.
    - Chặn các câu lệnh ghi/xóa (unsupported_action) ngay lập tức.
    - Chặn câu hỏi nhạy cảm và không bao giờ fallback sang RAG để chống lộ dữ liệu.
    - Nếu NLQ trả về out_of_scope (unanswerable) mà không nhạy cảm -> tự động fallback sang RAG đúng 1 lần.
    """
    q = request.question.strip()
    trace_id = create_trace_id()

    # 1. Chế độ ép RAG
    if request.mode == "rag":
        rag_req = RagAskRequest(
            question=q,
            session_id=request.session_id,
            user_id=request.user_id,
            top_k=request.top_k,
        )
        rag_res = await _answer_rag(rag_req, trace_id=trace_id)
        return AskUnifiedResponse(
            route="rag",
            status="ok" if rag_res.sources else "out_of_scope",
            answer=rag_res.answer,
            sources=[s.model_dump() for s in rag_res.sources],
            trace_id=trace_id,
        )

    # 2. Chế độ ép NLQ
    if request.mode == "nlq":
        nlq_res = ask_nlq(q, session_id=request.session_id, user_id=request.user_id)
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

    # B. Ngoài phạm vi hỗ trợ
    if decision.category == RouteCategory.OUT_OF_SCOPE:
        return AskUnifiedResponse(
            route="out_of_scope",
            status="out_of_scope",
            answer="Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi hỗ trợ của hệ thống (hệ thống chỉ hỗ trợ tra cứu tài liệu quy định và dữ liệu bán hàng).",
            reasoning=decision.reasoning,
            trace_id=trace_id,
        )

    # C. Định tuyến CSDL (NLQ)
    if decision.category == RouteCategory.NLQ:
        nlq_res = ask_nlq(q, session_id=request.session_id, user_id=request.user_id)
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
            rag_res = await _answer_rag(rag_req, trace_id=trace_id)
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
    rag_res = await _answer_rag(rag_req, trace_id=trace_id)
    return AskUnifiedResponse(
        route="rag",
        status="ok" if rag_res.sources else "out_of_scope",
        answer=rag_res.answer,
        sources=[s.model_dump() for s in rag_res.sources],
        reasoning=decision.reasoning,
        trace_id=trace_id,
    )
