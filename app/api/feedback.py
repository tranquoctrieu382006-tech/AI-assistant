"""
Feedback API router.
Ghi nhận phản hồi đánh giá chất lượng (👍 / 👎) từ người dùng cuối về Langfuse.
"""

import logging
import re
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Security, status

from app.core.security import require_api_key
from app.core.tracing import get_langfuse_client, flush_langfuse

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)

HEX_32_REGEX = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)
HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", status.HTTP_422_UNPROCESSABLE_ENTITY)


class FeedbackRequest(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Điểm đánh giá: 1.0 (hài lòng/👍) hoặc 0.0 (chưa đúng/👎)")
    trace_id: str | None = Field(default=None, description="Mã trace_id của lượt sinh câu trả lời (32 ký tự hex)")
    session_id: str | None = Field(default=None, description="Mã phiên làm việc")
    comment: str | None = Field(default="", description="Góp ý chi tiết của người dùng")


class FeedbackResponse(BaseModel):
    status: str
    recorded: bool
    score: float
    trace_id: str | None = None
    session_id: str | None = None


@router.post("/feedback", response_model=FeedbackResponse, summary="Ghi nhận phản hồi người dùng")
def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    if not request.trace_id and not request.session_id:
        raise HTTPException(
            status_code=HTTP_422,
            detail="Ít nhất một trong trace_id hoặc session_id phải được cung cấp.",
        )

    if request.trace_id and not HEX_32_REGEX.match(request.trace_id):
        raise HTTPException(
            status_code=HTTP_422,
            detail="trace_id không hợp lệ (phải là chuỗi 32 ký tự thập lục phân).",
        )

    recorded = False
    client = get_langfuse_client()

    if client:
        try:
            score_id = f"feedback-{request.trace_id}" if request.trace_id else f"feedback-{request.session_id}"
            if request.trace_id:
                client.create_score(
                    name="user_feedback",
                    value=request.score,
                    trace_id=request.trace_id,
                    score_id=score_id,
                    comment=request.comment or "",
                )
                recorded = True
            elif request.session_id:
                client.create_score(
                    name="user_feedback",
                    value=request.score,
                    session_id=request.session_id,
                    score_id=score_id,
                    comment=request.comment or "",
                )
                recorded = True
            flush_langfuse()
            logger.info("Feedback recorded successfully for trace_id=%s session_id=%s score=%s score_id=%s",
                        request.trace_id, request.session_id, request.score, score_id)
        except Exception as exc:
            logger.warning("Không thể gửi phản hồi lên Langfuse: %s", exc)
    else:
        logger.info(
            "Langfuse chưa cấu hình; feedback không thể gửi về cloud (trace_id=%s, session_id=%s, score=%s)",
            request.trace_id, request.session_id, request.score
        )
        recorded = False

    return FeedbackResponse(
        status="ok",
        recorded=recorded,
        score=request.score,
        trace_id=request.trace_id,
        session_id=request.session_id,
    )
