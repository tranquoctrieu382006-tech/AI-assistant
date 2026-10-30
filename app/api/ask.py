import logging

from fastapi import APIRouter, HTTPException, Response, Security
import psycopg

from app.core.nlq_graph import NlqRequestTimeout
from app.core.nlq_policy import PrivacyViolation, restricted_answer
from app.core.orchestrator import AskUnifiedRequest, AskUnifiedResponse, ask_orchestrated
from app.core.security import require_api_key
from app.core.tracing import create_trace_id

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AskUnifiedResponse, summary="Điểm truy vấn hợp nhất (Auto / RAG / NLQ với Fallback an toàn)")
async def ask_endpoint(request: AskUnifiedRequest, response: Response = None) -> AskUnifiedResponse:
    trace_id = create_trace_id()
    headers = {"X-Trace-Id": trace_id} if trace_id else None

    try:
        result = await ask_orchestrated(request, trace_id=trace_id)
    except NlqRequestTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Yêu cầu NLQ vượt quá thời gian xử lý.",
            headers=headers,
        ) from exc
    except (psycopg.OperationalError, psycopg.Error) as exc:
        logger.exception("Unified ask: database failure")
        raise HTTPException(
            status_code=503,
            detail="Cơ sở dữ liệu hiện không sẵn sàng.",
            headers=headers,
        ) from exc
    except PrivacyViolation as exc:
        # Phòng thủ: vi phạm quyền riêng tư là quyết định nội dung, không phải lỗi 5xx.
        logger.warning("Unified ask: privacy violation escaped: %s", exc)
        result = AskUnifiedResponse(
            route="nlq",
            status="restricted",
            answer=restricted_answer(),
            reasoning="Chặn truy cập dữ liệu nhạy cảm",
            error_code="policy_denied",
            trace_id=trace_id,
        )
    except HTTPException:
        # Orchestrator đã chuẩn hóa 503/504 kèm header; không bọc lại thành 502.
        raise
    except Exception as exc:
        logger.exception("Unified ask: unexpected failure")
        raise HTTPException(
            status_code=502,
            detail="Không thể xử lý câu hỏi do lỗi nhà cung cấp mô hình hoặc hạ tầng.",
            headers=headers,
        ) from exc

    if response is not None and result.trace_id:
        response.headers["X-Trace-Id"] = result.trace_id
    return result
