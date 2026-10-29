import logging
import uuid

from fastapi import APIRouter, HTTPException, Security
import psycopg

from app.core.config import settings
from app.core.nlq_graph import NlqRequestTimeout, ask_nlq
from app.core.security import require_api_key
from app.schemas.nlq import NlqAskRequest, NlqAskResponse

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


@router.post("/nlq/ask", response_model=NlqAskResponse, summary="Hỏi dữ liệu bằng ngôn ngữ tự nhiên")
def ask_database(request: NlqAskRequest) -> NlqAskResponse:
    if not settings.nlq_database_url.strip():
        raise HTTPException(status_code=503, detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.")
    try:
        result = ask_nlq(
            request.question,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        trace_id = result.pop("trace_id", None) or f"nlq-{uuid.uuid4().hex[:12]}"
        return NlqAskResponse(**result, trace_id=trace_id)
    except NlqRequestTimeout as exc:
        raise HTTPException(status_code=504, detail="NLQ request timed out.") from exc
    except (psycopg.OperationalError, psycopg.Error) as exc:
        logger.exception("NLQ database connection failure")
        raise HTTPException(status_code=503, detail="NLQ database is unavailable.") from exc
    except RuntimeError as exc:
        if "NLQ_DATABASE_URL" in str(exc):
            raise HTTPException(status_code=503, detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.") from exc
        logger.exception("NLQ configuration/runtime failure")
        raise HTTPException(status_code=503, detail="NLQ hiện chưa sẵn sàng.") from exc
    except Exception as exc:
        logger.exception("NLQ request failed")
        raise HTTPException(status_code=502, detail="Không thể tạo hoặc chạy truy vấn NLQ an toàn.") from exc
