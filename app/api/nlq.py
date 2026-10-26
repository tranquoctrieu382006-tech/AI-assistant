import logging

from fastapi import APIRouter, HTTPException, Security
import psycopg

from app.core.config import settings
from langgraph.errors import GraphRecursionError
from psycopg_pool import PoolTimeout

from app.core.nlq_graph import (
    NlqDatabaseUnavailable,
    NlqExecutionError,
    NlqUnsafeQueryError,
    ask_nlq,
)
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
        )
        return NlqAskResponse(**result)
    except GraphRecursionError as exc:
        logger.exception("NLQ graph exceeded its execution limit")
        raise HTTPException(status_code=500, detail="NLQ workflow không hoàn tất.") from exc
    except NlqUnsafeQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NlqExecutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NlqDatabaseUnavailable as exc:
        logger.exception("NLQ database unavailable")
        raise HTTPException(status_code=503, detail="NLQ database hiện không khả dụng.") from exc
    except PoolTimeout as exc:
        logger.exception("NLQ database connection pool exhausted")
        raise HTTPException(status_code=503, detail="NLQ database hiện quá tải.") from exc
    except psycopg.OperationalError as exc:
        logger.exception("NLQ database connection failed")
        raise HTTPException(status_code=503, detail="NLQ database hiện không khả dụng.") from exc
    except RuntimeError as exc:
        if "NLQ_DATABASE_URL" in str(exc):
            raise HTTPException(status_code=503, detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.") from exc
        logger.exception("NLQ runtime failure")
        raise HTTPException(status_code=502, detail="NLQ hiện chưa thể xử lý yêu cầu.") from exc
    except Exception as exc:
        logger.exception("NLQ request failed")
        raise HTTPException(status_code=502, detail="Không thể tạo hoặc chạy truy vấn NLQ an toàn.") from exc
