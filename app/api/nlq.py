import logging

from fastapi import APIRouter, HTTPException, Response, Security
import psycopg

from app.core.config import settings
from app.core.nlq_graph import NlqRequestTimeout, ask_nlq
from app.core.nlq_policy import InvalidSqlError, PrivacyViolation, SchemaMismatch, restricted_answer
from app.core.security import require_api_key
from app.core.tracing import create_trace_id, langfuse_observation, update_langfuse_observation
from app.schemas.nlq import NlqAskRequest, NlqAskResponse

router = APIRouter(dependencies=[Security(require_api_key)])
logger = logging.getLogger(__name__)


@router.post("/nlq/ask", response_model=NlqAskResponse, summary="Hỏi dữ liệu bằng ngôn ngữ tự nhiên")
def ask_database(request: NlqAskRequest, response: Response = None) -> NlqAskResponse:
    trace_id = create_trace_id()
    headers = {"X-Trace-Id": trace_id} if trace_id else None

    if not settings.nlq_database_url.strip():
        raise HTTPException(
            status_code=503,
            detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.",
            headers=headers,
        )

    try:
        with langfuse_observation(name="nlq-request", trace_id=trace_id) as observation:
            result = ask_nlq(
                request.question,
                session_id=request.session_id,
                user_id=request.user_id,
            )
            update_langfuse_observation(
                observation,
                output={
                    "sql": result.get("sql"),
                    "row_count": len(result.get("rows", [])),
                    "status": result.get("status"),
                },
            )

        if trace_id and response is not None:
            response.headers["X-Trace-Id"] = trace_id

        return NlqAskResponse(**result, trace_id=trace_id)
    except NlqRequestTimeout as exc:
        raise HTTPException(status_code=504, detail="NLQ request timed out.", headers=headers) from exc
    except (psycopg.OperationalError, psycopg.Error) as exc:
        logger.exception("NLQ database connection failure")
        raise HTTPException(status_code=503, detail="NLQ database is unavailable.", headers=headers) from exc
    except RuntimeError as exc:
        if "NLQ_DATABASE_URL" in str(exc):
            raise HTTPException(
                status_code=503,
                detail="NLQ chưa được cấu hình tài khoản PostgreSQL chỉ đọc.",
                headers=headers,
            ) from exc
        logger.exception("NLQ configuration/runtime failure")
        raise HTTPException(status_code=503, detail="NLQ hiện chưa sẵn sàng.", headers=headers) from exc
    except PrivacyViolation as exc:
        logger.warning("NLQ privacy violation: %s", exc)
        if trace_id and response is not None:
            response.headers["X-Trace-Id"] = trace_id
        return NlqAskResponse(
            sql="",
            columns=[],
            rows=[],
            answer=restricted_answer(),
            repaired=False,
            truncated=False,
            trace_id=trace_id,
            status="restricted",
            error_code="policy_denied",
        )
    except SchemaMismatch as exc:
        logger.warning("NLQ schema mismatch: %s", exc)
        if trace_id and response is not None:
            response.headers["X-Trace-Id"] = trace_id
        return NlqAskResponse(
            sql="",
            columns=[],
            rows=[],
            answer=f"Không tìm thấy dữ liệu phù hợp trong CSDL: {exc}",
            repaired=False,
            truncated=False,
            trace_id=trace_id,
            status="out_of_scope",
            error_code="unanswerable",
        )
    except InvalidSqlError as exc:
        logger.warning("NLQ invalid SQL: %s", exc)
        if trace_id and response is not None:
            response.headers["X-Trace-Id"] = trace_id
        return NlqAskResponse(
            sql="",
            columns=[],
            rows=[],
            answer=f"Không thể tạo truy vấn an toàn: {exc}",
            repaired=False,
            truncated=False,
            trace_id=trace_id,
            status="error",
            error_code="exec_error",
        )
    except Exception as exc:
        logger.exception("NLQ unexpected failure")
        raise HTTPException(
            status_code=502,
            detail="Không thể tạo hoặc chạy truy vấn NLQ an toàn.",
            headers=headers,
        ) from exc
