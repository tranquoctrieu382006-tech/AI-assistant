"""
Health check API router.
Hỗ trợ cả liveness và readiness probe, không chặn event loop.
"""

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from app.schemas.health import HealthResponse
from app.core.config import settings
from app.core.llm import (
    MODEL_REGISTRY,
    check_ollama_status_async,
    verify_model,
)
from app.core.tracing import is_langfuse_configured
import psycopg

router = APIRouter()


def _database_readiness() -> dict:
    if not settings.database_url.strip():
        return {"status": "error", "detail": "DATABASE_URL is not configured"}
    try:
        with psycopg.connect(settings.database_url, connect_timeout=3) as conn:
            vector = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        return {"status": "ok" if vector else "missing_vector_extension",
                "pgvector_version": vector[0] if vector else None}
    except Exception as exc:
        return {"status": "error", "detail": type(exc).__name__}


@router.get("/health/live", summary="Liveness probe (kiểm tra tiến trình còn sống)")
async def liveness():
    """Liveness probe: Trả lời 200 OK ngay lập tức, không kiểm tra dependency ngoài."""
    return {"status": "alive"}


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái hệ thống đầy đủ")
async def health_check() -> HealthResponse:
    """
    Trả về thông tin trạng thái ứng dụng, model mặc định,
    danh sách roles khả dụng và tình trạng Ollama / Langfuse.
    Nếu Ollama offline hoặc lỗi, trạng thái chuyển thành 'degraded'.
    """
    ollama_info = await check_ollama_status_async()
    langfuse_on = is_langfuse_configured()

    # Xác định status theo tình trạng thành phần phụ thuộc
    database_info = await run_in_threadpool(_database_readiness)
    active_model = settings.rag_model
    requires_ollama = active_model == "local" or active_model.startswith("ollama/")
    is_healthy = database_info["status"] == "ok" and (not requires_ollama or ollama_info.get("status") == "ok")
    system_status = "ok" if is_healthy else "degraded"

    return HealthResponse(
        status=system_status,
        version=settings.app_version,
        default_model=settings.default_model,
        available_roles=list(MODEL_REGISTRY.keys()),
        langfuse_enabled=langfuse_on,
        ollama_status=ollama_info,
        database_status=database_info,
    )


@router.get("/health/verify-model", summary="Kiểm tra trực tiếp một model (chỉ bật ở debug/dev)")
async def verify_model_endpoint(
    role: str = Query(default="local", description="Role cần test (strong, fast, local)"),
):
    """
    Gọi thử 1 câu ngắn tới model để verify kết nối runtime.
    Chỉ bật ở môi trường debug/development để tránh lạm dụng chi phí.
    Được chạy trong threadpool để không block event loop.
    """
    if not settings.debug:
        raise HTTPException(
            status_code=403,
            detail="Endpoint này chỉ được bật trong môi trường development/debug.",
        )

    if role not in MODEL_REGISTRY:
        raise HTTPException(status_code=422, detail="role must be one of: strong, fast, local")
    # Chạy hàm gọi model trong worker thread để tránh chặn event loop
    result = await run_in_threadpool(verify_model, role)
    return result
