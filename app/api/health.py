"""
Health check API router.
Hỗ trợ cả liveness và readiness probe, không chặn event loop.
"""

from fastapi import APIRouter, Query
from app.schemas.health import HealthResponse
from app.core.config import settings
from app.core.llm import (
    MODEL_REGISTRY,
    check_ollama_status_async,
    verify_model,
)

router = APIRouter()


@router.get("/health/live", summary="Liveness probe (nhẹ nhất, kiểm tra app còn sống)")
async def liveness():
    """Liveness probe: Trả lời ngay lập tức, không kiểm tra dependency ngoài."""
    return {"status": "alive"}


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái hệ thống đầy đủ")
async def health_check() -> HealthResponse:
    """
    Trả về thông tin trạng thái ứng dụng, model mặc định,
    danh sách roles khả dụng và tình trạng Ollama / Langfuse.
    Chạy async để không chặn luồng xử lý chính.
    """
    ollama_info = await check_ollama_status_async()
    langfuse_on = bool(
        settings.langfuse_secret_key
        and not settings.langfuse_secret_key.startswith("sk-lf-...")
    )

    return HealthResponse(
        status="ok",
        version="0.1.0",
        default_model=settings.default_model,
        available_roles=list(MODEL_REGISTRY.keys()),
        langfuse_enabled=langfuse_on,
        ollama_status=ollama_info,
    )


@router.get("/health/verify-model", summary="Kiểm tra trực tiếp một model")
async def verify_model_endpoint(
    role: str = Query(default="local", description="Role cần test (strong, fast, local)"),
):
    """Gọi thử 1 câu ngắn tới model để verify kết nối runtime."""
    return verify_model(role)
