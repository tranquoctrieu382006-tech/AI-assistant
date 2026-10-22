"""
Health check API router.
"""

from fastapi import APIRouter, Query
from app.schemas.health import HealthResponse
from app.core.config import settings
from app.core.llm import (
    MODEL_REGISTRY,
    check_ollama_status,
    verify_model,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra trạng thái hệ thống")
async def health_check() -> HealthResponse:
    """
    Trả về thông tin trạng thái ứng dụng, model mặc định,
    danh sách roles khả dụng và tình trạng Ollama / Langfuse.
    """
    ollama_info = check_ollama_status()
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
