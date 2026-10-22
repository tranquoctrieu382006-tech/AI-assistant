"""
Lớp cấu hình LLM đa mô hình.

Đổi mô hình chỉ bằng biến môi trường, không sửa code.
Sử dụng init_chat_model (LangChain native) để tự nhận provider từ prefix.

Cải tiến:
- Cache instance theo (model_name, temperature, timeout) để tái sử dụng
- Hỗ trợ timeout và max_retries cho từng provider
- Cung cấp cả sync và async check cho Ollama status
"""

import logging
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Registry: role → model string (lấy từ config/env) ──────────
MODEL_REGISTRY: dict[str, str] = {
    "strong": settings.strong_model,
    "fast": settings.fast_model,
    "local": settings.local_model,
}

# Cache instance LLM để tránh tạo mới mỗi request
_LLM_CACHE: dict[tuple, BaseChatModel] = {}


class LLMInitError(Exception):
    """Không thể khởi tạo LLM model."""
    pass


def _resolve_model_name(model_role: str) -> str:
    """Chuyển role → model name. Cho phép truyền model string trực tiếp."""
    if model_role == "default":
        return settings.default_model
    if model_role in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_role]
    return model_role  # truyền trực tiếp, vd "openai/gpt-4o"


def _build_hint(model_name: str) -> str:
    """Gợi ý nguyên nhân lỗi phổ biến theo provider."""
    name_lower = model_name.lower()
    if "ollama" in name_lower:
        return (
            "\n→ Kiểm tra: Ollama đang chạy? (`ollama list`)"
            f"\n→ Model đã pull? (`ollama pull {model_name.split('/')[-1]}`)"
            f"\n→ OLLAMA_BASE_URL đúng chưa? (hiện tại: {settings.ollama_base_url})"
        )
    if "openai" in name_lower:
        has_key = bool(settings.openai_api_key and settings.openai_api_key != "sk-...")
        return f"\n→ OPENAI_API_KEY đã set trong .env? (hiện tại: {'Có' if has_key else 'CHƯA CÓ'})"
    if "google" in name_lower:
        has_key = bool(settings.google_api_key and settings.google_api_key != "AI...")
        return f"\n→ GOOGLE_API_KEY đã set trong .env? (hiện tại: {'Có' if has_key else 'CHƯA CÓ'})"
    return ""


def _parse_provider_and_model(model_str: str) -> tuple[str | None, str]:
    """
    Phân tích chuỗi model thành (provider, model_name).
    Ví dụ:
        'ollama/gemma3:12b' -> ('ollama', 'gemma3:12b')
        'openai/gpt-4o'     -> ('openai', 'gpt-4o')
        'gpt-4o'            -> (None, 'gpt-4o')
    """
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
        return provider.strip().lower(), model_name.strip()
    return None, model_str.strip()


def get_llm(
    model_role: str = "default",
    temperature: float = 0.0,
    timeout: float | None = None,
    max_retries: int = 2,
    use_cache: bool = True,
    **kwargs,
) -> BaseChatModel:
    """
    Khởi tạo LLM theo vai trò hoặc model string có hỗ trợ cache instance.

    Args:
        model_role: "strong" | "fast" | "local" | "default" | model string trực tiếp
        temperature: 0.0 – 1.0
        timeout: Thời gian timeout tính bằng giây (mặc định 120s cho local, None cho cloud)
        max_retries: Số lần thử lại khi gặp lỗi mạng/rate limit
        use_cache: Bật/tắt tái sử dụng instance đã tạo

    Returns:
        BaseChatModel sẵn sàng gọi .invoke() / .stream()
    """
    raw_model_name = _resolve_model_name(model_role)
    provider, model_name = _parse_provider_and_model(raw_model_name)

    # Đặt timeout mặc định cho local model nếu chưa truyền
    effective_timeout = timeout
    if effective_timeout is None and provider == "ollama":
        effective_timeout = 120.0

    cache_key = (raw_model_name, temperature, effective_timeout, max_retries)
    if use_cache and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    call_kwargs: dict[str, Any] = dict(kwargs)
    if provider == "ollama":
        if "base_url" not in call_kwargs:
            call_kwargs["base_url"] = settings.ollama_base_url
        if effective_timeout is not None and "timeout" not in call_kwargs:
            call_kwargs["timeout"] = effective_timeout
    elif effective_timeout is not None and "timeout" not in call_kwargs:
        call_kwargs["timeout"] = effective_timeout

    if "max_retries" not in call_kwargs:
        call_kwargs["max_retries"] = max_retries

    try:
        if provider:
            llm = init_chat_model(
                model_name,
                model_provider=provider,
                temperature=temperature,
                **call_kwargs,
            )
        else:
            llm = init_chat_model(
                model_name,
                temperature=temperature,
                **call_kwargs,
            )
    except Exception as e:
        error_msg = (
            f"Không khởi tạo được model '{raw_model_name}' (role='{model_role}'): {e}"
            + _build_hint(raw_model_name)
        )
        logger.error(error_msg)
        raise LLMInitError(error_msg) from e

    logger.info("Khởi tạo LLM thành công: %s (role=%s, timeout=%s)", raw_model_name, model_role, effective_timeout)

    if use_cache:
        _LLM_CACHE[cache_key] = llm

    return llm


def clear_llm_cache():
    """Xóa sạch cache instance LLM (dùng cho test hoặc khi đổi env động)."""
    _LLM_CACHE.clear()


def get_available_models() -> dict[str, str]:
    """Trả về registry kèm default: role → model name."""
    return {**MODEL_REGISTRY, "default": settings.default_model}


def check_ollama_status() -> dict:
    """Kiểm tra Ollama server đồng bộ (dùng trong script/startup)."""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "ok", "models": models}
        return {"status": "error", "detail": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"status": "offline", "detail": f"Không kết nối được {settings.ollama_base_url}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def check_ollama_status_async() -> dict:
    """Kiểm tra Ollama server bất đồng bộ (dùng cho FastAPI endpoint để không chặn event loop)."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "error", "detail": f"HTTP {r.status_code}"}
    except httpx.ConnectError:
        return {"status": "offline", "detail": f"Không kết nối được {settings.ollama_base_url}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def verify_model(model_role: str) -> dict:
    """
    Kiểm tra model có hoạt động không bằng cách gọi thử.
    Dùng cho health check nâng cao và debug.
    """
    model_name = _resolve_model_name(model_role)
    try:
        llm = get_llm(model_role)
        response = llm.invoke("ping")
        content = response.content if hasattr(response, "content") else str(response)
        return {
            "role": model_role,
            "model": model_name,
            "status": "ok",
            "response_preview": content[:80],
        }
    except LLMInitError as e:
        return {
            "role": model_role,
            "model": model_name,
            "status": "error",
            "error": str(e),
        }
