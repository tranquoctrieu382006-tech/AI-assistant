"""
Lớp cấu hình LLM đa mô hình.

Đổi mô hình chỉ bằng biến môi trường, không sửa code.
Sử dụng init_chat_model (LangChain native) để tự nhận provider từ prefix.

Cải tiến:
- Cache instance theo (model_name, temperature, timeout, max_retries, kwargs)
- Xử lý timeout theo provider (Ollama dùng client_kwargs, cloud dùng timeout trực tiếp)
- verify_model bắt mọi Exception để tránh HTTP 500
- Async check cho Ollama status
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
    return model_role  # Direct model names are used by configured RAG/model workflows.


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


def get_model_provider(model_role: str = "default") -> str | None:
    """Return the provider resolved for a role or configured model name."""
    raw_model_name = _resolve_model_name(model_role)
    provider, _ = _parse_provider_and_model(raw_model_name)
    return provider


def _make_hashable(val: Any) -> Any:
    """Chuyển đổi đệ quy các cấu trúc dữ liệu sang tuple có thể hash được."""
    if isinstance(val, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in val.items()))
    elif isinstance(val, (list, set, tuple)):
        return tuple(_make_hashable(v) for v in val)
    return val


def extract_message_text(message: Any, *, strip: bool = True) -> str:
    """Extract visible text from LangChain message content without metadata."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip() if strip else content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in (None, "text") and block.get("text"):
                    parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        text = "\n".join(parts)
        return text.strip() if strip else text
    text = str(content)
    return text.strip() if strip else text


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
        timeout: Thời gian timeout (giây). Mặc định lấy từ LLM_TIMEOUT_SECONDS.
        max_retries: Số lần thử lại khi gặp lỗi mạng/rate limit của provider cloud.
        use_cache: Bật/tắt tái sử dụng instance đã tạo
        **kwargs: Các tham số bổ sung truyền cho LLM (vd: num_predict, stop, ...)

    Returns:
        BaseChatModel sẵn sàng gọi .invoke() / .stream()
    """
    raw_model_name = _resolve_model_name(model_role)
    provider, model_name = _parse_provider_and_model(raw_model_name)

    # Keep local and hosted requests bounded unless explicitly overridden.
    effective_timeout = timeout
    if effective_timeout is None:
        effective_timeout = settings.llm_timeout_seconds

    # Khóa cache bao gồm toàn bộ kwargs đã được chuẩn hóa để tránh trả nhầm cấu hình cũ
    kwargs_hashable = _make_hashable(kwargs)
    cache_key = (raw_model_name, temperature, effective_timeout, max_retries, kwargs_hashable)
    if use_cache and cache_key in _LLM_CACHE:
        return _LLM_CACHE[cache_key]

    call_kwargs: dict[str, Any] = dict(kwargs)

    # Settings are loaded from .env but are not automatically exported to
    # os.environ. Pass provider credentials directly to LangChain so the
    # configured keys work for both API-backed providers.
    if provider == "google_genai" and settings.google_api_key:
        call_kwargs.setdefault("api_key", settings.google_api_key)
    elif provider == "openai" and settings.openai_api_key:
        call_kwargs.setdefault("api_key", settings.openai_api_key)
    elif provider == "groq" and settings.groq_api_key:
        call_kwargs.setdefault("api_key", settings.groq_api_key)

    # Xử lý tham số đặc thù cho từng provider
    if provider == "ollama":
        if "base_url" not in call_kwargs:
            call_kwargs["base_url"] = settings.ollama_base_url
        # ChatOllama cấu hình timeout thông qua client_kwargs
        if effective_timeout is not None:
            existing_client_kwargs = call_kwargs.get("client_kwargs", {})
            if "timeout" not in existing_client_kwargs:
                existing_client_kwargs["timeout"] = effective_timeout
            call_kwargs["client_kwargs"] = existing_client_kwargs
    else:
        # OpenAI / Cloud providers hỗ trợ trực tiếp timeout và max_retries
        if effective_timeout is not None and "timeout" not in call_kwargs:
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

    logger.info(
        "Khởi tạo LLM thành công: %s (role=%s, timeout=%s)",
        raw_model_name,
        model_role,
        effective_timeout,
    )

    if use_cache:
        _LLM_CACHE[cache_key] = llm

    return llm


def get_llm_with_fallbacks(
    primary_role: str = "fast",
    fallback_roles: list[str] | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """
    Khởi tạo LLM với cơ chế fallback tự động của LangChain (`with_fallbacks`).
    Nếu model chính (primary_role) gặp lỗi runtime như HTTP 429 hoặc timeout,
    hệ thống tự động chuyển tiếp truy vấn sang danh sách fallback_roles theo thứ tự.
    """
    if fallback_roles is None:
        fallback_roles = ["local"]

    primary_llm = get_llm(primary_role, temperature=temperature, **kwargs)
    fallbacks: list[BaseChatModel] = []
    for role in fallback_roles:
        if role and role != primary_role:
            try:
                fallbacks.append(get_llm(role, temperature=temperature, **kwargs))
            except Exception as e:
                logger.warning("Không thể khởi tạo fallback model %s: %s", role, e)

    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)
    return primary_llm


def get_rag_llm(
    model_role: str | None = None,
    temperature: float = 0.0,
    *,
    timeout: float | None = None,
    max_retries: int | None = None,
    with_fallback: bool = False,
    fallback_roles: list[str] | None = None,
) -> BaseChatModel:
    """Build the configured RAG model with bounded provider-specific retries and optional fallback."""
    model_role = model_role or settings.rag_model
    raw_model_name = _resolve_model_name(model_role)
    provider, _ = _parse_provider_and_model(raw_model_name)
    kwargs = {"num_ctx": settings.rag_num_ctx} if provider == "ollama" else {}
    if timeout is None:
        timeout = (
            settings.rag_local_model_timeout_seconds
            if provider == "ollama"
            else settings.rag_model_timeout_seconds
        )
    if max_retries is None:
        # Disable hidden SDK retries; the non-stream RAG route owns the single
        # explicit 429/503 retry and the stream has no safe replay point.
        max_retries = 0

    if with_fallback:
        return get_llm_with_fallbacks(
            primary_role=model_role,
            fallback_roles=fallback_roles or (["local"] if model_role != "local" else None),
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    return get_llm(
        model_role,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


def clear_llm_cache():
    """Xóa sạch cache instance LLM."""
    _LLM_CACHE.clear()


def get_available_models() -> dict[str, str]:
    """Trả về registry kèm default: role → model name."""
    return {**MODEL_REGISTRY, "default": settings.default_model}


def check_ollama_status() -> dict:
    """Kiểm tra Ollama server đồng bộ (dùng trong script chạy một lần)."""
    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return {"status": "ok", "models": models}
        return {"status": "error", "detail": f"HTTP {r.status_code}"}
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
        return {"status": "offline", "detail": f"Không kết nối được {settings.ollama_base_url}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def check_ollama_status_async() -> dict:
    """Kiểm tra Ollama server bất đồng bộ (không chặn event loop của FastAPI)."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return {"status": "ok", "models": models}
            return {"status": "error", "detail": f"HTTP {r.status_code}"}
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError):
        return {"status": "offline", "detail": f"Không kết nối được {settings.ollama_base_url}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def verify_model(model_role: str) -> dict:
    """
    Kiểm tra model có hoạt động không bằng cách gọi thử.
    Bắt toàn bộ Exception (bao gồm timeout, lỗi mạng, 401) để không làm crash HTTP 500.
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
    except Exception as e:
        return {
            "role": model_role,
            "model": model_name,
            "status": "error",
            "error": str(e),
        }
