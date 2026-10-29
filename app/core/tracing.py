"""
Langfuse tracing integration.

Tạo và quản lý Langfuse CallbackHandler cho LangChain/LangGraph.
Hỗ trợ session_id, user_id, tags và metadata cho mỗi request.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def _trace_context(client, trace_id: str, parent_span_id: str | None = None) -> dict[str, str]:
    """Build an explicit Langfuse context so child observations stay nested.

    Passing only ``trace_id`` makes Langfuse invent a random remote parent and
    mark the observation as a root, which flattens the trace tree.
    """
    context = {"trace_id": trace_id}
    if parent_span_id:
        context["parent_span_id"] = parent_span_id
        return context
    try:
        current = client.get_current_observation_id()
    except Exception:
        current = None
    if current:
        context["parent_span_id"] = current
    return context


@contextmanager
def langfuse_observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
):
    """Create a Langfuse observation when configured without blocking local use."""
    if not is_langfuse_configured():
        yield None
        return
    try:
        import langfuse
        client = langfuse.get_client()
        obs_kwargs = {
            "as_type": as_type,
            "name": name,
            "input": input,
        }
        if trace_id:
            obs_kwargs["trace_context"] = _trace_context(client, trace_id, parent_span_id)
        observation = client.start_as_current_observation(**obs_kwargs)
        span = observation.__enter__()
    except Exception as exc:
        logger.warning("Không thể khởi tạo Langfuse observation '%s': %s", name, exc)
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        try:
            observation.__exit__(type(exc), exc, exc.__traceback__)
        except Exception as tracing_error:
            logger.warning("Không thể kết thúc Langfuse observation '%s': %s", name, tracing_error)
        raise
    else:
        try:
            observation.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("Không thể kết thúc Langfuse observation '%s': %s", name, exc)


def start_langfuse_observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
):
    """Start a span that is *not* attached to the current context.

    This is intended for streaming generators, whose lifetime outlives the
    request coroutine and may be closed from a copied context. Manually ending
    the returned observation avoids OpenTelemetry detach errors while still
    keeping child observations linked through ``parent_span_id``.
    """
    if not is_langfuse_configured():
        return None
    try:
        import langfuse
        client = langfuse.get_client()
        obs_kwargs = {"as_type": as_type, "name": name, "input": input}
        if trace_id:
            obs_kwargs["trace_context"] = _trace_context(client, trace_id, parent_span_id)
        return client.start_observation(**obs_kwargs)
    except Exception as exc:
        logger.warning("Không thể khởi tạo Langfuse observation '%s': %s", name, exc)
        return None


def end_langfuse_observation(observation, **kwargs) -> None:
    """Update and end a manually started observation, ignoring tracing errors."""
    if observation is None:
        return
    try:
        if kwargs:
            observation.update(**kwargs)
        observation.end()
    except Exception as exc:
        logger.warning("Không thể kết thúc Langfuse observation: %s", exc)


def create_trace_id() -> str | None:
    """Tạo trước một trace_id hợp lệ (32 ký tự hex) khi Langfuse được cấu hình."""
    if not is_langfuse_configured():
        return None
    try:
        client = get_langfuse_client()
        if client:
            return client.create_trace_id()
    except Exception as exc:
        logger.warning("Không thể sinh trace_id từ Langfuse: %s", exc)
    return None


def update_langfuse_observation(observation, **kwargs) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception as exc:
        logger.warning("Không thể cập nhật Langfuse observation: %s", exc)


def init_tracing():
    """
    Khởi tạo các biến môi trường và cấu hình tracing.
    Được gọi tường minh trong FastAPI lifespan hoặc đầu các script,
    tránh phụ thuộc vào thứ tự import module.
    """
    if settings.langfuse_public_key and not settings.langfuse_public_key.startswith("pk-lf-..."):
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    if settings.langfuse_secret_key and not settings.langfuse_secret_key.startswith("sk-lf-..."):
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if settings.langfuse_host:
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host


def is_langfuse_configured() -> bool:
    """Kiểm tra xem Langfuse đã được cấu hình với key hợp lệ chưa."""
    return bool(
        settings.langfuse_secret_key
        and not settings.langfuse_secret_key.startswith("sk-lf-...")
        and settings.langfuse_public_key
        and not settings.langfuse_public_key.startswith("pk-lf-...")
    )


def get_langfuse_handler(trace_context: dict[str, str] | None = None):
    """
    Tạo một CallbackHandler mới cho request.

    Lưu ý: Trong Langfuse v5+, CallbackHandler tự động bắt ngữ cảnh
    (session_id, user_id, tags, metadata) từ context manager `langfuse_context()`.
    Truyền `trace_context` (kèm `parent_span_id`) khi cần gắn chuỗi LangChain
    vào một span cụ thể thay vì tạo trace gốc mới.
    """
    if not is_langfuse_configured():
        return None
    try:
        from langfuse.langchain import CallbackHandler
        if trace_context:
            return CallbackHandler(trace_context=trace_context)
        return CallbackHandler()
    except Exception as e:
        logger.warning("Không thể khởi tạo Langfuse CallbackHandler: %s", e)
        return None


def get_langfuse_client():
    """Return the SDK's process-wide client so its prompt cache is shared."""
    if not is_langfuse_configured():
        return None
    init_tracing()
    import langfuse

    return langfuse.get_client()


@contextmanager
def langfuse_context(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager an toàn thiết lập ngữ cảnh trace (session_id, user_id, tags).
    Tránh lỗi 'generator didn't stop after throw()' bằng cách chỉ bọc phần
    chuẩn bị context manager, để exception trong yield lan truyền tự nhiên.
    """
    if not is_langfuse_configured():
        yield
        return

    cm = None
    try:
        import langfuse
        cm = langfuse.propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug("Không thể tạo propagate_attributes: %s", e)

    if cm:
        with cm:
            yield
    else:
        yield


def flush_langfuse():
    """
    Gửi toàn bộ hàng đợi trace về server Langfuse Cloud.
    Trong production: Gọi tại sự kiện shutdown của FastAPI lifespan.
    Trong batch script: Gọi khi kết thúc script.
    """
    if not is_langfuse_configured():
        return
    try:
        import langfuse
        client = langfuse.get_client()
        if client:
            client.flush()
            logger.info("Đã flush thành công các trace về Langfuse Cloud.")
    except Exception as e:
        logger.warning("Lỗi khi flush Langfuse: %s", e)
