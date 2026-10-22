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


def get_langfuse_handler():
    """
    Tạo một CallbackHandler mới cho request.

    Lưu ý: Trong Langfuse v5+, CallbackHandler tự động bắt ngữ cảnh
    (session_id, user_id, tags, metadata) từ context manager `langfuse_context()`.
    Do đó mỗi request nên tạo một instance riêng thay vì dùng singleton toàn cục.
    """
    if not is_langfuse_configured():
        return None

    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except Exception as e:
        logger.warning("Không thể khởi tạo Langfuse CallbackHandler: %s", e)
        return None


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
