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

# Đảm bảo các biến môi trường chuẩn của Langfuse được thiết lập từ Settings
if settings.langfuse_public_key and not settings.langfuse_public_key.startswith("pk-lf-..."):
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
if settings.langfuse_secret_key and not settings.langfuse_secret_key.startswith("sk-lf-..."):
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
if settings.langfuse_base_url:
    os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url


def is_langfuse_configured() -> bool:
    """Kiểm tra xem Langfuse đã được cấu hình với key hợp lệ chưa."""
    return bool(
        settings.langfuse_secret_key
        and not settings.langfuse_secret_key.startswith("sk-lf-...")
        and settings.langfuse_public_key
        and not settings.langfuse_public_key.startswith("pk-lf-...")
    )


def get_langfuse_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Tạo Langfuse CallbackHandler gắn vào LangChain config.
    Tương thích cả Langfuse v2 và Langfuse v5+.
    """
    if not is_langfuse_configured():
        logger.debug("Langfuse chưa cấu hình key hợp lệ, bỏ qua tracing.")
        return None

    try:
        # Langfuse v5+
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except ImportError:
        try:
            # Langfuse v2
            from langfuse.callback import CallbackHandler
            return CallbackHandler(
                secret_key=settings.langfuse_secret_key,
                public_key=settings.langfuse_public_key,
                host=settings.langfuse_base_url or settings.langfuse_host,
                session_id=session_id,
                user_id=user_id,
                tags=tags or [],
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning("Không thể khởi tạo Langfuse CallbackHandler: %s", e)
            return None
    except Exception as e:
        logger.warning("Lỗi khởi tạo CallbackHandler: %s", e)
        return None


@contextmanager
def langfuse_context(
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager thiết lập thuộc tính trace (session_id, user_id, tags)
    trước khi gọi chuỗi LangChain trong Langfuse v5.
    """
    if not is_langfuse_configured():
        yield
        return

    try:
        import langfuse
        with langfuse.propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            metadata=metadata,
        ):
            yield
    except Exception as e:
        logger.debug("Không thể propagate Langfuse attributes: %s", e)
        yield


def flush_langfuse():
    """Gửi toàn bộ hàng đợi trace về server Langfuse Cloud."""
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
