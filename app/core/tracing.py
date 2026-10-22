"""
Langfuse tracing integration.

Tạo và quản lý Langfuse CallbackHandler cho LangChain/LangGraph.
Hỗ trợ session_id, user_id, tags và metadata cho mỗi request.
"""

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


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

    Nếu Langfuse chưa được cấu hình, trả về None (không crash app).

    Ví dụ:
        handler = get_langfuse_handler(
            session_id="session-123",
            user_id="intern_trieu",
            tags=["model-comparison", "local"]
        )
        if handler:
            callbacks = [handler]
            llm.invoke("...", config={"callbacks": callbacks})
            handler.flush()
    """
    if not is_langfuse_configured():
        logger.debug("Langfuse chưa cấu hình key hợp lệ, bỏ qua tracing.")
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        return handler
    except Exception as e:
        logger.warning("Không thể khởi tạo Langfuse CallbackHandler: %s", e)
        return None
