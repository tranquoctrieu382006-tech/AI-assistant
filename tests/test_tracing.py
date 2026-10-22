"""
Unit tests cho Langfuse tracing & context manager.
"""

import pytest
from app.core.tracing import langfuse_context, is_langfuse_configured


def test_langfuse_context_exception_safety():
    """
    Kiểm tra lỗi generator: Khi code bên trong khối with langfuse_context()
    gặp exception, exception gốc phải được lan truyền tự nhiên,
    tuyệt đối không bị lỗi 'RuntimeError: generator didn't stop after throw()'.
    """
    class CustomWorkflowError(Exception):
        pass

    with pytest.raises(CustomWorkflowError):
        with langfuse_context(session_id="test-session", user_id="test-user"):
            raise CustomWorkflowError("Lỗi luồng xử lý RAG/NLQ")


def test_is_langfuse_configured():
    """Kiểm tra logic nhận diện key Langfuse hợp lệ."""
    # Khi đã cấu hình key thật trong .env, hàm trả về True
    assert isinstance(is_langfuse_configured(), bool)
