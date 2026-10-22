"""
Unit tests cho Langfuse tracing & context manager.
"""

import pytest
from app.core.config import settings
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


@pytest.mark.parametrize(
    "pub_key,sec_key,expected",
    [
        # Key giả định dạng hợp lệ (hoàn toàn là chuỗi dummy, không chứa phần nào của key thật)
        ("pk-lf-mock-valid-public-key-12345678", "sk-lf-mock-valid-secret-key-87654321", True),
        ("pk-lf-...", "sk-lf-...", False),                              # Placeholder
        ("", "", False),                                                # Rỗng
        ("pk-lf-mock-valid-public-key-12345678", "", False),             # Thiếu secret
        ("", "sk-lf-mock-valid-secret-key-87654321", False),             # Thiếu public
        ("pk-lf-...", "sk-lf-mock-valid-secret-key-87654321", False),    # 1 trong 2 là placeholder
    ],
)
def test_is_langfuse_configured_parametrized(monkeypatch, pub_key, sec_key, expected):
    """Kiểm tra độ chính xác của logic nhận diện key Langfuse với nhiều kịch bản."""
    monkeypatch.setattr(settings, "langfuse_public_key", pub_key)
    monkeypatch.setattr(settings, "langfuse_secret_key", sec_key)
    assert is_langfuse_configured() is expected
