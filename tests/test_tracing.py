"""
Unit tests cho Langfuse tracing & context manager.
"""

import pytest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from app.core.config import settings
from app.core.tracing import (
    is_langfuse_configured,
    langfuse_context,
    langfuse_observation,
    update_langfuse_observation,
)


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


def test_langfuse_observation_creates_named_span_when_enabled(monkeypatch):
    span = MagicMock()

    @contextmanager
    def start_as_current_observation(**kwargs):
        assert kwargs == {
            "as_type": "span",
            "name": "rag-retrieval",
            "input": {"question": "question"},
        }
        yield span

    client = MagicMock(start_as_current_observation=start_as_current_observation)
    langfuse_module = MagicMock(get_client=MagicMock(return_value=client))
    monkeypatch.setattr("app.core.tracing.is_langfuse_configured", lambda: True)
    with patch.dict("sys.modules", {"langfuse": langfuse_module}):
        with langfuse_observation("rag-retrieval", input={"question": "question"}) as observation:
            assert observation is span
            observation.update(output={"match_count": 1})
    span.update.assert_called_once_with(output={"match_count": 1})


def test_langfuse_observation_is_disabled_for_local_runs(monkeypatch):
    monkeypatch.setattr("app.core.tracing.is_langfuse_configured", lambda: False)
    with langfuse_observation("rag-retrieval") as observation:
        assert observation is None


def test_langfuse_observation_failures_do_not_break_rag(monkeypatch, caplog):
    monkeypatch.setattr("app.core.tracing.is_langfuse_configured", lambda: True)
    client = MagicMock()
    client.start_as_current_observation.side_effect = RuntimeError("trace unavailable")
    with patch.dict("sys.modules", {"langfuse": MagicMock(get_client=lambda: client)}):
        with langfuse_observation("rag-retrieval") as observation:
            assert observation is None
    update_langfuse_observation(MagicMock(update=MagicMock(side_effect=RuntimeError("flush failed"))),
                                output={"answer": "answer"})
    assert "trace unavailable" in caplog.text
    assert "flush failed" in caplog.text


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
