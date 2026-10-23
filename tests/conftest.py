import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def isolate_optional_langfuse_credentials(monkeypatch):
    """Keep local tests independent of a developer's real .env secrets."""
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
