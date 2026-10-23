import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def isolate_optional_langfuse_credentials(monkeypatch):
    """Keep local tests independent of a developer's real .env secrets."""
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    monkeypatch.setattr(settings, "app_api_key", "test-api-key-with-more-than-32-characters")
