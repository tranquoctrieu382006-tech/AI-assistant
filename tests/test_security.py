import httpx
import pytest

from app.core.config import settings
from app.main import app


async def request(
    *,
    api_key: str | None = None,
    client: tuple[str, int] = ("192.0.2.42", 12345),
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=client)
    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return await http.post("/api/rag/ask", json={"question": "Câu hỏi hợp lệ?"}, headers=headers)


@pytest.mark.asyncio
async def test_protected_api_requires_a_valid_api_key(monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "test-api-key-with-more-than-32-characters")
    missing = await request(client=("192.0.2.1", 12345))
    invalid = await request(api_key="wrong", client=("192.0.2.2", 12345))
    assert missing.status_code == 401
    assert invalid.status_code == 401


@pytest.mark.asyncio
async def test_protected_api_fails_closed_when_key_is_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "")
    response = await request(client=("192.0.2.3", 12345))
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_api_rate_limit_is_applied_by_client_ip(monkeypatch):
    monkeypatch.setattr(settings, "app_api_key", "test-api-key-with-more-than-32-characters")
    monkeypatch.setattr(settings, "api_rate_limit_per_minute", 2)
    client = ("192.0.2.4", 12345)
    first = await request(client=client)
    second = await request(client=client)
    limited = await request(client=client)
    assert first.status_code == 401
    assert second.status_code == 401
    assert limited.status_code == 429
    assert limited.headers["retry-after"]
