"""HTTP-level health route checks using an in-process ASGI transport."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.config import settings
from app.main import app
from app.api.health import _database_readiness


async def request(method: str, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path)


def test_database_readiness_checks_pgvector(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://test/db")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("0.8.7",)
    with patch("app.api.health.psycopg.connect", return_value=connection):
        assert _database_readiness() == {"status": "ok", "pgvector_version": "0.8.7"}


@pytest.mark.asyncio
async def test_liveness_endpoint():
    response = await request("GET", "/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
@patch("app.api.health.psycopg.connect")
async def test_health_check_healthy(mock_connect, mock_ollama):
    mock_ollama.return_value = {"status": "ok", "models": ["gemma3:12b"]}
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("0.8.7",)
    mock_connect.return_value = connection

    response = await request("GET", "/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert data["default_model"] == settings.default_model
    assert {"strong", "fast", "local"}.issubset(data["available_roles"])
    assert data["ollama_status"]["status"] == "ok"
    assert data["database_status"]["pgvector_version"] == "0.8.7"


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
@patch("app.api.health.psycopg.connect")
async def test_health_degraded_when_required_local_model_is_offline(mock_connect, mock_ollama, monkeypatch):
    mock_ollama.return_value = {"status": "offline", "detail": "offline"}
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("0.8.7",)
    mock_connect.return_value = connection
    monkeypatch.setattr(settings, "rag_model", "local")
    data = (await request("GET", "/api/health")).json()
    assert data["status"] == "degraded"
    assert data["database_status"]["status"] == "ok"


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
@patch("app.api.health.psycopg.connect")
async def test_health_does_not_require_ollama_for_cloud_rag(mock_connect, mock_ollama, monkeypatch):
    mock_ollama.return_value = {"status": "offline", "detail": "offline"}
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.return_value.fetchone.return_value = ("0.8.7",)
    mock_connect.return_value = connection
    monkeypatch.setattr(settings, "rag_model", "strong")
    data = (await request("GET", "/api/health")).json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_verify_model_forbidden_in_production(monkeypatch):
    monkeypatch.setattr(settings, "debug", False)
    response = await request("GET", "/api/health/verify-model?role=local")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_verify_model_rejects_unregistered_model_even_in_debug(monkeypatch):
    monkeypatch.setattr(settings, "debug", True)
    response = await request("GET", "/api/health/verify-model?role=openai%2Fgpt-secret")
    assert response.status_code == 422
