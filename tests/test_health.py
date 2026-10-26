"""HTTP-level health route checks using an in-process ASGI transport."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import psycopg
import pytest

from app.core.config import settings
from app.main import app
from app.api.health import _database_readiness


async def request(
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    client: tuple[str, int] = ("127.0.0.1", 12345),
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, client=client)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, headers=headers)


def test_database_readiness_checks_pgvector(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "postgresql://test/db")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=("0.8.7",))),
    ]
    with patch("app.api.health.vector_store._connect") as connect, \
         patch("app.api.health.vector_store._register_vector_if_needed"):
        connect.return_value.__enter__.return_value = connection
        assert _database_readiness() == {"status": "ok", "pgvector_version": "0.8.7"}


@pytest.mark.asyncio
async def test_liveness_endpoint():
    response = await request("GET", "/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
async def test_health_check_healthy(mock_ollama, monkeypatch):
    mock_ollama.return_value = {"status": "ok", "models": ["gemma3:12b"]}
    monkeypatch.setattr(settings, "database_url", "postgresql://health-test")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=("0.8.7",))),
    ]
    with patch("app.api.health.vector_store._connect") as connect, \
         patch("app.api.health.vector_store._register_vector_if_needed"), \
         patch("app.api.health._nlq_database_readiness", return_value={"status": "ok"}):
        connect.return_value.__enter__.return_value = connection
        response = await request("GET", "/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert data["default_model"] == settings.default_model
    assert {"strong", "fast", "local"}.issubset(data["available_roles"])
    assert data["ollama_status"]["status"] == "ok"
    assert "models" not in data["ollama_status"]
    assert data["database_status"]["pgvector_version"] == "0.8.7"


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
async def test_health_degraded_when_required_local_model_is_offline(mock_ollama, monkeypatch):
    mock_ollama.return_value = {"status": "offline", "detail": "offline"}
    monkeypatch.setattr(settings, "database_url", "postgresql://health-test")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=("0.8.7",))),
    ]
    monkeypatch.setattr(settings, "rag_model", "local")
    with patch("app.api.health.vector_store._connect") as connect, \
         patch("app.api.health.vector_store._register_vector_if_needed"), \
         patch("app.api.health._nlq_database_readiness", return_value={"status": "ok"}):
        connect.return_value.__enter__.return_value = connection
        data = (await request("GET", "/api/health")).json()
    assert data["status"] == "degraded"
    assert data["database_status"]["status"] == "ok"


@pytest.mark.asyncio
@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
async def test_health_requires_ollama_for_cloud_rag_embeddings(mock_ollama, monkeypatch):
    mock_ollama.return_value = {"status": "offline", "detail": "offline"}
    monkeypatch.setattr(settings, "database_url", "postgresql://health-test")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=("0.8.7",))),
    ]
    monkeypatch.setattr(settings, "rag_model", "strong")
    with patch("app.api.health.vector_store._connect") as connect, \
         patch("app.api.health.vector_store._register_vector_if_needed"), \
         patch("app.api.health._nlq_database_readiness", return_value={"status": "ok"}):
        connect.return_value.__enter__.return_value = connection
        data = (await request("GET", "/api/health")).json()
    assert data["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_degraded_when_nlq_database_is_not_configured(monkeypatch):
    mock_ollama = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr("app.api.health.check_ollama_status_async", mock_ollama)
    monkeypatch.setattr(settings, "database_url", "postgresql://health-test")
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=("vector", "rag_chunks"))),
        MagicMock(fetchone=MagicMock(return_value=("0.8.7",))),
    ]
    with patch("app.api.health.vector_store._connect") as connect, \
         patch("app.api.health.vector_store._register_vector_if_needed"), \
         patch("app.api.health._nlq_database_readiness", return_value={"status": "not_configured"}):
        connect.return_value.__enter__.return_value = connection
        response = await request("GET", "/api/health")
    assert response.json()["status"] == "degraded"
    assert response.json()["nlq_database_status"]["status"] == "not_configured"


@pytest.mark.asyncio
async def test_nlq_schema_database_connection_error_returns_503(monkeypatch):
    monkeypatch.setattr(settings, "nlq_database_url", "postgresql://nlq-test")
    transport = httpx.ASGITransport(app=app)
    with patch("app.api.nlq.ask_nlq", side_effect=psycopg.OperationalError("database unavailable")):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/nlq/ask",
                headers={"X-API-Key": settings.app_api_key},
                json={"question": "Có bao nhiêu đơn hàng?"},
            )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_verify_model_forbidden_in_production(monkeypatch):
    monkeypatch.setattr(settings, "debug", False)
    response = await request(
        "GET", "/api/health/verify-model?role=local",
        headers={"X-API-Key": settings.app_api_key},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_verify_model_rejects_unregistered_model_even_in_debug(monkeypatch):
    monkeypatch.setattr(settings, "debug", True)
    response = await request(
        "GET", "/api/health/verify-model?role=openai%2Fgpt-secret",
        headers={"X-API-Key": settings.app_api_key},
    )
    assert response.status_code == 422
