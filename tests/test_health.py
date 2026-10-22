"""
Unit tests cho health check endpoints (được cô lập hoàn toàn khỏi network ngoài).
"""

from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings

client = TestClient(app)


def test_liveness_endpoint():
    """Liveness probe phải trả lời 200 OK ngay lập tức."""
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
def test_health_check_healthy(mock_ollama):
    """Khi Ollama hoạt động bình thường, endpoint trả status 'ok'."""
    mock_ollama.return_value = {"status": "ok", "models": ["gemma3:12b"]}

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert data["default_model"] == settings.default_model
    assert "strong" in data["available_roles"]
    assert "fast" in data["available_roles"]
    assert "local" in data["available_roles"]
    assert data["ollama_status"]["status"] == "ok"


@patch("app.api.health.check_ollama_status_async", new_callable=AsyncMock)
def test_health_check_degraded_when_ollama_offline(mock_ollama):
    """Khi Ollama offline, endpoint trả status 'degraded' thay vì 'ok'."""
    mock_ollama.return_value = {
        "status": "offline",
        "detail": "Không kết nối được http://localhost:11434",
    }

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["ollama_status"]["status"] == "offline"


def test_verify_model_forbidden_in_production(monkeypatch):
    """Khi settings.debug = False, endpoint verify-model phải trả về HTTP 403 Forbidden."""
    monkeypatch.setattr(settings, "debug", False)
    response = client.get("/api/health/verify-model?role=local")
    assert response.status_code == 403
    assert "development/debug" in response.json()["detail"]
