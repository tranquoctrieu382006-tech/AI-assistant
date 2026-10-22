"""
Unit tests cho health check endpoints.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_liveness_endpoint():
    """Liveness probe phải trả lời 200 OK ngay lập tức."""
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_check_endpoint():
    """Health check đầy đủ trả về đúng schema và thông tin runtime."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "default_model" in data
    assert "available_roles" in data
    assert "strong" in data["available_roles"]
    assert "fast" in data["available_roles"]
    assert "local" in data["available_roles"]
    assert "langfuse_enabled" in data
    assert "ollama_status" in data
