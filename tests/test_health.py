from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_model_version() -> None:
    response = client.get("/health")

    assert response.json()["model_version"]
    assert response.json()["service"] == "ai-video-detection-ai-service"
    assert response.json()["ai_mode"] in {"mock", "real"}
    assert response.json()["provider_mode"]


def test_model_diagnostics_returns_safe_model_info() -> None:
    response = client.get("/model-diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["ai_mode"]
    assert body["provider_mode"]
    assert "is_mock" in body
    assert "fallback_used" in body
    assert "allow_model_download" in body
