from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_model_version() -> None:
    response = client.get("/health")

    assert response.json()["model_version"] == "mock-video-ai-v1"
    assert response.json()["service"] == "ai-video-detection-ai-service"
