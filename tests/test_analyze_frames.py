from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def valid_payload() -> dict:
    return {
        "video_id": 123,
        "job_id": 456,
        "frames": [
            {
                "frame_id": 1,
                "frame_url": "frames/123/frame_000001.jpg",
                "frame_index": 1,
                "timestamp_seconds": 2.0,
            },
            {
                "frame_id": 2,
                "frame_url": "frames/123/frame_000002.jpg",
                "frame_index": 2,
                "timestamp_seconds": 4.0,
            },
        ],
    }


def test_analyze_frames_valid_request_returns_200() -> None:
    response = client.post("/analyze-frames", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == 123
    assert body["job_id"] == 456
    assert body["model_version"] == "mock-video-ai-v1"
    assert body["is_mock"] is True
    assert body["model_capability"] == "mock"
    assert len(body["frames"]) == 2
    assert "mock AI response" in body["notes"][0]


def test_same_input_returns_same_ai_score_every_time() -> None:
    first = client.post("/analyze-frames", json=valid_payload()).json()
    second = client.post("/analyze-frames", json=valid_payload()).json()

    assert first["overall_ai_score"] == second["overall_ai_score"]
    assert first["frames"][0]["ai_score"] == second["frames"][0]["ai_score"]


def test_ai_score_is_between_zero_and_one() -> None:
    body = client.post("/analyze-frames", json=valid_payload()).json()

    assert 0 <= body["overall_ai_score"] <= 1
    for frame in body["frames"]:
        assert 0 <= frame["ai_score"] <= 1


def test_confidence_is_between_zero_and_one() -> None:
    body = client.post("/analyze-frames", json=valid_payload()).json()

    assert 0 <= body["overall_confidence"] <= 1
    for frame in body["frames"]:
        assert 0 <= frame["confidence"] <= 1
        assert 0 <= frame["real_probability"] <= 1


def test_empty_frames_returns_validation_error() -> None:
    payload = valid_payload()
    payload["frames"] = []

    response = client.post("/analyze-frames", json=payload)

    assert response.status_code == 422


def test_missing_frame_url_returns_validation_error() -> None:
    payload = valid_payload()
    del payload["frames"][0]["frame_url"]

    response = client.post("/analyze-frames", json=payload)

    assert response.status_code == 422
