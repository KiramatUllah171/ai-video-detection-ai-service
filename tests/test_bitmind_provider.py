from pathlib import Path
import subprocess

import httpx
import pytest

from app.core.config import Settings
from app.models.requests import AnalyzeFrameItem, AnalyzeVideoRequest
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult
from app.services.providers import BitMindAiProvider, BitMindProviderError, LocalAiProvider, ProviderOrchestrator


class FakeHttpClient:
    last_headers = {}
    last_url = None
    sent_file_name = None
    sent_file_bytes = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, **kwargs):
        FakeHttpClient.last_url = url
        FakeHttpClient.last_headers = headers or {}
        files = kwargs.get("files") or {}
        if "video" in files:
            file_name, file_handle, _content_type = files["video"]
            FakeHttpClient.sent_file_name = file_name
            FakeHttpClient.sent_file_bytes = file_handle.read()
            file_handle.seek(0)
        return httpx.Response(200, json={"isAI": True, "confidence": 0.90, "similarity": 0, "request_id": "req-1"})


class ModerateAiHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(200, json={"isAI": True, "confidence": 0.7529020309448242, "similarity": 0})


class LikelyRealHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(200, json={"isAI": False, "confidence": 0.90, "similarity": 0})


class ModerateRealHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(200, json={"isAI": False, "confidence": 0.70, "similarity": 0})


class MissingConfidenceHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(200, json={"isAI": True, "similarity": 0})


class FailingHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(500, json={"error": {"message": "provider failed"}})


class UnauthorizedHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})


class SlowHttpClient(FakeHttpClient):
    def post(self, url, headers=None, **kwargs):
        raise httpx.TimeoutException("timeout")


class StaticLocalService:
    model_loaded = True
    model_capability = "mock"

    def analyze_frames(self, request):
        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id="local",
            model_version="local-v1",
            model_capability="mock",
            is_mock=True,
            overall_ai_score=0.42,
            real_probability=0.58,
            overall_confidence=0.64,
            label_hint="Inconclusive",
            frames=[
                FrameAnalysisResult(
                    frame_id=1,
                    frame_index=0,
                    timestamp_seconds=0,
                    ai_score=0.42,
                    real_probability=0.58,
                    confidence=0.64,
                    notes=["local"],
                )
            ],
            notes=["local"],
            warnings=[],
        )


def make_settings(**overrides):
    values = {
        "bitmind_enabled": True,
        "bitmind_api_key": "secret-test-key",
        "bitmind_base_url": "https://api.bitmind.ai",
        "local_fallback_enabled": True,
        "external_provider_policy": "Always",
    }
    values.update(overrides)
    return Settings(**values)


def make_request(path: Path):
    return AnalyzeVideoRequest(
        video_id=1,
        job_id=2,
        provider_mode="bitmind",
        original_video_path=str(path),
        frames=[
            AnalyzeFrameItem(frame_id=1, frame_url="frames/1.jpg", frame_index=0, image_base64="abc")
        ],
    )


def test_bitmind_client_builds_authenticated_request_without_exposing_key(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert FakeHttpClient.last_url == "https://api.bitmind.ai/detect-video"
    assert FakeHttpClient.last_headers["Authorization"] == "Bearer secret-test-key"
    assert FakeHttpClient.last_headers["x-bitmind-application"] == "oracle-api"
    assert "secret-test-key" not in response.model_dump_json()


def test_bitmind_client_normalizes_legacy_base_url(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    BitMindAiProvider(make_settings(bitmind_base_url="https://api.bitmind.ai/34")).analyze_video(make_request(video))

    assert FakeHttpClient.last_url == "https://api.bitmind.ai/detect-video"


def test_bitmind_rejects_local_file_outside_allowed_roots(tmp_path):
    allowed_root = tmp_path / "allowed"
    outside_root = tmp_path / "outside"
    allowed_root.mkdir()
    outside_root.mkdir()
    video = outside_root / "video.mp4"
    video.write_bytes(b"video")

    provider = BitMindAiProvider(make_settings(ai_allowed_video_roots_csv=str(allowed_root)))

    with pytest.raises(BitMindProviderError) as exception:
        provider.analyze_video(make_request(video))

    assert "outside the allowed analysis work directories" in exception.value.message


def test_bitmind_requires_allowed_roots_for_local_paths_in_production(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    provider = BitMindAiProvider(make_settings(app_env="production", ai_allowed_video_roots_csv=""))

    with pytest.raises(BitMindProviderError) as exception:
        provider.analyze_video(make_request(video))

    assert "AI_ALLOWED_VIDEO_ROOTS" in exception.value.message


def test_bitmind_response_normalization_maps_common_schema(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.provider == "BitMind"
    assert response.external_provider_result["provider_status"] == "Completed"
    assert response.overall_ai_score == pytest.approx(0.90)
    assert response.real_probability == pytest.approx(0.10)
    assert response.external_provider_result["provider_score"] == pytest.approx(0.90)
    assert response.label_hint == "LikelyAiGenerated"
    assert response.final_decision_source == "BitMind"


def test_bitmind_moderate_ai_confidence_is_suspicious_not_likely_ai(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", ModerateAiHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.overall_ai_score == pytest.approx(0.7529)
    assert response.real_probability == pytest.approx(0.2471)
    assert response.external_provider_result["provider_score"] == pytest.approx(0.7529020309448242)
    assert response.external_provider_result["provider_decision_band"] == "ai_moderate_confidence"
    assert response.label_hint == "Suspicious"
    assert response.label_hint != "LikelyAiGenerated"


def test_bitmind_likely_real_uses_inverse_confidence_not_similarity(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", LikelyRealHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.overall_ai_score == pytest.approx(0.10)
    assert response.real_probability == pytest.approx(0.90)
    assert response.external_provider_result["provider_score"] == pytest.approx(0.10)
    assert response.label_hint == "LikelyReal"


def test_bitmind_moderate_real_confidence_is_inconclusive(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", ModerateRealHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.overall_ai_score == pytest.approx(0.30)
    assert response.real_probability == pytest.approx(0.70)
    assert response.external_provider_result["provider_score"] == pytest.approx(0.30)
    assert response.external_provider_result["provider_decision_band"] == "real_moderate_confidence"
    assert response.label_hint == "Inconclusive"


def test_bitmind_similarity_is_not_used_as_ai_score(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.external_provider_result["provider_raw_response"]["similarity"] == 0
    assert response.overall_ai_score != 0


def test_bitmind_missing_confidence_does_not_default_to_zero(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", MissingConfidenceHttpClient)

    response = BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert response.overall_ai_score == pytest.approx(0.5)
    assert response.external_provider_result["provider_score"] is None
    assert response.external_provider_result["provider_confidence"] is None


def test_bitmind_failed_status_returns_local_fallback(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", FailingHttpClient)
    orchestrator = ProviderOrchestrator(LocalAiProvider(StaticLocalService()), BitMindAiProvider(make_settings()), make_settings())

    response = orchestrator.analyze_video(make_request(video))

    assert response.fallback_used is True
    assert response.final_decision_source == "FallbackLocal"
    assert response.external_provider_result["provider_status"] == "Failed"


def test_bitmind_unauthorized_uses_auth_error_code_and_safe_message(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", UnauthorizedHttpClient)

    with pytest.raises(BitMindProviderError) as exception:
        BitMindAiProvider(make_settings()).analyze_video(make_request(video))

    assert exception.value.status_code == 401
    assert exception.value.error_code == "BITMIND_AUTH_FAILED"
    assert exception.value.message == "provider authentication failed."


def test_bitmind_only_route_returns_safe_error_message(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from app.api.routes import analyze
    from app.main import app

    def fail_bitmind(self, request):
        raise BitMindProviderError("BitMind returned HTTP 401.", 401, {"message": "unauthorized"})

    monkeypatch.setattr(analyze, "settings", make_settings(local_fallback_enabled=False))
    monkeypatch.setattr(BitMindAiProvider, "analyze_video", fail_bitmind)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    response = TestClient(app).post(
        "/analyze-video",
        json={
            "video_id": 1,
            "job_id": 2,
            "provider_mode": "bitmind",
            "original_video_path": str(video),
            "frames": [{"frame_id": 1, "frame_url": "frames/1.jpg", "frame_index": 0, "image_base64": "abc"}],
        },
    )

    body = response.json()
    assert response.status_code == 502
    assert body["error_code"] == "BITMIND_UNAVAILABLE"
    assert body["message"] == "External video analysis is temporarily unavailable."
    assert "HTTP 401" not in body["message"]


def test_bitmind_timeout_returns_provider_error_fallback(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr("app.services.providers.httpx.Client", SlowHttpClient)
    orchestrator = ProviderOrchestrator(LocalAiProvider(StaticLocalService()), BitMindAiProvider(make_settings()), make_settings())

    response = orchestrator.analyze_video(make_request(video))

    assert response.fallback_used is True
    assert "failed/unavailable" in " ".join(response.warnings)


def test_bitmind_under_limit_sends_original_file(monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    original_bytes = b"original-video"
    video.write_bytes(original_bytes)
    FakeHttpClient.sent_file_name = None
    FakeHttpClient.sent_file_bytes = None
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    response = BitMindAiProvider(make_settings(bitmind_max_upload_bytes=200, bitmind_direct_upload_limit_bytes=200)).analyze_video(make_request(video))

    assert response.external_provider_result["compression_used"] is False
    assert response.external_provider_result["provider_sent_file_name"] == "video.mp4"
    assert FakeHttpClient.sent_file_name == "video.mp4"
    assert FakeHttpClient.sent_file_bytes == original_bytes
    assert video.read_bytes() == original_bytes


def test_bitmind_over_limit_compresses_sends_copy_and_cleans_up(monkeypatch, tmp_path):
    video = tmp_path / "video.mov"
    original_bytes = b"x" * 20
    compressed_bytes = b"mp4"
    video.write_bytes(original_bytes)
    FakeHttpClient.sent_file_name = None
    FakeHttpClient.sent_file_bytes = None
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)

    def fake_run(command, capture_output=True, text=True, timeout=None):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="10.0\n", stderr="")
        output_path = Path(command[-1])
        output_path.write_bytes(compressed_bytes)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("app.services.providers.subprocess.run", fake_run)

    response = BitMindAiProvider(make_settings(
        bitmind_max_upload_bytes=10,
        bitmind_compression_target_bytes=8,
        bitmind_direct_upload_limit_bytes=10,
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
    )).analyze_video(make_request(video))

    analysis_copy_path = Path(response.external_provider_result["analysis_copy_path"])
    assert response.external_provider_result["compression_used"] is True
    assert response.external_provider_result["compression_attempts"] == 1
    assert response.external_provider_result["original_file_size_bytes"] == len(original_bytes)
    assert response.external_provider_result["bitmind_file_size_bytes"] == len(compressed_bytes)
    assert response.external_provider_result["provider_sent_file_name"].endswith(".mp4")
    assert FakeHttpClient.sent_file_name == response.external_provider_result["provider_sent_file_name"]
    assert FakeHttpClient.sent_file_bytes == compressed_bytes
    assert video.read_bytes() == original_bytes
    assert not analysis_copy_path.exists()
    assert "compressed analysis copy" in " ".join(response.warnings)


def test_bitmind_compression_retries_lower_bitrate(monkeypatch, tmp_path):
    video = tmp_path / "video.mov"
    video.write_bytes(b"x" * 20)
    monkeypatch.setattr("app.services.providers.httpx.Client", FakeHttpClient)
    ffmpeg_outputs = [b"x" * 11, b"x" * 9]
    ffmpeg_commands = []

    def fake_run(command, capture_output=True, text=True, timeout=None):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="10.0\n", stderr="")
        ffmpeg_commands.append(command)
        output_path = Path(command[-1])
        output_path.write_bytes(ffmpeg_outputs.pop(0))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("app.services.providers.subprocess.run", fake_run)

    response = BitMindAiProvider(make_settings(
        bitmind_max_upload_bytes=10,
        bitmind_compression_target_bytes=2 * 1024 * 1024,
        bitmind_direct_upload_limit_bytes=10,
    )).analyze_video(make_request(video))

    assert response.external_provider_result["compression_attempts"] == 2
    assert len(ffmpeg_commands) == 2
    assert ffmpeg_commands[0][ffmpeg_commands[0].index("-b:v") + 1] != ffmpeg_commands[1][ffmpeg_commands[1].index("-b:v") + 1]


def test_bitmind_compression_failure_falls_back_to_local(monkeypatch, tmp_path):
    video = tmp_path / "video.mov"
    original_bytes = b"x" * 20
    video.write_bytes(original_bytes)

    def fake_run(command, capture_output=True, text=True, timeout=None):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, stdout="10.0\n", stderr="")
        output_path = Path(command[-1])
        output_path.write_bytes(b"x" * 20)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("app.services.providers.subprocess.run", fake_run)
    orchestrator = ProviderOrchestrator(
        LocalAiProvider(StaticLocalService()),
        BitMindAiProvider(make_settings(
            bitmind_max_upload_bytes=10,
            bitmind_compression_target_bytes=8,
            bitmind_direct_upload_limit_bytes=10,
        )),
        make_settings(local_fallback_enabled=True),
    )

    response = orchestrator.analyze_video(make_request(video))

    assert response.fallback_used is True
    assert response.fallback_reason == "Video exceeded BitMind upload limit and compression failed. Local analysis was used."
    assert response.external_provider_result["provider_status"] == "Failed"
    assert response.external_provider_result["compression_used"] is True
    assert response.external_provider_result["compression_error"]
    assert video.read_bytes() == original_bytes
