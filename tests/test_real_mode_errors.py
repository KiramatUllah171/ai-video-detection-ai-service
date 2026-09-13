from dataclasses import replace

from app.core.config import settings
from app.models.requests import AnalyzeFrameItem, AnalyzeFramesRequest
from app.services.local_video_ai_service import LocalVideoAiService
from app.services.model_loader import ModelLoader
from app.utils.errors import AiServiceError


def test_real_mode_missing_model_returns_real_model_not_configured() -> None:
    real_settings = replace(settings, ai_mode="real", enable_mock_fallback=False)
    service = LocalVideoAiService(real_settings, model=None, processor=None)
    request = AnalyzeFramesRequest(
        video_id=1,
        job_id=2,
        frames=[AnalyzeFrameItem(frame_id=1, frame_url="frame.jpg", frame_index=0, image_base64="abc")],
    )

    try:
        service.analyze_frames(request)
    except AiServiceError as exception:
        assert exception.error_code == "REAL_MODEL_NOT_CONFIGURED"
    else:
        raise AssertionError("Expected REAL_MODEL_NOT_CONFIGURED")


def test_mock_fallback_returns_mock_service() -> None:
    real_settings = replace(settings, ai_mode="real", enable_mock_fallback=True)
    loader = ModelLoader(real_settings)
    service = loader._real_model_failure("MODEL_LOAD_FAILED", "failed")

    assert service.model_capability == "mock"


def test_production_rejects_mock_ai_mode() -> None:
    production_settings = replace(
        settings,
        app_env="production",
        ai_service_api_key="x" * 32,
        ai_mode="mock",
        enable_mock_fallback=False,
        provider_mode="local",
        external_provider_policy="OnUncertain",
    )

    assert "AI_MODE=mock is not allowed in production." in production_settings.production_configuration_errors()


def test_production_accepts_real_hybrid_configuration() -> None:
    production_settings = replace(
        settings,
        app_env="production",
        ai_service_api_key="x" * 32,
        ai_mode="real",
        provider_mode="hybrid",
        external_provider_policy="OnUncertain",
        bitmind_enabled=True,
        bitmind_api_key="bitmind-secret",
        ai_allowed_video_roots_csv="D:/app/storage/work",
        enable_mock_fallback=False,
        ai_debug_output=False,
    )

    assert production_settings.production_configuration_errors() == []
