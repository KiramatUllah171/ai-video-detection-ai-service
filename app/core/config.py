import os
from dataclasses import dataclass
from pathlib import Path


def _provider_mode() -> str:
    return os.getenv("AI_PROVIDER_MODE", os.getenv("AI_PROVIDER", os.getenv("PROVIDER_MODE", "local"))).lower()


def _ai_provider() -> str:
    return os.getenv("AI_PROVIDER", os.getenv("PROVIDER_MODE", os.getenv("AI_PROVIDER_MODE", "local"))).lower()


@dataclass(frozen=True)
class Settings:
    service_name: str = "ai-video-detection-ai-service"
    app_env: str = os.getenv("APP_ENV", "development")
    ai_service_api_key: str | None = os.getenv("AI_SERVICE_API_KEY") or None
    ai_allowed_video_roots_csv: str = os.getenv("AI_ALLOWED_VIDEO_ROOTS", "")
    ai_mode: str = os.getenv("AI_MODE", "mock").lower()
    provider_mode: str = _provider_mode()
    ai_provider: str = _ai_provider()
    bitmind_enabled: bool = os.getenv("BITMIND_ENABLED", "false").lower() == "true"
    bitmind_api_key: str | None = os.getenv("BITMIND_API_KEY") or None
    bitmind_base_url: str = os.getenv("BITMIND_BASE_URL", "https://api.bitmind.ai/oracle/v1")
    bitmind_timeout_seconds: int = int(os.getenv("BITMIND_TIMEOUT_SECONDS", "180"))
    bitmind_monthly_quota: int = int(os.getenv("BITMIND_MONTHLY_QUOTA", "100"))
    bitmind_use_on_inconclusive: bool = os.getenv("BITMIND_USE_ON_INCONCLUSIVE", "true").lower() == "true"
    bitmind_use_on_suspicious: bool = os.getenv("BITMIND_USE_ON_SUSPICIOUS", "true").lower() == "true"
    bitmind_always_use_for_paid_users: bool = os.getenv("BITMIND_ALWAYS_USE_FOR_PAID_USERS", "false").lower() == "true"
    local_fallback_enabled: bool = os.getenv("LOCAL_FALLBACK_ENABLED", "true").lower() == "true"
    external_provider_policy: str = os.getenv("EXTERNAL_PROVIDER_POLICY", "OnUncertain")
    bitmind_direct_upload_limit_bytes: int = int(os.getenv("BITMIND_DIRECT_UPLOAD_LIMIT_BYTES", str(10 * 1024 * 1024)))
    bitmind_max_upload_bytes: int = int(os.getenv("BITMIND_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    bitmind_compression_target_bytes: int = int(os.getenv("BITMIND_COMPRESSION_TARGET_BYTES", str(190 * 1024 * 1024)))
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "ffmpeg")
    ffprobe_path: str = os.getenv("FFPROBE_PATH", "ffprobe")
    enable_mock_fallback: bool = os.getenv("ENABLE_MOCK_FALLBACK", "false").lower() == "true"
    model_id: str = os.getenv("MODEL_ID", "SoraExplora/VideoMae")
    fallback_video_model_id: str = os.getenv("FALLBACK_VIDEO_MODEL_ID", "Naman712/Deep-fake-detection")
    fallback_image_model_id: str = os.getenv("FALLBACK_IMAGE_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model")
    model_backend: str = os.getenv("MODEL_BACKEND", "transformers")
    model_version: str = os.getenv("MODEL_VERSION", "real-video-ai-v1")
    mock_model_version: str = "mock-video-ai-v1"
    device: str = os.getenv("DEVICE", "auto")
    frame_batch_size: int = int(os.getenv("FRAME_BATCH_SIZE", "8"))
    max_frames_per_request: int = int(os.getenv("MAX_FRAMES_PER_REQUEST", "30"))
    model_frame_count: int = int(os.getenv("MODEL_FRAME_COUNT", "16"))
    ensemble_enabled: bool = os.getenv("AI_ENSEMBLE_ENABLED", "false").lower() == "true"
    primary_video_model_id: str = os.getenv("PRIMARY_VIDEO_MODEL_ID", os.getenv("MODEL_ID", "SoraExplora/VideoMae"))
    frame_model_id: str = os.getenv("FRAME_MODEL_ID", "prithivMLmods/Deep-Fake-Detector-v2-Model")
    secondary_video_model_id: str = os.getenv("SECONDARY_VIDEO_MODEL_ID", "Naman712/Deep-fake-detection")
    ensemble_video_weight: float = float(os.getenv("ENSEMBLE_VIDEO_WEIGHT", "0.65"))
    ensemble_frame_weight: float = float(os.getenv("ENSEMBLE_FRAME_WEIGHT", "0.35"))
    frame_model_ids_csv: str = os.getenv(
        "FRAME_MODEL_IDS",
        "prithivMLmods/Deep-Fake-Detector-v2-Model,prithivMLmods/deepfake-detector-model-v1",
    )
    calibration_enabled: bool = os.getenv("CALIBRATION_ENABLED", os.getenv("FRAME_CALIBRATION_ENABLED", "true")).lower() == "true"
    calibration_minimum_samples_required: int = int(os.getenv("CALIBRATION_MINIMUM_SAMPLES_REQUIRED", "4"))
    auto_down_weight_unreliable_detector: bool = os.getenv("AUTO_DOWN_WEIGHT_UNRELIABLE_DETECTOR", "true").lower() == "true"
    frame_detector_max_weight: float = float(os.getenv("FRAME_DETECTOR_MAX_WEIGHT", "0.35"))
    video_detector_max_weight: float = float(os.getenv("VIDEO_DETECTOR_MAX_WEIGHT", "0.65"))
    unreliable_detector_weight: float = float(os.getenv("UNRELIABLE_DETECTOR_WEIGHT", "0.10"))
    reliability_report_path: str = os.getenv("RELIABILITY_REPORT_PATH", "calibration_reports/latest_evaluation.json")
    frame_bias_offset: float = float(os.getenv("FRAME_BIAS_OFFSET", os.getenv("FRAME_MODEL_BIAS_OFFSET", "0.15")))
    frame_temperature: float = float(os.getenv("FRAME_TEMPERATURE", "1.25"))
    use_temperature_calibration: bool = os.getenv("USE_TEMPERATURE_CALIBRATION", "true").lower() == "true"
    preserve_strong_raw_signal: bool = os.getenv("PRESERVE_STRONG_RAW_SIGNAL", "true").lower() == "true"
    strong_raw_frame_threshold: float = float(os.getenv("STRONG_RAW_FRAME_THRESHOLD", "0.85"))
    strong_calibrated_frame_threshold: float = float(os.getenv("STRONG_CALIBRATED_FRAME_THRESHOLD", "0.65"))
    require_agreement_for_likely_ai: bool = os.getenv("REQUIRE_AGREEMENT_FOR_LIKELY_AI", "true").lower() == "true"
    allow_single_detector_suspicious: bool = os.getenv("ALLOW_SINGLE_DETECTOR_SUSPICIOUS", "true").lower() == "true"
    frame_model_high_threshold: float = float(os.getenv("FRAME_MODEL_HIGH_THRESHOLD", "0.88"))
    frame_model_medium_threshold: float = float(os.getenv("FRAME_MODEL_MEDIUM_THRESHOLD", "0.70"))
    require_video_support_for_likely_ai: bool = os.getenv("REQUIRE_VIDEO_SUPPORT_FOR_LIKELY_AI", "true").lower() == "true"
    require_metadata_or_video_support_for_suspicious: bool = os.getenv("REQUIRE_METADATA_OR_VIDEO_SUPPORT_FOR_SUSPICIOUS", "true").lower() == "true"
    model_disagreement_threshold: float = float(os.getenv("MODEL_DISAGREEMENT_THRESHOLD", "0.30"))
    hf_trust_remote_code: bool = os.getenv("HF_TRUST_REMOTE_CODE", "false").lower() == "true"
    allow_model_download: bool = os.getenv("ALLOW_MODEL_DOWNLOAD", "false").lower() == "true"
    ai_debug_output: bool = os.getenv("AI_DEBUG_OUTPUT", "false").lower() == "true"
    hf_model_id: str | None = os.getenv("HF_MODEL_ID") or None
    hf_timeout_seconds: int = int(os.getenv("HF_TIMEOUT_SECONDS", "60"))
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models/cache")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @property
    def require_api_key(self) -> bool:
        return self.is_production or bool(self.ai_service_api_key)

    @property
    def allowed_video_roots(self) -> list[Path]:
        values = [value.strip() for value in self.ai_allowed_video_roots_csv.split(";")]
        return [Path(value).resolve() for value in values if value]

    @property
    def effective_model_version(self) -> str:
        return self.mock_model_version if self.ai_mode == "mock" else self.model_version

    @property
    def frame_calibration_enabled(self) -> bool:
        return self.calibration_enabled

    @property
    def frame_model_bias_offset(self) -> float:
        return self.frame_bias_offset

    @property
    def frame_model_ids(self) -> list[str]:
        values = [value.strip() for value in self.frame_model_ids_csv.split(",")]
        return [value for value in values if value]


settings = Settings()
