import logging

from app.core.config import Settings, settings
from app.services.ensemble_ai_service import EnsembleAiService
from app.services.base_ai_service import BaseAiService
from app.services.hosted_ai_service import HostedAiService
from app.services.local_frame_ai_service import LocalFrameAiService
from app.services.local_video_ai_service import LocalVideoAiService
from app.services.model_registry import build_model_registry
from app.services.mock_ai_service import MockAiService
from app.utils.errors import AiServiceError

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, current_settings: Settings = settings) -> None:
        self.settings = current_settings
        self._service: BaseAiService | None = None
        self.load_error: AiServiceError | None = None
        self.device = "cpu"

    def get_service(self) -> BaseAiService:
        if self._service is not None:
            return self._service

        if self.settings.ai_mode == "mock":
            self._service = MockAiService()
            return self._service

        if self.settings.provider_mode == "hosted":
            self._service = HostedAiService()
            return self._service

        self._service = self._load_local_service()
        return self._service

    def health(self) -> dict:
        service = self._service
        if service is None and self.settings.ai_mode == "mock":
            service = self.get_service()

        capability = service.model_capability if service else self._expected_capability()
        loaded = bool(service and service.model_loaded)
        is_mock = bool(service and service.model_capability == "mock") or self.settings.ai_mode == "mock"
        fallback_used = bool(service and service.model_capability == "mock" and self.settings.ai_mode != "mock")
        video_model_id = getattr(getattr(service, "video_service", None), "model_id", None)
        frame_model_id = getattr(getattr(service, "frame_service", None), "model_id", None)
        frame_service = getattr(service, "frame_service", None)
        load_warnings = getattr(service, "load_warnings", []) or []
        reliability = getattr(service, "reliability", {}) or {}
        return {
            "ai_mode": self.settings.ai_mode,
            "provider_mode": self.settings.provider_mode,
            "model_id": self._model_id_for_health(capability),
            "model_version": self.settings.effective_model_version,
            "model_loaded": loaded,
            "model_capability": capability,
            "device": self.device,
            "allow_model_download": self.settings.allow_model_download,
            "is_mock": is_mock,
            "fallback_used": fallback_used,
            "trust_remote_code": self.settings.hf_trust_remote_code,
            "supported_input": "image_base64 frames" if capability != "hosted" else "hosted provider request",
            "label_mapping_status": "mock" if is_mock else "requires model id2label inspection",
            "last_model_load_error": self.load_error.message if self.load_error else None,
            "video_model_id": video_model_id,
            "frame_model_id": frame_model_id,
            "frame_model_loaded": bool(frame_service and frame_service.model_loaded),
            "frame_model_load_error": next((warning for warning in load_warnings if "Frame detector model" in warning), None),
            "ensemble_real_components_count": int(bool(video_model_id)) + int(bool(frame_model_id)),
            "model_registry": build_model_registry(self.settings, reliability),
            "detectors_loaded": [model_id for model_id in [video_model_id, frame_model_id] if model_id],
            "detectors_failed": load_warnings,
        }

    def _load_local_service(self) -> BaseAiService:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoModelForVideoClassification
        except ImportError as exception:
            return self._real_model_failure("MODEL_LOAD_FAILED", "PyTorch and Transformers are required for local real AI inference.", exception)

        self.device = self._select_device(torch)
        trust_remote_code = self.settings.hf_trust_remote_code
        local_files_only = not self.settings.allow_model_download

        if self.settings.ensemble_enabled:
            warnings: list[str] = []
            video_service = self._try_load_video_service(
                self.settings.primary_video_model_id,
                AutoImageProcessor,
                AutoModelForVideoClassification,
                trust_remote_code,
                local_files_only,
                warnings,
            )
            frame_service = None
            for frame_model_id in self.settings.frame_model_ids:
                frame_service = self._try_load_frame_service(
                    frame_model_id,
                    AutoImageProcessor,
                    AutoModelForImageClassification,
                    trust_remote_code,
                    local_files_only,
                    warnings,
                )
                if frame_service is not None:
                    break
            if video_service or frame_service:
                return EnsembleAiService(self.settings, video_service, frame_service, warnings)

        for model_id in [self.settings.model_id, self.settings.fallback_video_model_id]:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    model_id,
                    cache_dir=self.settings.model_cache_dir,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                )
                model = AutoModelForVideoClassification.from_pretrained(
                    model_id,
                    cache_dir=self.settings.model_cache_dir,
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only,
                ).to(self.device)
                model.eval()
                return LocalVideoAiService(self.settings, model=model, processor=processor, device=self.device, model_id=model_id)
            except Exception as exception:
                logger.warning("Video model %s could not be loaded: %s", model_id, exception)

        try:
            model_id = self.settings.fallback_image_model_id
            processor = AutoImageProcessor.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model = AutoModelForImageClassification.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ).to(self.device)
            model.eval()
            return LocalFrameAiService(self.settings, model=model, processor=processor, device=self.device, model_id=model_id)
        except Exception as exception:
            logger.warning("Frame model %s could not be loaded: %s", self.settings.fallback_image_model_id, exception)

        return self._real_model_failure(
            "REAL_MODEL_NOT_CONFIGURED",
            "Real AI model is not configured. Configure MODEL_ID/MODEL_PATH or enable mock mode for development.",
        )

    def force_load(self) -> dict:
        self._service = None
        self.load_error = None
        service = self.get_service()
        health = self.health()
        health["model_class"] = type(getattr(service, "model", service)).__name__
        model = getattr(service, "model", None)
        config = getattr(model, "config", None)
        health["id2label"] = {str(key): value for key, value in getattr(config, "id2label", {}).items()} if config else {}
        return health

    def _try_load_video_service(
        self,
        model_id: str,
        processor_cls,
        model_cls,
        trust_remote_code: bool,
        local_files_only: bool,
        warnings: list[str],
    ) -> LocalVideoAiService | None:
        try:
            processor = processor_cls.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model = model_cls.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ).to(self.device)
            model.eval()
            service = LocalVideoAiService(self.settings, model=model, processor=processor, device=self.device, model_id=model_id)
            service.model_capability = "video_temporal"
            return service
        except Exception as exception:
            message = f"Video detector model {model_id} could not be loaded."
            warnings.append(message)
            logger.warning("%s %s", message, exception)
            return None

    def _try_load_frame_service(
        self,
        model_id: str,
        processor_cls,
        model_cls,
        trust_remote_code: bool,
        local_files_only: bool,
        warnings: list[str],
    ) -> LocalFrameAiService | None:
        try:
            processor = processor_cls.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            model = model_cls.from_pretrained(
                model_id,
                cache_dir=self.settings.model_cache_dir,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            ).to(self.device)
            model.eval()
            return LocalFrameAiService(self.settings, model=model, processor=processor, device=self.device, model_id=model_id)
        except Exception as exception:
            message = f"Frame detector model {model_id} could not be loaded."
            warnings.append(message)
            logger.warning("%s %s", message, exception)
            return None

    def _real_model_failure(self, error_code: str, message: str, exception: Exception | None = None) -> BaseAiService:
        self.load_error = AiServiceError(error_code, message, status_code=503)
        if self.settings.enable_mock_fallback:
            logger.warning("Using mock fallback because real model loading failed: %s", message)
            return MockAiService()
        if exception:
            logger.warning("Real model loading failed: %s", exception)
        return LocalVideoAiService(self.settings, model=None, processor=None, device=self.device)

    def _select_device(self, torch) -> str:
        if self.settings.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.settings.device == "auto" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _expected_capability(self) -> str:
        if self.settings.ai_mode == "mock":
            return "mock"
        return "video_temporal" if self.settings.provider_mode == "local" else "hosted"

    def _model_id_for_health(self, capability: str) -> str:
        if capability == "mock":
            return "mock-deterministic-frame-hash"
        if capability == "frame_image":
            return self.settings.fallback_image_model_id
        if capability == "ensemble_video_frame":
            return f"{self.settings.primary_video_model_id}+{self.settings.frame_model_id}"
        return self.settings.hf_model_id or self.settings.model_id


model_loader = ModelLoader()
