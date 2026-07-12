import logging
from statistics import fmean

from app.core.config import Settings
from app.models.requests import AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult
from app.services.base_ai_service import BaseAiService
from app.services.image_preprocessing import decode_base64_image
from app.services.score_mapping import aggregate_frame_scores, aggregation_debug, label_hint, map_label_scores, mapping_debug
from app.utils.errors import AiServiceError

logger = logging.getLogger(__name__)


class LocalFrameAiService(BaseAiService):
    model_capability = "frame_image"

    def __init__(
        self,
        settings: Settings,
        model: object | None = None,
        processor: object | None = None,
        device: str = "cpu",
        model_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.processor = processor
        self.device = device
        self.model_id = model_id or settings.fallback_image_model_id
        self.model_loaded = model is not None and processor is not None

    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        if not self.model_loaded:
            raise AiServiceError(
                "REAL_MODEL_NOT_CONFIGURED",
                "Real AI model is not configured. Configure MODEL_ID/MODEL_PATH or enable mock mode for development.",
                status_code=503,
            )

        frames = request.frames[: self.settings.max_frames_per_request]
        logger.info(
            "Running frame-level AI analysis for video_id=%s job_id=%s frame_count=%s frame_ids=%s has_images=%s",
            request.video_id,
            request.job_id,
            len(frames),
            [frame.frame_id for frame in frames[:5]],
            [bool(frame.image_base64) for frame in frames[:5]],
        )
        frame_results = [self._analyze_frame(frame) for frame in frames]
        aggregation = aggregate_frame_scores([frame.ai_score for frame in frame_results])
        overall_ai_score = round(aggregation.visual_score, 4)
        overall_confidence = round(fmean(frame.confidence for frame in frame_results), 4)
        debug = aggregation_debug(aggregation)

        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id=self.model_id,
            model_version=self.settings.model_version,
            model_capability=self.model_capability,
            is_mock=False,
            overall_ai_score=overall_ai_score,
            real_probability=round(1.0 - overall_ai_score, 4),
            overall_confidence=overall_confidence,
            label_hint=label_hint(overall_ai_score),
            frames=frame_results,
            notes=["This is a probability-based analysis."],
            warnings=["Frame-level model used. Temporal video consistency is not evaluated."],
            debug=debug if self.settings.ai_debug_output else None,
        )

    def _analyze_frame(self, frame) -> FrameAnalysisResult:
        image = decode_base64_image(frame.image_base64 or "")
        try:
            import torch
        except ImportError as exception:
            raise AiServiceError("MODEL_LOAD_FAILED", "PyTorch is required for local real AI inference.", status_code=503) from exception

        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()
            labels = getattr(self.model.config, "id2label", {})
            scores = {str(labels.get(index, f"LABEL_{index}")): probability for index, probability in enumerate(probabilities)}
            mapped = map_label_scores(scores)
        except AiServiceError:
            raise
        except Exception as exception:
            raise AiServiceError("AI_INFERENCE_FAILED", "AI inference failed for one or more frames.", status_code=502) from exception

        return FrameAnalysisResult(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            ai_score=round(mapped.ai_score, 4),
            real_probability=round(mapped.real_probability, 4),
            confidence=round(mapped.confidence, 4),
            notes=["Frame analyzed by configured AI model."],
            debug=mapping_debug(mapped) if self.settings.ai_debug_output else None,
        )
