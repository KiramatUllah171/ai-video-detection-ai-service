import logging
from dataclasses import dataclass

from app.core.config import Settings
from app.models.requests import AnalyzeFrameItem, AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult
from app.services.base_ai_service import BaseAiService
from app.services.image_preprocessing import decode_base64_image
from app.services.score_mapping import label_hint, map_label_scores, mapping_debug
from app.utils.errors import AiServiceError

logger = logging.getLogger(__name__)


class LocalVideoAiService(BaseAiService):
    model_capability = "video_temporal"

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
        self._model_id = model_id or settings.model_id
        self.model_loaded = model is not None and processor is not None

    @property
    def model_id(self) -> str:
        return self._model_id

    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        if not self.model_loaded:
            raise AiServiceError(
                "REAL_MODEL_NOT_CONFIGURED",
                "Real AI model is not configured. Configure MODEL_ID/MODEL_PATH or enable mock mode for development.",
                status_code=503,
            )

        frames = request.frames[: self.settings.max_frames_per_request]
        logger.info(
            "Running video-level AI analysis for video_id=%s job_id=%s frame_count=%s frame_ids=%s has_images=%s",
            request.video_id,
            request.job_id,
            len(frames),
            [frame.frame_id for frame in frames[:5]],
            [bool(frame.image_base64) for frame in frames[:5]],
        )
        frame_images = [(frame, decode_base64_image(frame.image_base64 or "")) for frame in frames]

        try:
            import torch
        except ImportError as exception:
            raise AiServiceError("MODEL_LOAD_FAILED", "PyTorch is required for local real AI inference.", status_code=503) from exception

        try:
            sampled = _sample_or_pad(frame_images, self.settings.model_frame_count)
            sampled_images = [item.image for item in sampled.items]
            inputs = self.processor(sampled_images, return_tensors="pt")
            input_shapes_before = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
            if "pixel_values" in inputs and getattr(inputs["pixel_values"], "ndim", 0) == 4:
                inputs["pixel_values"] = inputs["pixel_values"].unsqueeze(0)
            input_shapes_after = {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0].detach().cpu().tolist()
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].detach().cpu().tolist()
            labels = getattr(self.model.config, "id2label", {})
            label2id = getattr(self.model.config, "label2id", {})
            scores = {str(labels.get(index, f"LABEL_{index}")): probability for index, probability in enumerate(probabilities)}
            mapped = map_label_scores(scores)
            predicted_class_index = int(max(range(len(probabilities)), key=lambda index: probabilities[index]))
            predicted_class_label = str(labels.get(predicted_class_index, f"LABEL_{predicted_class_index}"))
        except AiServiceError:
            raise
        except Exception as exception:
            raise AiServiceError("AI_INFERENCE_FAILED", "AI inference failed for the configured video model.", status_code=502) from exception

        frame_results = [
            FrameAnalysisResult(
                frame_id=frame.frame_id,
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
                ai_score=round(mapped.ai_score, 4),
                real_probability=round(mapped.real_probability, 4),
                confidence=round(mapped.confidence, 4),
                notes=["Video-level model output distributed to frame items for reporting."],
            )
            for frame in frames
        ]
        overall_ai_score = round(mapped.ai_score, 4)
        debug = {
            "model": {
                "model_id": self.model_id,
                "model_class": type(self.model).__name__,
                "model_capability": self.model_capability,
                "model_loaded": self.model_loaded,
                "device": self.device,
                "id2label": {str(key): value for key, value in getattr(self.model.config, "id2label", {}).items()},
                "label2id": {str(key): value for key, value in getattr(self.model.config, "label2id", {}).items()},
                "selected_fake_label": mapped.mapped_fake_label,
                "selected_real_label": mapped.mapped_real_label,
                "label_mapping_confidence": mapped.mapping_confidence,
            },
            "prediction": {
                "raw_logits": logits,
                "softmax_probabilities": probabilities,
                "predicted_class_index": predicted_class_index,
                "predicted_class_label": predicted_class_label,
                "mapped_fake_probability": mapped.ai_score,
                "mapped_real_probability": mapped.real_probability,
            },
            "preprocessing": {
                "frames_received": len(frames),
                "frames_used_by_model": len(sampled.items),
                "input_tensor_shape_before_fix": input_shapes_before,
                "input_tensor_shape": input_shapes_after,
                "frame_image_sizes_before_preprocessing": [[image.width, image.height] for _, image in frame_images],
                "sampled_frame_ids": [item.frame.frame_id for item in sampled.items],
                "sampled_frame_indexes": [item.frame.frame_index for item in sampled.items],
                "sampled_timestamps": [item.frame.timestamp_seconds for item in sampled.items],
                "sampled_source_positions": sampled.source_positions,
                "padded": sampled.padded,
                "truncated": sampled.truncated,
            },
            "scoring": {
                "video_model_ai_score": overall_ai_score,
                "frame_mean_score": overall_ai_score,
                "frame_top_k_mean": overall_ai_score,
                "frame_p90_score": overall_ai_score,
                "frame_max_score": overall_ai_score,
                "high_frame_count": len(frames) if overall_ai_score >= 0.75 else 0,
                "final_visual_score_before_backend": overall_ai_score,
            },
            **mapping_debug(mapped),
        }

        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id=self.model_id,
            model_version=self.settings.model_version,
            model_capability=self.model_capability,
            is_mock=False,
            overall_ai_score=overall_ai_score,
            real_probability=round(1.0 - overall_ai_score, 4),
            overall_confidence=round(mapped.confidence, 4),
            label_hint=label_hint(overall_ai_score),
            frames=frame_results,
            notes=["This is a probability-based analysis."],
            warnings=[],
            debug=debug if self.settings.ai_debug_output else None,
        )


@dataclass(frozen=True)
class SampledVideoFrames:
    items: list
    source_positions: list[int]
    padded: bool
    truncated: bool


@dataclass(frozen=True)
class FrameImage:
    frame: AnalyzeFrameItem
    image: object


def _sample_or_pad(frame_images: list[tuple[AnalyzeFrameItem, object]], target_count: int) -> SampledVideoFrames:
    if not frame_images:
        raise AiServiceError("FRAME_IMAGE_REQUIRED", "At least one frame image is required for real local AI analysis.")

    target_count = max(1, target_count)
    wrapped = [FrameImage(frame, image) for frame, image in frame_images]
    if len(wrapped) >= target_count:
        if len(wrapped) == target_count:
            return SampledVideoFrames(wrapped, list(range(len(wrapped))), padded=False, truncated=False)
        step = (len(wrapped) - 1) / (target_count - 1)
        positions = [round(index * step) for index in range(target_count)]
        return SampledVideoFrames([wrapped[position] for position in positions], positions, padded=False, truncated=True)

    positions = list(range(len(wrapped))) + [len(wrapped) - 1] * (target_count - len(wrapped))
    return SampledVideoFrames(wrapped + [wrapped[-1]] * (target_count - len(wrapped)), positions, padded=True, truncated=False)
