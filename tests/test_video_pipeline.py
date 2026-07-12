import math
from types import SimpleNamespace

from app.core.config import Settings
from app.models.requests import AnalyzeFrameItem
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult
from app.services.ensemble_ai_service import EnsembleAiService
from app.services.local_video_ai_service import _sample_or_pad
from app.services.score_mapping import map_label_scores


def frame_item(index: int) -> AnalyzeFrameItem:
    return AnalyzeFrameItem(
        frame_id=index + 1,
        frame_url=f"frames/1/frame_{index + 1:06d}.jpg",
        frame_index=index,
        timestamp_seconds=float(index),
        image_base64="present",
    )


def test_videomae_real_fake_label_mapping_uses_fake_class_probability() -> None:
    id2label = {0: "real", 1: "fake"}
    probabilities = [0.41, 0.59]
    scores = {id2label[index]: probability for index, probability in enumerate(probabilities)}

    result = map_label_scores(scores)

    assert result.ai_score == 0.59
    assert result.real_probability == 0.41
    assert result.mapped_fake_label == "fake"
    assert result.mapped_real_label == "real"


def test_uniform_sampling_returns_exact_model_frame_count() -> None:
    items = [(frame_item(index), SimpleNamespace(width=100, height=100)) for index in range(30)]

    sampled = _sample_or_pad(items, 16)

    assert len(sampled.items) == 16
    assert sampled.source_positions[0] == 0
    assert sampled.source_positions[-1] == 29
    assert sampled.truncated is True


def test_fewer_frames_are_padded_safely() -> None:
    items = [(frame_item(index), SimpleNamespace(width=100, height=100)) for index in range(3)]

    sampled = _sample_or_pad(items, 16)

    assert len(sampled.items) == 16
    assert sampled.source_positions[-1] == 2
    assert sampled.padded is True


def test_softmax_probabilities_sum_to_one() -> None:
    logits = [0.2, 1.1, -0.4]
    exps = [math.exp(value) for value in logits]
    probabilities = [value / sum(exps) for value in exps]

    assert round(sum(probabilities), 6) == 1.0


def test_ensemble_uses_separate_model_ids() -> None:
    settings = Settings()
    video = FakeService("video-model", "video_temporal", 0.40, 0.60)
    frame = FakeService("frame-model", "frame_image", 0.50, 0.70)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(0)]))

    assert result.model_capability == "ensemble_video_frame"
    assert result.model_id == "video-model+frame-model"
    assert any("Frame-level model helps" in warning for warning in result.warnings)


def test_frame_model_unavailable_does_not_claim_ensemble() -> None:
    settings = Settings()
    video = FakeService("video-model", "video_temporal", 0.40, 0.60)
    service = EnsembleAiService(settings, video, None)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(0)]))

    assert result.model_capability == "video_temporal"
    assert "Frame detector unavailable; only video model was used." in result.warnings


def test_strong_frame_evidence_prevents_combined_score_below_suspicious_floor() -> None:
    settings = Settings()
    video = FakeService("video-model", "video_temporal", 0.55, 0.74)
    frame = FakeService("frame-model", "frame_image", 0.96, 0.80, high_count=3, top_k=1.0)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(5)]))

    assert result.overall_ai_score >= 0.60
    assert result.minimum_recommended_score == 0.60
    assert result.strong_frame_evidence is True


def test_very_strong_frame_evidence_prevents_combined_score_below_high_floor() -> None:
    settings = Settings(frame_bias_offset=0.05)
    video = FakeService("video-model", "video_temporal", 0.65, 0.7376)
    frame = FakeService("frame-model", "frame_image", 1.0, 0.80, high_count=8, top_k=1.0)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(10)]))

    assert result.overall_ai_score >= 0.68
    assert result.minimum_recommended_score == 0.68
    assert result.model_disagreement is True
    assert any("Model components disagree" in warning for warning in result.warnings)


def test_real_like_frame_high_video_low_has_no_floor() -> None:
    settings = Settings()
    video = FakeService("video-model", "video_temporal", 0.4429, 0.60)
    frame = FakeService("frame-model", "frame_image", 0.8394, 0.80, high_count=29, top_k=0.8569)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(30)]))

    assert result.model_disagreement is True
    assert result.minimum_recommended_score is None
    assert result.ensemble_strategy == "frame_high_video_low_no_floor"
    assert "Frame detector gave high scores, but video temporal model did not confirm. Treat as inconclusive." in result.warnings


def test_frame_calibration_offset_is_applied_and_debug_keeps_raw_score() -> None:
    settings = Settings(ai_debug_output=True)
    video = FakeService("video-model", "video_temporal", 0.44, 0.60)
    frame = FakeService("frame-model", "frame_image", 0.84, 0.80, high_count=10, top_k=0.86)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(10)]))

    assert result.component_scores["frame"]["raw_frame_ai_score"] == 0.84
    assert result.component_scores["frame"]["calibrated_frame_ai_score"] == 0.6403
    assert result.debug["ensemble"]["raw_frame_ai_score"] == 0.84
    assert result.debug["ensemble"]["calibrated_frame_ai_score"] == 0.6403


def test_strong_raw_frame_signal_is_preserved_after_calibration() -> None:
    settings = Settings(ai_debug_output=True)
    video = FakeService("video-model", "video_temporal", 0.60, 0.70)
    frame = FakeService("frame-model", "frame_image", 0.85, 0.80, high_count=10, top_k=0.85)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(10)]))

    assert result.component_scores["frame"]["calibrated_frame_ai_score"] >= settings.strong_calibrated_frame_threshold
    assert result.debug["ensemble"]["raw_frame_ai_score"] == 0.85


def test_model_disagreement_uses_configured_threshold() -> None:
    settings = Settings()
    video = FakeService("video-model", "video_temporal", 0.44, 0.60)
    frame = FakeService("frame-model", "frame_image", 0.74, 0.80, high_count=5, top_k=0.74)
    service = EnsembleAiService(settings, video, frame)

    result = service.analyze_frames(SimpleNamespace(video_id=1, job_id=2, frames=[frame_item(index) for index in range(10)]))

    assert result.model_disagreement is True


class FakeService:
    model_loaded = True

    def __init__(
        self,
        model_id: str,
        model_capability: str,
        score: float,
        confidence: float,
        high_count: int = 0,
        top_k: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_capability = model_capability
        self.score = score
        self.confidence = confidence
        self.high_count = high_count
        self.top_k = top_k or score

    def analyze_frames(self, request) -> AnalyzeFramesResponse:
        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id=self.model_id,
            model_version="test",
            model_capability=self.model_capability,
            is_mock=False,
            overall_ai_score=self.score,
            real_probability=1.0 - self.score,
            overall_confidence=self.confidence,
            label_hint="Inconclusive",
            frames=[
                FrameAnalysisResult(
                    frame_id=frame.frame_id,
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    ai_score=self.score,
                    real_probability=1.0 - self.score,
                    confidence=self.confidence,
                    notes=[],
                )
                for frame in request.frames
            ],
            notes=[],
            warnings=[],
            debug={
                "top_k_mean": self.top_k,
                "p90_score": self.top_k,
                "high_frame_count": self.high_count,
            },
        )
