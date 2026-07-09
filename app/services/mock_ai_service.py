import hashlib
from statistics import fmean

from app.core.config import settings
from app.models.requests import AnalyzeFrameItem, AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult


class MockAiService:
    frame_note = "Mock score generated from deterministic frame identifier hash."
    response_notes = [
        "This is a mock AI response for pipeline integration only.",
        "Do not treat this result as real AI detection.",
    ]

    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        frame_results = [self._analyze_frame(frame) for frame in request.frames]
        overall_ai_score = round(fmean(frame.ai_score for frame in frame_results), 2)
        overall_confidence = round(fmean(frame.confidence for frame in frame_results), 2)

        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_version=settings.model_version,
            overall_ai_score=overall_ai_score,
            overall_confidence=overall_confidence,
            label_hint=self._label_hint(overall_ai_score),
            frames=frame_results,
            notes=self.response_notes,
        )

    def _analyze_frame(self, frame: AnalyzeFrameItem) -> FrameAnalysisResult:
        seed = f"{frame.frame_url}|{frame.frame_index}"
        ai_score = self._normalized_hash(seed, "score")
        confidence = 0.5 + (self._normalized_hash(seed, "confidence") * 0.45)

        return FrameAnalysisResult(
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            ai_score=round(ai_score, 2),
            confidence=round(confidence, 2),
            notes=[self.frame_note],
        )

    @staticmethod
    def _normalized_hash(value: str, salt: str) -> float:
        digest = hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()
        number = int(digest[:12], 16)
        return number / float(0xFFFFFFFFFFFF)

    @staticmethod
    def _label_hint(overall_ai_score: float) -> str:
        if overall_ai_score >= 0.75:
            return "Likely AI-Generated"
        if overall_ai_score >= 0.55:
            return "Suspicious"
        if overall_ai_score >= 0.35:
            return "Inconclusive"
        return "Likely Real"
