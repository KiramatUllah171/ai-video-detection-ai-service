import json
import math
from pathlib import Path
from statistics import fmean

from app.core.config import Settings
from app.models.requests import AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse, FrameAnalysisResult
from app.services.base_ai_service import BaseAiService
from app.services.score_mapping import label_hint


class EnsembleAiService(BaseAiService):
    model_capability = "ensemble_video_frame"

    def __init__(
        self,
        settings: Settings,
        video_service: BaseAiService | None,
        frame_service: BaseAiService | None,
        load_warnings: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self.video_service = video_service
        self.frame_service = frame_service
        self.load_warnings = load_warnings or []
        self.reliability = _load_reliability_report(settings)
        self.model_loaded = bool(video_service and video_service.model_loaded) or bool(frame_service and frame_service.model_loaded)

    def analyze_frames(self, request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
        responses: list[tuple[str, float, AnalyzeFramesResponse]] = []
        warnings = list(self.load_warnings)

        video_weight = _detector_weight(
            "video",
            self.settings.ensemble_video_weight,
            self.settings.video_detector_max_weight,
            self.reliability,
            self.settings,
            warnings,
        )
        frame_weight = _detector_weight(
            "frame_calibrated",
            self.settings.ensemble_frame_weight,
            self.settings.frame_detector_max_weight,
            self.reliability,
            self.settings,
            warnings,
        )

        if self.video_service and self.video_service.model_loaded:
            responses.append(("video", video_weight, self.video_service.analyze_frames(request)))
        else:
            warnings.append("Primary video detector model was not available.")

        if self.frame_service and self.frame_service.model_loaded:
            responses.append(("frame", frame_weight, self.frame_service.analyze_frames(request)))
            warnings.append("Frame-level model helps detect visual artifacts but does not replace temporal analysis.")
        else:
            warnings.append("Frame-level detector model was not available.")

        if not responses:
            from app.utils.errors import AiServiceError

            raise AiServiceError(
                "REAL_MODEL_NOT_CONFIGURED",
                "No detector model is available for ensemble analysis.",
                status_code=503,
            )

        if len(responses) == 1:
            warnings.append("Only one detector model was available.")
            only_response = responses[0][2]
            if responses[0][0] == "video":
                warnings.append("Frame detector unavailable; only video model was used.")
            return only_response.model_copy(update={
                "warnings": [*only_response.warnings, *warnings],
                "model_capability": only_response.model_capability,
                "debug": _debug_with_availability(only_response.debug, self, responses, warnings) if self.settings.ai_debug_output else only_response.debug,
            })

        total_weight = sum(weight for _, weight, _ in responses)
        if total_weight <= 0:
            total_weight = len(responses)
            responses = [(name, 1.0, response) for name, _, response in responses]

        video_response = next((response for name, _, response in responses if name == "video"), None)
        frame_response = next((response for name, _, response in responses if name == "frame"), None)
        calibrated_frame_response = _calibrate_frame_response(frame_response, self.settings)
        score_responses = [
            (name, weight, calibrated_frame_response if name == "frame" and calibrated_frame_response is not None else response)
            for name, weight, response in responses
        ]
        weighted_average = sum(response.overall_ai_score * weight for _, weight, response in score_responses) / total_weight
        scoring_decision = _score_conflict_aware(weighted_average, video_response, calibrated_frame_response, frame_response, warnings, self.settings)
        overall_ai_score = round(scoring_decision.score, 4)
        overall_confidence = round(
            min(fmean(response.overall_confidence for _, _, response in responses), 0.92 if len(responses) > 1 else 0.75),
            4,
        )
        if scoring_decision.model_disagreement:
            overall_confidence = round(max(0.0, overall_confidence - 0.06), 4)

        frames = _merge_frame_results([response for _, _, response in responses])
        component_scores = {
            "video": _component_score(video_response, reliability=_detector_reliability("video", self.reliability)),
            "frame": _component_score(
                calibrated_frame_response,
                raw_response=frame_response,
                reliability=_detector_reliability("frame", self.reliability),
            ),
            "combined": {
                "weighted_average": round(weighted_average, 4),
                "adjusted_score": overall_ai_score,
                "minimum_recommended_score": scoring_decision.minimum_recommended_score,
                "strategy": scoring_decision.strategy,
                "detector_reliability_status": self.reliability.get("status"),
            },
        }
        debug = {
            "ensemble": {
                "enabled": True,
                "video_model_id": video_response.model_id if video_response else None,
                "frame_model_id": frame_response.model_id if frame_response else None,
                "frame_model_loaded": frame_response is not None,
                "frame_model_load_error": None,
                "ensemble_real_components_count": len(responses),
                "components": [
                    {
                        "name": name,
                        "weight": weight,
                        "reliability": _detector_reliability(name, self.reliability),
                        "model_id": response.model_id,
                        "model_capability": response.model_capability,
                        "overall_ai_score": response.overall_ai_score,
                        "overall_confidence": response.overall_confidence,
                    }
                    for name, weight, response in responses
                ],
                "weighted_average_ai_score": round(weighted_average, 4),
                "combined_ai_score": overall_ai_score,
                "combined_confidence": overall_confidence,
                "model_disagreement": scoring_decision.model_disagreement,
                "strong_frame_evidence": scoring_decision.strong_frame_evidence,
                "minimum_recommended_score": scoring_decision.minimum_recommended_score,
                "ensemble_strategy": scoring_decision.strategy,
                "raw_frame_ai_score": frame_response.overall_ai_score if frame_response else None,
                "calibrated_frame_ai_score": calibrated_frame_response.overall_ai_score if calibrated_frame_response else None,
                "detector_reliability": self.reliability,
            },
            "component_debug": {
                name: response.debug for name, _, response in responses if response.debug is not None
            },
        }

        return AnalyzeFramesResponse(
            video_id=request.video_id,
            job_id=request.job_id,
            model_id="+".join(response.model_id for _, _, response in responses),
            model_version=self.settings.model_version,
            model_capability=self.model_capability,
            is_mock=False,
            overall_ai_score=overall_ai_score,
            real_probability=round(1.0 - overall_ai_score, 4),
            overall_confidence=overall_confidence,
            label_hint=label_hint(overall_ai_score),
            frames=frames,
            notes=["This is a probability-based ensemble analysis."],
            warnings=warnings,
            debug=debug if self.settings.ai_debug_output else None,
            model_disagreement=scoring_decision.model_disagreement,
            strong_frame_evidence=scoring_decision.strong_frame_evidence,
            minimum_recommended_score=scoring_decision.minimum_recommended_score,
            ensemble_strategy=scoring_decision.strategy,
            component_scores=component_scores,
        )


def _merge_frame_results(responses: list[AnalyzeFramesResponse]) -> list[FrameAnalysisResult]:
    if not responses:
        return []

    by_frame_id: dict[int, list[FrameAnalysisResult]] = {}
    for response in responses:
        for frame in response.frames:
            by_frame_id.setdefault(frame.frame_id, []).append(frame)

    merged: list[FrameAnalysisResult] = []
    for frame_id, items in by_frame_id.items():
        first = items[0]
        ai_score = round(fmean(item.ai_score for item in items), 4)
        confidence = round(fmean(item.confidence for item in items), 4)
        merged.append(
            FrameAnalysisResult(
                frame_id=frame_id,
                frame_index=first.frame_index,
                timestamp_seconds=first.timestamp_seconds,
                ai_score=ai_score,
                real_probability=round(1.0 - ai_score, 4),
                confidence=confidence,
                notes=["Frame score combined from available detector models."],
            )
        )

    return sorted(merged, key=lambda frame: frame.frame_index)


class EnsembleScoreDecision:
    def __init__(
        self,
        score: float,
        strategy: str,
        model_disagreement: bool = False,
        strong_frame_evidence: bool = False,
        minimum_recommended_score: float | None = None,
    ) -> None:
        self.score = score
        self.strategy = strategy
        self.model_disagreement = model_disagreement
        self.strong_frame_evidence = strong_frame_evidence
        self.minimum_recommended_score = minimum_recommended_score


def _score_conflict_aware(
    weighted_average: float,
    video_response: AnalyzeFramesResponse | None,
    frame_response: AnalyzeFramesResponse | None,
    raw_frame_response: AnalyzeFramesResponse | None,
    warnings: list[str],
    settings: Settings,
) -> EnsembleScoreDecision:
    if video_response is None or frame_response is None:
        return EnsembleScoreDecision(weighted_average, "single_available_detector")

    video_score = video_response.overall_ai_score
    frame_score = frame_response.overall_ai_score
    raw_frame_score = raw_frame_response.overall_ai_score if raw_frame_response is not None else frame_score
    frame_debug = frame_response.debug or {}
    frame_top_k_mean = float(frame_debug.get("top_k_mean", frame_score))
    frame_p90_score = float(frame_debug.get("p90_score", frame_score))
    frame_high_count = int(frame_debug.get("high_frame_count", sum(1 for frame in frame_response.frames if frame.ai_score >= 0.75)))
    model_disagreement = abs(video_score - raw_frame_score) >= settings.model_disagreement_threshold
    strong_frame_evidence = frame_score >= 0.75 and frame_top_k_mean >= 0.80 and frame_high_count >= 3
    very_strong_frame_evidence = frame_score >= 0.85 and frame_top_k_mean >= 0.85 and frame_high_count >= 5
    minimum_score: float | None = None
    strategy = "weighted_average"

    if raw_frame_score >= settings.frame_model_medium_threshold and video_score < 0.50:
        warnings.append("Frame detector gave high scores, but video temporal model did not confirm. Treat as inconclusive.")
        strategy = "frame_high_video_low_no_floor"
    elif frame_score >= 0.75 and video_score >= 0.50:
        minimum_score = 0.60
        strategy = "strong_frame_evidence_floor"
        warnings.append("Frame-level detector found strong AI-like visual signals, while video temporal model disagreed.")

    if very_strong_frame_evidence and video_score >= 0.60:
        minimum_score = 0.68
        strategy = "very_strong_frame_evidence_floor"

    if model_disagreement:
        warnings.append("Model components disagree; treat result with caution.")

    score = max(weighted_average, minimum_score or 0.0)
    return EnsembleScoreDecision(
        score,
        strategy,
        model_disagreement=model_disagreement,
        strong_frame_evidence=strong_frame_evidence,
        minimum_recommended_score=minimum_score,
    )


def _component_score(
    response: AnalyzeFramesResponse | None,
    raw_response: AnalyzeFramesResponse | None = None,
    reliability: dict | None = None,
) -> dict | None:
    if response is None:
        return None
    payload = {
        "model_id": response.model_id,
        "model_capability": response.model_capability,
        "ai_score": response.overall_ai_score,
        "real_probability": response.real_probability,
        "confidence": response.overall_confidence,
    }
    if raw_response is not None:
        payload["raw_frame_ai_score"] = raw_response.overall_ai_score
        payload["calibrated_frame_ai_score"] = response.overall_ai_score
    if reliability is not None:
        payload["reliability"] = reliability
    return payload


def _load_reliability_report(settings: Settings) -> dict:
    if not settings.calibration_enabled:
        return {"status": "disabled"}

    path = Path(settings.reliability_report_path)
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "minimum_samples_required": settings.calibration_minimum_samples_required,
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid", "path": str(path)}

    summary = payload.get("summary") or {}
    detector_reliability = summary.get("detector_reliability") or {}
    sample_count = int(summary.get("sample_count") or 0)
    status = "active" if sample_count >= settings.calibration_minimum_samples_required else "limited_samples"
    return {
        "status": status,
        "path": str(path),
        "sample_count": sample_count,
        "minimum_samples_required": settings.calibration_minimum_samples_required,
        **detector_reliability,
    }


def _detector_weight(
    detector_key: str,
    configured_weight: float,
    max_weight: float,
    reliability: dict,
    settings: Settings,
    warnings: list[str],
) -> float:
    weight = min(max(configured_weight, 0.0), max_weight)
    if not settings.auto_down_weight_unreliable_detector or reliability.get("status") != "active":
        if reliability.get("status") == "limited_samples":
            warnings.append("Calibration report has too few samples; detector weights remain conservative defaults.")
        return weight

    detector = reliability.get(detector_key) or {}
    accuracy = detector.get("accuracy")
    average_real = detector.get("average_real_score")
    average_ai = detector.get("average_ai_score")
    unreliable = accuracy is not None and accuracy < 0.50
    worse_than_random = average_real is not None and average_ai is not None and average_ai <= average_real

    if unreliable or worse_than_random:
        warnings.append(f"{detector_key} detector was down-weighted by calibration reliability checks.")
        return min(weight, settings.unreliable_detector_weight)

    return weight


def _detector_reliability(name: str, reliability: dict) -> dict | None:
    if not reliability:
        return None
    key = "video" if name == "video" else "frame_calibrated"
    value = reliability.get(key)
    return value if isinstance(value, dict) else None


def _calibrate_frame_response(response: AnalyzeFramesResponse | None, settings: Settings) -> AnalyzeFramesResponse | None:
    if response is None or response.model_capability != "frame_image" or not settings.frame_calibration_enabled:
        return response

    calibrated_score = _calibrate_frame_score(response.overall_ai_score, settings)
    calibrated_frames = [
        frame.model_copy(update={
            "ai_score": _calibrate_frame_score(frame.ai_score, settings),
            "real_probability": round(1.0 - _calibrate_frame_score(frame.ai_score, settings), 4),
        })
        for frame in response.frames
    ]
    debug = dict(response.debug or {})
    debug["raw_frame_ai_score"] = response.overall_ai_score
    debug["calibrated_frame_ai_score"] = round(calibrated_score, 4)
    debug["calibration"] = {
        "enabled": settings.frame_calibration_enabled,
        "frame_bias_offset": settings.frame_bias_offset,
        "frame_temperature": settings.frame_temperature,
        "use_temperature_calibration": settings.use_temperature_calibration,
        "preserve_strong_raw_signal": settings.preserve_strong_raw_signal,
        "strong_raw_frame_threshold": settings.strong_raw_frame_threshold,
        "strong_calibrated_frame_threshold": settings.strong_calibrated_frame_threshold,
    }
    if "top_k_mean" in debug:
        debug["raw_frame_top_k_mean"] = debug["top_k_mean"]
        debug["top_k_mean"] = _calibrate_frame_score(float(debug["top_k_mean"]), settings)
    if "p90_score" in debug:
        debug["raw_frame_p90_score"] = debug["p90_score"]
        debug["p90_score"] = _calibrate_frame_score(float(debug["p90_score"]), settings)
    if "max_score" in debug:
        debug["raw_frame_max_score"] = debug["max_score"]
        debug["max_score"] = _calibrate_frame_score(float(debug["max_score"]), settings)

    return response.model_copy(update={
        "overall_ai_score": round(calibrated_score, 4),
        "real_probability": round(1.0 - calibrated_score, 4),
        "frames": calibrated_frames,
        "debug": debug,
    })


def _calibrate_frame_score(raw_score: float, settings: Settings) -> float:
    score = max(0.0, min(raw_score, 1.0))
    if settings.use_temperature_calibration:
        score = _temperature_scale_probability(score, settings.frame_temperature)

    score = max(0.0, score - settings.frame_bias_offset)
    if settings.preserve_strong_raw_signal and raw_score >= settings.strong_raw_frame_threshold:
        score = max(score, settings.strong_calibrated_frame_threshold)

    return round(max(0.0, min(score, 1.0)), 4)


def _temperature_scale_probability(score: float, temperature: float) -> float:
    if score <= 0.0 or score >= 1.0 or temperature <= 0:
        return score

    logit = math.log(score / (1.0 - score))
    return 1.0 / (1.0 + math.exp(-(logit / temperature)))


def _debug_with_availability(debug: dict | None, service: EnsembleAiService, responses, warnings: list[str]) -> dict:
    payload = debug.copy() if debug else {}
    payload["ensemble"] = {
        "enabled": True,
        "video_model_id": responses[0][2].model_id if responses and responses[0][0] == "video" else None,
        "frame_model_id": None,
        "frame_model_loaded": False,
        "frame_model_load_error": next((warning for warning in warnings if "Frame detector model" in warning), None),
        "ensemble_real_components_count": len(responses),
    }
    return payload
