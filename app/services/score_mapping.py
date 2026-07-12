from dataclasses import dataclass
import math

from app.utils.errors import AiServiceError

FAKE_TERMS = ("fake", "deepfake", "ai", "synthetic", "manipulated", "generated", "spoof")
REAL_TERMS = ("real", "authentic", "original", "genuine", "human")


@dataclass(frozen=True)
class ScoreMappingResult:
    ai_score: float
    real_probability: float
    confidence: float
    raw_labels: dict[str, float]
    mapped_fake_label: str | None
    mapped_real_label: str | None
    mapping_confidence: str


@dataclass(frozen=True)
class ScoreAggregationResult:
    visual_score: float
    mean_score: float
    top_k_mean: float
    p90_score: float
    max_score: float
    high_frame_count: int


def label_hint(ai_score: float) -> str:
    if ai_score >= 0.75:
        return "Likely AI-Generated"
    if ai_score >= 0.55:
        return "Suspicious"
    if ai_score >= 0.35:
        return "Inconclusive"
    return "Likely Real"


def map_label_scores(scores: dict[str, float]) -> ScoreMappingResult:
    fake_probability: float | None = None
    real_probability: float | None = None
    mapped_fake_label: str | None = None
    mapped_real_label: str | None = None

    for label, score in scores.items():
        normalized = _normalize_label(label)
        raw_normalized = label.lower()
        if any(term in normalized for term in FAKE_TERMS):
            score_value = float(score)
            if fake_probability is None or score_value > fake_probability:
                fake_probability = score_value
                mapped_fake_label = label
        elif any(term in normalized for term in REAL_TERMS):
            score_value = float(score)
            if real_probability is None or score_value > real_probability:
                real_probability = score_value
                mapped_real_label = label
        elif raw_normalized.startswith("label_"):
            continue

    if fake_probability is None and real_probability is None:
        raise AiServiceError(
            "AI_SERVICE_INVALID_RESPONSE",
            "Model label mapping is unclear.",
            status_code=502,
        )

    mapping_confidence = "high" if fake_probability is not None and real_probability is not None else "medium"
    if fake_probability is None:
        fake_probability = 1.0 - float(real_probability)
    if real_probability is None:
        real_probability = 1.0 - float(fake_probability)

    fake_probability = _clamp(fake_probability)
    real_probability = _clamp(real_probability)
    confidence = _clamp(max(fake_probability, real_probability))
    return ScoreMappingResult(
        fake_probability,
        real_probability,
        confidence,
        {label: _clamp(score) for label, score in scores.items()},
        mapped_fake_label,
        mapped_real_label,
        mapping_confidence,
    )


def aggregate_frame_scores(scores: list[float]) -> ScoreAggregationResult:
    if not scores:
        raise AiServiceError("AI_SERVICE_INVALID_RESPONSE", "AI service returned no frame scores.", status_code=502)

    normalized_scores = [_clamp(score) for score in scores]
    sorted_scores = sorted(normalized_scores, reverse=True)
    mean_score = sum(normalized_scores) / len(normalized_scores)
    top_k_count = min(len(sorted_scores), max(3, math.ceil(len(sorted_scores) * 0.2)))
    top_k_mean = sum(sorted_scores[:top_k_count]) / top_k_count
    p90_score = _percentile(normalized_scores, 0.90)
    max_score = sorted_scores[0]
    high_frame_count = sum(1 for score in normalized_scores if score >= 0.75)
    visual_score = (0.45 * mean_score) + (0.35 * top_k_mean) + (0.20 * p90_score)

    return ScoreAggregationResult(
        _clamp(visual_score),
        round(mean_score, 4),
        round(top_k_mean, 4),
        round(p90_score, 4),
        round(max_score, 4),
        high_frame_count,
    )


def mapping_debug(mapped: ScoreMappingResult) -> dict:
    return {
        "raw_labels": mapped.raw_labels,
        "mapped_fake_label": mapped.mapped_fake_label,
        "mapped_real_label": mapped.mapped_real_label,
        "mapping_confidence": mapped.mapping_confidence,
    }


def aggregation_debug(aggregation: ScoreAggregationResult) -> dict:
    return {
        "mean_score": aggregation.mean_score,
        "top_k_mean": aggregation.top_k_mean,
        "p90_score": aggregation.p90_score,
        "max_score": aggregation.max_score,
        "high_frame_count": aggregation.high_frame_count,
        "visual_score": round(aggregation.visual_score, 4),
    }


def _normalize_label(label: str) -> str:
    return label.lower().replace("-", " ").replace("_", " ")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
