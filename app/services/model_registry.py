from dataclasses import asdict, dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class DetectorRegistryEntry:
    model_id: str
    model_type: str
    capability: str
    labels: list[str]
    reliability_score: float | None
    enabled: bool
    notes: str


def build_model_registry(settings: Settings, reliability: dict | None = None) -> list[dict]:
    reliability = reliability or {}
    entries = [
        DetectorRegistryEntry(
            model_id=settings.primary_video_model_id,
            model_type="video",
            capability="video_temporal",
            labels=["real", "fake"],
            reliability_score=_reliability_score(reliability, "video"),
            enabled=True,
            notes="Primary temporal detector. Calibrate against sample_data before trusting strong labels.",
        )
    ]

    for model_id in settings.frame_model_ids:
        entries.append(
            DetectorRegistryEntry(
                model_id=model_id,
                model_type="frame",
                capability="frame_image",
                labels=["real", "fake", "synthetic", "deepfake"],
                reliability_score=_reliability_score(reliability, "frame_calibrated"),
                enabled=model_id == settings.frame_model_id,
                notes="Frame image detector. Useful for visual artifacts, but may be noisy on compressed or stylized video.",
            )
        )

    return [asdict(entry) for entry in entries]


def _reliability_score(reliability: dict, key: str) -> float | None:
    value = reliability.get(key) or {}
    score = value.get("accuracy")
    return float(score) if score is not None else None
