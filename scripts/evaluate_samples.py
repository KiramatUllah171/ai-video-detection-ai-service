import argparse
import base64
import csv
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from urllib import request as urlrequest


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local real/AI samples against the AI service.")
    parser.add_argument("--root", default="sample_data", help="Sample root containing real/ and ai/ folders.")
    parser.add_argument("--url", default="http://localhost:8000/analyze-frames", help="AI service analyze endpoint.")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames to sample per video/folder.")
    parser.add_argument("--out", default="calibration_reports", help="Directory for JSON/CSV calibration reports.")
    args = parser.parse_args()

    rows: list[dict] = []
    for expected in ["real", "ai"]:
        folder = Path(args.root) / expected
        if not folder.exists():
            continue
        for sample in sorted(folder.iterdir()):
            if sample.name.startswith("."):
                continue
            frames = collect_frames(sample, args.frames)
            if not frames:
                print(f"Skipping {sample}: no frames found")
                continue
            response = analyze(args.url, sample.name, expected, frames)
            score = float(response["overall_ai_score"])
            label_hint = str(response.get("label_hint", ""))
            predicted = classify(score, label_hint)
            component_scores = response.get("component_scores") or {}
            video_component = component_scores.get("video") or {}
            frame_component = component_scores.get("frame") or {}
            combined_component = component_scores.get("combined") or {}
            rows.append({
                "file": sample.name,
                "expected_label": expected,
                "predicted_label": predicted,
                "label_hint": label_hint,
                "ai_score": score,
                "real_score": float(response["real_probability"]),
                "confidence": float(response["overall_confidence"]),
                "video_score": nullable_float(video_component.get("ai_score")),
                "raw_frame_score": nullable_float(frame_component.get("raw_frame_ai_score")),
                "calibrated_frame_score": nullable_float(frame_component.get("calibrated_frame_ai_score") or frame_component.get("ai_score")),
                "final_score": score,
                "model_disagreement": bool(response.get("model_disagreement", False)),
                "minimum_recommended_score": response.get("minimum_recommended_score"),
                "ensemble_strategy": response.get("ensemble_strategy"),
                "combined_weighted_average": nullable_float(combined_component.get("weighted_average")),
                "model_id": response["model_id"],
                "model_capability": response["model_capability"],
                "correct": is_development_correct(expected, predicted),
                "warnings": response.get("warnings", []),
            })

    print("file\texpected\tpredicted\tai_score\tvideo_score\traw_frame_score\tcalibrated_frame_score\tconfidence\tstrategy\tcorrect")
    for row in rows:
        print("\t".join([
            str(row["file"]),
            str(row["expected_label"]),
            str(row["predicted_label"]),
            f"{row['ai_score']:.4f}",
            format_optional(row["video_score"]),
            format_optional(row["raw_frame_score"]),
            format_optional(row["calibrated_frame_score"]),
            f"{row['confidence']:.4f}",
            str(row["ensemble_strategy"]),
            str(row["correct"]),
        ]))

    if rows:
        summary = build_summary(rows)
        print()
        print(f"average_real_ai_score={summary['average_real_ai_score']:.4f}" if summary["average_real_ai_score"] is not None else "average_real_ai_score=n/a")
        print(f"average_ai_ai_score={summary['average_ai_ai_score']:.4f}" if summary["average_ai_ai_score"] is not None else "average_ai_ai_score=n/a")
        print(f"rough_accuracy={summary['rough_accuracy']:.2%}")
        print(f"false_positives={summary['false_positives']}")
        print(f"false_negatives={summary['false_negatives']}")
        write_reports(Path(args.out), rows, summary, args)


def collect_frames(sample: Path, frame_count: int) -> list[Path]:
    if sample.is_dir():
        images = [path for path in sorted(sample.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
        return uniform_sample(images, frame_count)
    if sample.suffix.lower() in IMAGE_EXTENSIONS:
        return [sample]
    if sample.suffix.lower() in VIDEO_EXTENSIONS:
        temp_dir = Path(tempfile.mkdtemp(prefix="ai-video-eval-"))
        pattern = temp_dir / "frame_%06d.jpg"
        duration = probe_duration(sample)
        fps = max(frame_count / duration, 0.1) if duration > 0 else 1
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(sample), "-vf", f"fps={fps}", "-frames:v", str(frame_count), str(pattern)],
            check=False,
        )
        images = [path for path in sorted(temp_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
        return uniform_sample(images, frame_count)
    return []


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def uniform_sample(items: list[Path], count: int) -> list[Path]:
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


def classify(score: float, label_hint: str) -> str:
    normalized_hint = label_hint.strip().lower()
    if "inconclusive" in normalized_hint:
        return "inconclusive"
    if score >= 0.55:
        return "suspicious" if score < 0.70 else "ai"
    if score < 0.30:
        return "real"
    return "inconclusive"


def is_development_correct(expected: str, predicted: str) -> bool:
    if expected == "real":
        return predicted in {"real", "inconclusive"}
    if expected == "ai":
        return predicted in {"ai", "suspicious"}
    return expected == predicted


def nullable_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_summary(rows: list[dict]) -> dict:
    real_scores = [row["ai_score"] for row in rows if row["expected_label"] == "real"]
    ai_scores = [row["ai_score"] for row in rows if row["expected_label"] == "ai"]
    false_positives = [row for row in rows if row["expected_label"] == "real" and row["predicted_label"] == "ai"]
    false_negatives = [row for row in rows if row["expected_label"] == "ai" and row["predicted_label"] == "real"]
    correct_count = sum(1 for row in rows if row["correct"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(rows),
        "rough_accuracy": correct_count / len(rows),
        "average_real_ai_score": fmean(real_scores) if real_scores else None,
        "average_ai_ai_score": fmean(ai_scores) if ai_scores else None,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "suspicious_count": sum(1 for row in rows if row["predicted_label"] == "suspicious"),
        "inconclusive_count": sum(1 for row in rows if row["predicted_label"] == "inconclusive"),
        "detector_reliability": {
            "video": detector_reliability(rows, "video_score"),
            "frame_raw": detector_reliability(rows, "raw_frame_score"),
            "frame_calibrated": detector_reliability(rows, "calibrated_frame_score"),
        },
    }


def detector_reliability(rows: list[dict], field: str) -> dict:
    valid = [row for row in rows if row.get(field) is not None and row["expected_label"] in {"real", "ai"}]
    if not valid:
        return {"sample_count": 0, "accuracy": None, "average_real_score": None, "average_ai_score": None}

    real_scores = [row[field] for row in valid if row["expected_label"] == "real"]
    ai_scores = [row[field] for row in valid if row["expected_label"] == "ai"]
    correct = 0
    for row in valid:
        predicted = "ai" if row[field] >= 0.55 else "real" if row[field] < 0.30 else "inconclusive"
        correct += predicted == row["expected_label"]

    return {
        "sample_count": len(valid),
        "accuracy": correct / len(valid),
        "average_real_score": fmean(real_scores) if real_scores else None,
        "average_ai_score": fmean(ai_scores) if ai_scores else None,
    }


def write_reports(output_dir: Path, rows: list[dict], summary: dict, args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": {
            "root": args.root,
            "url": args.url,
            "frames": args.frames,
        },
        "summary": summary,
        "rows": rows,
    }
    json_path = output_dir / "latest_evaluation.json"
    csv_path = output_dir / "latest_evaluation.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "file",
            "expected_label",
            "predicted_label",
            "label_hint",
            "ai_score",
            "real_score",
            "confidence",
            "video_score",
            "raw_frame_score",
            "calibrated_frame_score",
            "final_score",
            "model_disagreement",
            "minimum_recommended_score",
            "ensemble_strategy",
            "combined_weighted_average",
            "model_id",
            "model_capability",
            "correct",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    print(f"wrote_json={json_path}")
    print(f"wrote_csv={csv_path}")


def analyze(url: str, name: str, expected: str, frames: list[Path]) -> dict:
    payload = {
        "video_id": abs(hash(name)) % 1_000_000 + 1,
        "job_id": abs(hash((name, expected))) % 1_000_000 + 1,
        "frames": [
            {
                "frame_id": index + 1,
                "frame_url": str(frame),
                "frame_index": index,
                "timestamp_seconds": float(index),
                "image_base64": base64.b64encode(frame.read_bytes()).decode("ascii"),
            }
            for index, frame in enumerate(frames)
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
