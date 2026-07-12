# AI Video Detection AI Service

FastAPI service for frame/video authenticity analysis. It supports:

- `AI_MODE=mock` for deterministic development responses.
- `AI_MODE=real` with local Hugging Face/PyTorch model loading.
- Optional hosted mode placeholder for future Hugging Face provider integration.

Current recommended free model IDs:

- `SoraExplora/VideoMae` as the preferred video-classification model.
- `Naman712/Deep-fake-detection` as a secondary video model if it loads cleanly.
- `prithivMLmods/Deep-Fake-Detector-v2-Model` as a frame-level image fallback.

The service never treats mock output as real. Mock responses include `is_mock=true` and a warning.

## Configuration

```powershell
$env:AI_MODE="mock"              # mock | real
$env:AI_PROVIDER_MODE="local"    # local | hosted
$env:ENABLE_MOCK_FALLBACK="false"
$env:MODEL_ID="SoraExplora/VideoMae"
$env:MODEL_VERSION="real-video-ai-v1"
$env:DEVICE="auto"
$env:MODEL_CACHE_DIR="./models/cache"
$env:ALLOW_MODEL_DOWNLOAD="false"
```

Real local mode requires frame images as `image_base64` in `/analyze-frames`.

If `AI_MODE=real` and no model can load while `ENABLE_MOCK_FALLBACK=false`, the service returns:

```json
{
  "success": false,
  "error_code": "REAL_MODEL_NOT_CONFIGURED",
  "message": "Real AI model is not configured. Configure MODEL_ID/MODEL_PATH or enable mock mode for development."
}
```

By default `ALLOW_MODEL_DOWNLOAD=false`, so real mode uses only locally cached model files and fails fast instead of blocking an API request while downloading large weights. To intentionally download model weights during development:

```powershell
$env:ALLOW_MODEL_DOWNLOAD="true"
```

Use a longer backend timeout during first download/model warmup.

### Real model diagnostics and ensemble mode

For local debugging:

```powershell
$env:AI_MODE="real"
$env:AI_PROVIDER_MODE="local"
$env:ENABLE_MOCK_FALLBACK="false"
$env:AI_DEBUG_OUTPUT="true"
$env:MODEL_FRAME_COUNT="16"
```

Useful endpoints:

- `GET /model-diagnostics`
- `POST /model-diagnostics/load`
- `POST /debug/analyze-frames-detailed`

To enable the free local ensemble:

```powershell
$env:AI_ENSEMBLE_ENABLED="true"
$env:PRIMARY_VIDEO_MODEL_ID="SoraExplora/VideoMae"
$env:FRAME_MODEL_ID="prithivMLmods/Deep-Fake-Detector-v2-Model"
$env:SECONDARY_VIDEO_MODEL_ID="Naman712/Deep-fake-detection"
$env:ENSEMBLE_VIDEO_WEIGHT="0.65"
$env:ENSEMBLE_FRAME_WEIGHT="0.35"
```

Calibration defaults are intentionally conservative because the frame-level detector can over-score real mobile videos:

```powershell
$env:CALIBRATION_ENABLED="true"
$env:FRAME_BIAS_OFFSET="0.15"
$env:FRAME_TEMPERATURE="1.25"
$env:USE_TEMPERATURE_CALIBRATION="true"
$env:PRESERVE_STRONG_RAW_SIGNAL="true"
$env:STRONG_RAW_FRAME_THRESHOLD="0.85"
$env:STRONG_CALIBRATED_FRAME_THRESHOLD="0.65"
$env:MODEL_DISAGREEMENT_THRESHOLD="0.30"
$env:FRAME_DETECTOR_MAX_WEIGHT="0.35"
$env:VIDEO_DETECTOR_MAX_WEIGHT="0.65"
$env:UNRELIABLE_DETECTOR_WEIGHT="0.10"
$env:RELIABILITY_REPORT_PATH="calibration_reports/latest_evaluation.json"
```

Run a small local sample evaluation:

```powershell
mkdir sample_data\real
mkdir sample_data\ai
python scripts/evaluate_samples.py --root sample_data --url http://localhost:8000/analyze-frames
```

The evaluator writes `calibration_reports/latest_evaluation.json` and `calibration_reports/latest_evaluation.csv` with video score, raw frame score, calibrated frame score, final score, disagreement flags, false positives, and false negatives.

Do not commit sample videos.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:AI_MODE="mock"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health:

```http
GET http://localhost:8000/health
```

Docs:

```http
http://localhost:8000/docs
```

## Analyze Frames

```http
POST /analyze-frames
```

```json
{
  "video_id": 123,
  "job_id": 456,
  "frames": [
    {
      "frame_id": 1,
      "frame_url": "frames/123/frame_000001.jpg",
      "frame_index": 1,
      "timestamp_seconds": 2.0,
      "image_base64": "optional-in-mock-required-in-real-local"
    }
  ]
}
```

Response includes model identity, capability, mock flag, probabilities, confidence, warnings, and per-frame scores.

## Tests

```powershell
python -m pytest
```

Tests do not require GPU or downloaded model weights.

## Docker

```powershell
docker build -t ai-video-detection-ai-service .
docker run -p 8000:8000 -e AI_MODE=mock ai-video-detection-ai-service
```

## Limitations

- Results are probability-based and not proof.
- Frame-level fallback models do not evaluate temporal video consistency.
- Mock mode is for development only and is marked in every response.
- Hosted inference is not enabled by default and must not hardcode API tokens.
