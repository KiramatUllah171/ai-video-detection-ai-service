# AI Video Detection AI Service

FastAPI service for AI Video Detection frame analysis.

The current model is a deterministic mock only. It is intended for backend pipeline integration and must not be treated as real AI detection.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```text
GET http://localhost:8000/health
```

Open API docs:

```text
http://localhost:8000/docs
```

## Tests

```powershell
pytest
```

## Docker

```powershell
docker build -t ai-video-detection-ai-service .
docker run -p 8000:8000 ai-video-detection-ai-service
```

## Configuration

Defaults work without an `.env` file.

| Variable | Default |
| --- | --- |
| `APP_ENV` | `development` |
| `MODEL_VERSION` | `mock-video-ai-v1` |
| `LOG_LEVEL` | `INFO` |

## Endpoints

### GET /health

Response:

```json
{
  "status": "healthy",
  "service": "ai-video-detection-ai-service",
  "model_version": "mock-video-ai-v1",
  "environment": "development"
}
```

### POST /analyze-frames

Request:

```json
{
  "video_id": 123,
  "job_id": 456,
  "frames": [
    {
      "frame_id": 1,
      "frame_url": "frames/123/frame_000001.jpg",
      "frame_index": 1,
      "timestamp_seconds": 2.0
    }
  ]
}
```

Response:

```json
{
  "video_id": 123,
  "job_id": 456,
  "model_version": "mock-video-ai-v1",
  "overall_ai_score": 0.62,
  "overall_confidence": 0.78,
  "label_hint": "Suspicious",
  "frames": [
    {
      "frame_id": 1,
      "frame_index": 1,
      "timestamp_seconds": 2.0,
      "ai_score": 0.64,
      "confidence": 0.81,
      "notes": [
        "Mock score generated from deterministic frame identifier hash."
      ]
    }
  ],
  "notes": [
    "This is a mock AI response for pipeline integration only.",
    "Do not treat this result as real AI detection."
  ]
}
```

Exact scores depend on the deterministic frame identifier hash.
