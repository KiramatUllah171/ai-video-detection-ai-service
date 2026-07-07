from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


app = FastAPI(
    title="AI Video Detection Service",
    version="1.0.0"
)


class AnalyzeVideoRequest(BaseModel):
    video_id: str
    video_path: str | None = None
    frame_paths: List[str] = []


class EvidenceItem(BaseModel):
    type: str
    title: str
    description: str
    severity: str
    score_impact: float


class AnalyzeVideoResponse(BaseModel):
    video_id: str
    ai_probability: float
    real_probability: float
    confidence: str
    final_label: str
    evidence: List[EvidenceItem]


@app.get("/")
def health_check():
    return {
        "service": "AI Video Detection Service",
        "status": "running"
    }


@app.post("/analyze-video", response_model=AnalyzeVideoResponse)
def analyze_video(request: AnalyzeVideoRequest):
    return AnalyzeVideoResponse(
        video_id=request.video_id,
        ai_probability=72.5,
        real_probability=27.5,
        confidence="Medium",
        final_label="Likely AI-generated",
        evidence=[
            EvidenceItem(
                type="visual",
                title="Unusual texture patterns",
                description="Some extracted frames show artificial texture consistency.",
                severity="medium",
                score_impact=18.5
            ),
            EvidenceItem(
                type="metadata",
                title="Missing metadata",
                description="Creation metadata is missing or stripped from the video.",
                severity="low",
                score_impact=8.0
            ),
            EvidenceItem(
                type="temporal",
                title="Frame consistency warning",
                description="Minor frame-to-frame inconsistency detected.",
                severity="medium",
                score_impact=12.0
            )
        ]
    )