import logging

from fastapi import APIRouter, HTTPException

from app.models.requests import AnalyzeFramesRequest
from app.models.responses import AnalyzeFramesResponse
from app.services.mock_ai_service import MockAiService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])
mock_ai_service = MockAiService()


@router.post("/analyze-frames", response_model=AnalyzeFramesResponse)
def analyze_frames(request: AnalyzeFramesRequest) -> AnalyzeFramesResponse:
    try:
        return mock_ai_service.analyze_frames(request)
    except Exception as exception:
        logger.exception(
            "Mock frame analysis failed for video_id=%s job_id=%s",
            request.video_id,
            request.job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Frame analysis failed.",
        ) from exception
