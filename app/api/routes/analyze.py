import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.requests import AnalyzeFramesRequest, AnalyzeVideoRequest
from app.models.responses import AnalyzeFramesResponse
from app.core.config import settings
from app.services.model_loader import model_loader
from app.services.providers import BitMindAiProvider, LocalAiProvider, ProviderOrchestrator
from app.utils.errors import AiServiceError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analysis"])


@router.post("/analyze-frames", response_model=AnalyzeFramesResponse)
def analyze_frames(request: AnalyzeFramesRequest):
    try:
        if settings.ai_mode == "real" and settings.provider_mode == "local":
            missing_images = [frame.frame_id for frame in request.frames if not frame.image_base64]
            if missing_images:
                raise AiServiceError(
                    "FRAME_IMAGE_REQUIRED",
                    "Frame image content is required for real local AI analysis.",
                    status_code=400,
                )

        service = model_loader.get_service()
        return service.analyze_frames(request)
    except AiServiceError as exception:
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "success": False,
                "error_code": exception.error_code,
                "message": exception.message,
            },
        )


@router.post("/analyze-video", response_model=AnalyzeFramesResponse)
def analyze_video(request: AnalyzeVideoRequest):
    try:
        provider_mode = (request.provider_mode or settings.ai_provider or "local").lower()
        service = model_loader.get_service()
        orchestrator = ProviderOrchestrator(LocalAiProvider(service), BitMindAiProvider(settings), settings)
        return orchestrator.analyze_video(request.model_copy(update={"provider_mode": provider_mode}))
    except AiServiceError as exception:
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "success": False,
                "error_code": exception.error_code,
                "message": exception.message,
            },
        )
    except Exception:
        logger.exception(
            "Video analysis failed for video_id=%s job_id=%s provider_mode=%s",
            request.video_id,
            request.job_id,
            request.provider_mode,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "AI_VIDEO_ANALYSIS_FAILED",
                "message": "AI video analysis failed.",
            },
        )


@router.post("/debug/analyze-frames-detailed")
def analyze_frames_detailed(request: AnalyzeFramesRequest):
    if settings.app_env.lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")

    try:
        service = model_loader.get_service()
        response = service.analyze_frames(request)
        return response.model_dump()
    except AiServiceError as exception:
        return JSONResponse(
            status_code=exception.status_code,
            content={
                "success": False,
                "error_code": exception.error_code,
                "message": exception.message,
            },
        )
    except Exception as exception:
        logger.exception(
            "Frame analysis failed for video_id=%s job_id=%s",
            request.video_id,
            request.job_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_code": "AI_INFERENCE_FAILED",
                "message": "AI inference failed.",
            },
        )
