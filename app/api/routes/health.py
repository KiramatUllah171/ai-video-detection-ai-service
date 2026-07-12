from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.responses import HealthResponse, ModelDiagnosticsResponse
from app.services.model_loader import model_loader

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_health = model_loader.health()
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        **model_health,
    )


@router.get("/model-diagnostics", response_model=ModelDiagnosticsResponse)
def model_diagnostics() -> ModelDiagnosticsResponse:
    model_health = model_loader.health()
    return ModelDiagnosticsResponse(
        status="healthy",
        service=settings.service_name,
        **model_health,
    )


@router.post("/model-diagnostics/load")
def force_load_model() -> dict:
    if settings.app_env.lower() == "production":
        raise HTTPException(status_code=404, detail="Not found")

    return model_loader.force_load()
