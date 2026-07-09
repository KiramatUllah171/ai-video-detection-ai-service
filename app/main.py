from fastapi import FastAPI

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Video Detection AI Service",
        version=settings.model_version,
        description="Mock AI service for frame analysis pipeline integration.",
    )
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()
