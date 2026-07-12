from fastapi import FastAPI

from app.api.routes.analyze import router as analyze_router
from app.api.routes.health import router as health_router
from app.core.config import settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="AI Video Detection AI Service",
        version=settings.effective_model_version,
        description="AI service for video authenticity analysis with real/local and mock development modes.",
    )
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()
