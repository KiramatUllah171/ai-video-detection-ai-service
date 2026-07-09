import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    service_name: str = "ai-video-detection-ai-service"
    app_env: str = os.getenv("APP_ENV", "development")
    model_version: str = os.getenv("MODEL_VERSION", "mock-video-ai-v1")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
