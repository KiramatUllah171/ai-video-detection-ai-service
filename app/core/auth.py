import secrets

from fastapi import Header, HTTPException, status

from app.core.config import settings


def require_internal_api_key(x_ai_service_key: str | None = Header(default=None)) -> None:
    if not settings.require_api_key:
        return

    if not settings.ai_service_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service API key is not configured.",
        )

    if not x_ai_service_key or not secrets.compare_digest(x_ai_service_key, settings.ai_service_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized AI service request.",
        )
