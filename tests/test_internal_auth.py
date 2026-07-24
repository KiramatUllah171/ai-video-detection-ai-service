from dataclasses import replace

import pytest
from fastapi import HTTPException

from app.core import auth
from app.core.config import settings


def test_internal_api_key_is_optional_in_development_when_unset(monkeypatch):
    monkeypatch.setattr(auth, "settings", replace(settings, app_env="development", ai_service_api_key=None))

    auth.require_internal_api_key()


def test_internal_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(auth, "settings", replace(settings, app_env="development", ai_service_api_key="secret"))

    with pytest.raises(HTTPException) as exception:
        auth.require_internal_api_key("wrong")

    assert exception.value.status_code == 401
    auth.require_internal_api_key("secret")


def test_internal_api_key_is_required_in_production_even_when_missing(monkeypatch):
    monkeypatch.setattr(auth, "settings", replace(settings, app_env="production", ai_service_api_key=None))

    with pytest.raises(HTTPException) as exception:
        auth.require_internal_api_key()

    assert exception.value.status_code == 503
