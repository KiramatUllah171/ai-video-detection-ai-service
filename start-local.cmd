@echo off
set AI_PROVIDER=local
set PROVIDER_MODE=local
set AI_PROVIDER_MODE=local
set AI_MODE=mock
set BITMIND_ENABLED=false
set EXTERNAL_PROVIDER_POLICY=Disabled
set LOCAL_FALLBACK_ENABLED=true

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
