@echo off
set AI_PROVIDER=bitmind
set PROVIDER_MODE=bitmind
set AI_PROVIDER_MODE=bitmind
set AI_MODE=real
set BITMIND_ENABLED=true
set BITMIND_BASE_URL=https://api.bitmind.ai/oracle/v1
set EXTERNAL_PROVIDER_POLICY=Always
set LOCAL_FALLBACK_ENABLED=false

if "%BITMIND_API_KEY%"=="" (
  echo BITMIND_API_KEY is not set. Run: set BITMIND_API_KEY=your_key
  exit /b 1
)

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
