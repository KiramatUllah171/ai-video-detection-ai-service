@echo off
if exist ".env.local" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env.local") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

set AI_PROVIDER=bitmind
set PROVIDER_MODE=bitmind
set AI_PROVIDER_MODE=bitmind
set AI_MODE=real
set BITMIND_ENABLED=true
if "%BITMIND_BASE_URL%"=="" set BITMIND_BASE_URL=https://api.bitmind.ai/oracle/v1
set EXTERNAL_PROVIDER_POLICY=Always
if "%LOCAL_FALLBACK_ENABLED%"=="" set LOCAL_FALLBACK_ENABLED=false
if "%AI_SERVICE_HOST%"=="" set AI_SERVICE_HOST=127.0.0.1

if "%BITMIND_API_KEY%"=="" (
  echo BITMIND_API_KEY is not set. Put it in .env.local as BITMIND_API_KEY=your_key.
  exit /b 1
)

uvicorn app.main:app --reload --host %AI_SERVICE_HOST% --port 8000
