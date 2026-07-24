@echo off
if exist ".env.local" (
  for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env.local") do (
    if not "%%A"=="" set "%%A=%%B"
  )
)

set AI_PROVIDER=local
set PROVIDER_MODE=local
set AI_PROVIDER_MODE=local
set AI_MODE=mock
set BITMIND_ENABLED=false
set EXTERNAL_PROVIDER_POLICY=Disabled
set LOCAL_FALLBACK_ENABLED=true
if "%AI_SERVICE_HOST%"=="" set AI_SERVICE_HOST=127.0.0.1

uvicorn app.main:app --reload --host %AI_SERVICE_HOST% --port 8000
