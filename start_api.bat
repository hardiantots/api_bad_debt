@echo off
setlocal

echo ============================================
echo  Bad Debt Early-Warning API
echo  Starting service...
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+
    pause
    exit /b 1
)

if "%API_HOST%"=="" set API_HOST=0.0.0.0
if "%API_PORT%"=="" set API_PORT=8000
if "%UVICORN_RELOAD%"=="" set UVICORN_RELOAD=false

echo [2/2] Starting API server on http://%API_HOST%:%API_PORT%

if /I "%UVICORN_RELOAD%"=="true" (
    python -m uvicorn api:app --host %API_HOST% --port %API_PORT% --reload
) else (
    python -m uvicorn api:app --host %API_HOST% --port %API_PORT%
)

pause
