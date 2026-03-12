@echo off
echo ============================================
echo  Bad Debt Early-Warning API
echo  Starting on http://localhost:8000
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

echo [2/2] Starting API server...
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause
