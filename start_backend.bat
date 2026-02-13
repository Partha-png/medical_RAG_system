@echo off
echo ========================================
echo Medical RAG Backend Startup
echo ========================================
echo.
echo Activating virtual environment...
cd /d "%~dp0"
call venv\Scripts\activate

echo.
echo Starting FastAPI backend server...
echo Backend will be available at: http://localhost:8000
echo API docs will be available at: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

