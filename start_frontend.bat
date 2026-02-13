@echo off
echo ========================================
echo Medical RAG Frontend Startup
echo ========================================
echo.
echo Activating virtual environment...
cd /d "%~dp0"
call venv\Scripts\activate

echo.
echo Starting Streamlit frontend...
echo Frontend will open automatically in your browser
echo URL: http://localhost:8501
echo.
echo Press CTRL+C to stop the server
echo.

streamlit run app.py
