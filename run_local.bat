@echo off
cd /d "%~dp0"
where streamlit >nul 2>&1
if errorlevel 1 (
  echo streamlit not found. Run: pip install -r requirements-local.txt
  pause
  exit /b 1
)
echo.
echo Starting app... If the browser does not open, go to: http://localhost:8501
echo.
streamlit run streamlit_app.py
pause
