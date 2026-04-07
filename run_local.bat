@echo off
cd /d "%~dp0"
where streamlit >nul 2>&1
if errorlevel 1 (
  echo streamlit not found. Run: pip install -r requirements-local.txt
  pause
  exit /b 1
)
echo.
echo Starting app... If the wrong browser opens, set Chrome as default or use run_local_chrome.bat
echo Manual URL: http://127.0.0.1:8501  or  http://localhost:8501
echo.
streamlit run streamlit_app.py
pause
