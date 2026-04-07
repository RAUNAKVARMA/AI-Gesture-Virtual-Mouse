@echo off
cd /d "%~dp0"
where streamlit >nul 2>&1
if errorlevel 1 (
  echo streamlit not found. Run: pip install -r requirements-local.txt
  pause
  exit /b 1
)

set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"

if not exist "%CHROME%" (
  echo Chrome not found in the usual install paths.
  echo Set Chrome as your default browser, or open this URL manually in Chrome:
  echo   http://127.0.0.1:8501
  pause
  exit /b 1
)

echo Starting server in a separate window, then opening Chrome...
echo If the tab is empty, wait a few seconds and press Refresh.
echo.
start "Streamlit AI-Gesture-Virtual-Mouse" /D "%~dp0" cmd /k streamlit run streamlit_app.py --server.headless true
timeout /t 6 /nobreak >nul
start "" "%CHROME%" "http://127.0.0.1:8501"
echo.
echo Close the black "Streamlit" window when you want to stop the server.
pause
