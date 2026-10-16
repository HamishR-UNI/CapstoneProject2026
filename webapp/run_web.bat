@echo off
rem Starts the web dashboard: http://127.0.0.1:8000
rem Needs the frontend built once:  cd frontend ^&^& npm install ^&^& npm run build
cd /d "%~dp0backend"
if not exist "%~dp0frontend\dist\index.html" (
  echo Frontend not built yet. Run:  cd frontend ^& npm install ^& npm run build
  pause
  exit /b 1
)
echo Starting the simulation server. The first session takes a minute or two to build.
python -m uvicorn app:app --host 127.0.0.1 --port 8000
if errorlevel 1 pause
