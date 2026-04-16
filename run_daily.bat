@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM Prefer repo venv Python when present (double-click safe).
if exist "%~dp0venv\Scripts\python.exe" (
  set "PY=%~dp0venv\Scripts\python.exe"
) else (
  set "PY=python"
)

set PYTHONUNBUFFERED=1

echo.
echo === ParlayPal daily: sync ^&^& train ^&^& predict ^&^& prep-sim ^&^& possession-sim ===
echo.

"%PY%" -u scripts\sync_nba_schedule.py && "%PY%" -u train_model.py && "%PY%" -u predict_today.py && "%PY%" -u scripts\prep_simulation.py && "%PY%" -u scripts\possession_simulator.py --prep "data\simulation_prep_tonight.json" --n-sims 300
if errorlevel 1 goto :fail

echo.
echo === ParlayPal daily: finished OK ===
pause
exit /b 0

:fail
echo.
echo === ParlayPal daily: FAILED (exit %ERRORLEVEL%) ===
pause
exit /b 1
