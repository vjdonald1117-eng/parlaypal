@echo off
echo ============================================================
echo  ParlayPal Daily Update
echo ============================================================

cd "C:\Users\Jayyy\OneDrive\Desktop\parlaypal"

echo.
echo [1/3] Syncing last 48h box scores...
python update_recent_games.py
if %errorlevel% neq 0 (
    echo ERROR: update_recent_games.py failed. Aborting.
    pause
    exit /b 1
)

echo.
echo [2/3] Scraping injuries + carry-forward...
python scrape_injuries.py
if %errorlevel% neq 0 (
    echo ERROR: scrape_injuries.py failed. Aborting.
    pause
    exit /b 1
)

echo.
echo [3/3] Building tomorrow's parlay dashboard...
python models/parlay_builder.py
if %errorlevel% neq 0 (
    echo ERROR: parlay_builder.py failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  ParlayPal Update Complete!
echo ============================================================
pause