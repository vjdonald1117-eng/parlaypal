# Start ParlayPal FastAPI (run from any folder).
# If you see no output: use "powershell -ExecutionPolicy Bypass -File .\scripts\start_backend.ps1"
# Or paste:  cd parlaypal; python -m uvicorn api:app --host 127.0.0.1 --port 8000

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONUNBUFFERED = '1'

$busy = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($busy) {
    $opid = $busy.OwningProcess
    Write-Host "Port 8000 in use (PID $opid). Stopping it so you run this folder's API..." -ForegroundColor Yellow
    Stop-Process -Id $opid -Force -ErrorAction SilentlyContinue
    taskkill /F /PID $opid 2>$null | Out-Null
    Start-Sleep -Milliseconds 600
    $still = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($still) {
        Write-Host "Could not free 8000 (try Task Manager as Administrator, or: taskkill /F /PID $opid)." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "ParlayPal API  ->  http://127.0.0.1:8000  (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "Working dir: $(Get-Location)" -ForegroundColor DarkGray
Write-Host ""

python -m uvicorn api:app --host 127.0.0.1 --port 8000
