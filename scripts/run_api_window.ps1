# Launched in its own window by start_parlaypal.ps1 — keeps uvicorn output visible.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$env:PYTHONUNBUFFERED = '1'

Write-Host ''
Write-Host '=== Parlay Pal API (uvicorn)  http://127.0.0.1:8000 ===' -ForegroundColor Cyan
Write-Host "Repo: $root" -ForegroundColor DarkGray
Write-Host ''

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
  Write-Host 'ERROR: python not on PATH in this window. Activate your venv or install Python, then re-run start_parlaypal.ps1.' -ForegroundColor Red
  Read-Host 'Press Enter to close'
  exit 1
}

python -c "import api; print('api module:', api.__file__)"
if ($LASTEXITCODE -ne 0) {
  Write-Host "ERROR: import api failed (exit $LASTEXITCODE). Run from repo root with deps installed." -ForegroundColor Red
  Read-Host 'Press Enter to close'
  exit $LASTEXITCODE
}

python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
Write-Host ''
Write-Host 'uvicorn exited.' -ForegroundColor Yellow
Read-Host 'Press Enter to close'
