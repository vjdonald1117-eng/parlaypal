# Launched in its own window by start_parlaypal.ps1.
$ErrorActionPreference = 'Continue'
$uiRoot = Join-Path (Split-Path -Parent $PSScriptRoot) 'parlay-ui'
Set-Location -LiteralPath $uiRoot

Write-Host ''
Write-Host '=== Parlay Pal UI (Vite)  http://127.0.0.1:5173 ===' -ForegroundColor Cyan
Write-Host "Dir: $uiRoot" -ForegroundColor DarkGray
Write-Host ''

npm run dev -- --host 127.0.0.1
Write-Host ''
Write-Host 'Vite exited.' -ForegroundColor Yellow
Read-Host 'Press Enter to close'
