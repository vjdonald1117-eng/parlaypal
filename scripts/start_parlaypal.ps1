# Start API + UI in two new PowerShell windows (logs stay visible).
# Uses -File launchers so paths/quotes don't break inline -Command (common on OneDrive paths).
$root = Split-Path -Parent $PSScriptRoot

& "$PSScriptRoot\stop_parlaypal_dev.ps1"

$shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { 'pwsh' } else { 'powershell' }
$apiScript = Join-Path $PSScriptRoot 'run_api_window.ps1'
$uiScript  = Join-Path $PSScriptRoot 'run_ui_window.ps1'

Start-Process $shell -ArgumentList '-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $apiScript
Start-Sleep -Milliseconds 600
Start-Process $shell -ArgumentList '-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $uiScript

Write-Host "Started ParlayPal in two windows: API :8000  |  UI :5173" -ForegroundColor Green
Write-Host "Open the app: http://127.0.0.1:5173/" -ForegroundColor Green
