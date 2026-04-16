# Single Cursor/VS Code terminal: [api] uvicorn + [ui] Vite (always visible).
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
& "$PSScriptRoot\stop_parlaypal_dev.ps1"
Set-Location -LiteralPath $root
$env:PYTHONUNBUFFERED = '1'

$npx = Get-Command npx -ErrorAction SilentlyContinue
if (-not $npx) {
  Write-Host 'npx not on PATH — opening two PowerShell windows instead.' -ForegroundColor Yellow
  & "$PSScriptRoot\start_parlaypal.ps1"
  exit 0
}

Write-Host ''
Write-Host '[Parlay Pal] One terminal: lines prefixed [api] = uvicorn, [ui] = Vite' -ForegroundColor Green
Write-Host 'App: http://127.0.0.1:5173/   API: http://127.0.0.1:8000/api/health' -ForegroundColor DarkGray
Write-Host ''

$uiPath = Join-Path $root 'parlay-ui'
# Run Vite from parlay-ui within concurrently.
$apiCmd = 'python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload'
$viteCmd = "npm run dev --prefix ""$uiPath"" -- --host 127.0.0.1"

# Pin concurrently to v8 on Windows PowerShell to avoid v9 argv parsing issues
# where --names can become a non-string under nested shell invocation.
npx --yes concurrently@8.2.2 `
  --kill-others `
  --names "api,ui" `
  --prefix-colors "cyan,magenta" `
  "$apiCmd" `
  "$viteCmd"
