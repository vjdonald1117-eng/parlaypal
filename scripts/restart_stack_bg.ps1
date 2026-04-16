# Free 8000 + 5173, then start API + Vite in background (no new windows).
$ErrorActionPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot

foreach ($port in @(8000, 5173)) {
    $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        if ($procId -gt 0) {
            Write-Host "Stopping PID $procId on port $port"
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
}
Start-Sleep -Milliseconds 800

# Prefer venv only if it can import FastAPI (some venvs are incomplete).
$venvPy = Join-Path $root 'venv\Scripts\python.exe'
$py = 'python'
if (Test-Path $venvPy) {
    & $venvPy -c "import fastapi" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $py = $venvPy }
}

$env:PYTHONUNBUFFERED = '1'
Set-Location $root
Start-Process -FilePath $py -ArgumentList @('-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', '8000') -WindowStyle Hidden
Start-Sleep -Milliseconds 600
$uiDir = Join-Path $root 'parlay-ui'
$npmCmd = "cd /d `"$uiDir`" && npm run dev -- --host 127.0.0.1"
Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $npmCmd) -WindowStyle Hidden

Write-Host 'Started API http://127.0.0.1:8000 and UI http://127.0.0.1:5173 (background)' -ForegroundColor Green
