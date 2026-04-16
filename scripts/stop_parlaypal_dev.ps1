<#
.SYNOPSIS
  Stop Parlay Pal dev servers: listeners on API/UI ports and matching uvicorn/vite orphans for this repo.

.DESCRIPTION
  Frees 127.0.0.1:8000 (uvicorn) and :5173 (Vite) so a new run does not hit stale processes or HTTP 409 from
  a hidden backend. Optionally targets only specific ports.

.PARAMETER Ports
  TCP ports to clear (default 8000, 5173).

.PARAMETER Quiet
  Minimal output.
#>
param(
  [int[]]$Ports = @(8000, 5173),
  [switch]$Quiet
)

function Write-Info([string]$msg) {
  if (-not $Quiet) { Write-Host $msg -ForegroundColor Yellow }
}

function Invoke-TaskkillSafe([int]$ProcessId) {
  # Some shells enable PSNativeCommandUseErrorActionPreference, which can turn
  # taskkill "not found" into a terminating error. Force non-terminating behavior.
  $prevNativePref = $null
  if ($PSVersionTable.PSVersion.Major -ge 7) {
    $prevNativePref = $PSNativeCommandUseErrorActionPreference
    $PSNativeCommandUseErrorActionPreference = $false
  }
  try {
    taskkill /F /PID $ProcessId 2>&1 | Out-Null
  } catch {
    # Ignore "process not found" and other best-effort cleanup failures.
  } finally {
    if ($PSVersionTable.PSVersion.Major -ge 7) {
      $PSNativeCommandUseErrorActionPreference = $prevNativePref
    }
  }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoEscaped = [regex]::Escape($repoRoot)

foreach ($port in $Ports) {
  $conns = @(
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  )
  foreach ($c in $conns) {
    $procId = [int]$c.OwningProcess
    if ($procId -le 0) { continue }
    Write-Info "Stopping PID $procId listening on port $port"
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Invoke-TaskkillSafe -ProcessId $procId
  }
}

# Orphans not in Listen (rare): same repo + uvicorn api:app / vite
try {
  Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    $cl = $_.CommandLine
    if (
      $cl -and
      $cl -match 'uvicorn' -and
      $cl -match 'api:app' -and
      $cl -match $repoEscaped
    ) {
      Write-Info "Stopping python uvicorn PID $($_.ProcessId) (repo match)"
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
  }
} catch { }

try {
  Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    $cl = $_.CommandLine
    if (
      $cl -and
      $cl -match 'vite' -and
      $cl -match $repoEscaped
    ) {
      Write-Info "Stopping node/vite PID $($_.ProcessId) (repo match)"
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
  }
} catch { }

Start-Sleep -Milliseconds 400
if (-not $Quiet) {
  Write-Host "Parlay Pal dev cleanup done (ports $($Ports -join ', '))." -ForegroundColor Green
}
