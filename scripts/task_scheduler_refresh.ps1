param(
  [ValidateSet("today","tomorrow")]
  [string]$Day = "today",

  [int]$NSims = 3000,

  [string]$ApiBase = "http://localhost:8000",

  [int]$WaitSeconds = 60
)

$ErrorActionPreference = "Stop"

function Wait-Api($baseUrl, $waitSeconds) {
  $deadline = (Get-Date).AddSeconds($waitSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      Invoke-RestMethod -Method Get -Uri "$baseUrl/api/health" -TimeoutSec 5 | Out-Null
      return $true
    } catch {
      Start-Sleep -Seconds 2
    }
  }
  return $false
}

if (-not (Wait-Api -baseUrl $ApiBase -waitSeconds $WaitSeconds)) {
  throw "API not reachable at $ApiBase within $WaitSeconds seconds. Start backend first."
}

$uri = "$ApiBase/api/refresh-all?day=$Day&n_sims=$NSims"
Write-Host "Calling $uri"

$resp = Invoke-RestMethod -Method Post -Uri $uri -TimeoutSec 3600
Write-Host ("Done. Stats refreshed: " + ($resp.stats.PSObject.Properties.Name -join ", "))

