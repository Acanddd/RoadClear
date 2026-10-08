param([string]$NodeExe = "")
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
if (Test-Path runtime/processes.json) { throw 'Process record exists. Run ./scripts/stop.ps1 first.' }
foreach ($port in @(8000,5173)) {
  if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) { throw "Port $port is already in use" }
}
if (!$NodeExe -and (Test-Path runtime/toolchain.json)) { $NodeExe = (Get-Content runtime/toolchain.json -Raw | ConvertFrom-Json).node }
if (!$NodeExe) { $NodeExe = (Get-Command node).Source }
if (!(Test-Path frontend/vue/dist/index.html)) { throw 'Build frontend first with setup.ps1' }
New-Item -ItemType Directory -Force runtime/logs | Out-Null
$pythonExe = Join-Path $projectRoot '.venv-local/Scripts/python.exe'
$env:ROADCLEAR_BACKEND_URL = 'http://127.0.0.1:8000'
$records = @()
try {
  $backend = Start-Process -FilePath $pythonExe -ArgumentList @('-m','uvicorn','app.main:app','--app-dir',('"' + (Join-Path $projectRoot 'backend') + '"'),'--host','127.0.0.1','--port','8000','--workers','1') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput runtime/logs/backend.log -RedirectStandardError runtime/logs/backend-error.log -PassThru
  $records += @{id=$backend.Id;start=$backend.StartTime.ToUniversalTime().ToString('o');name='backend'}
  $records | ConvertTo-Json -AsArray | Set-Content runtime/processes.json -Encoding utf8
  $vitePath = Join-Path $projectRoot 'frontend/vue/node_modules/vite/bin/vite.js'
  $frontend = Start-Process -FilePath $NodeExe -ArgumentList @(('"'+$vitePath+'"'),'preview') -WorkingDirectory (Join-Path $projectRoot 'frontend/vue') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $projectRoot 'runtime/logs/frontend.log') -RedirectStandardError (Join-Path $projectRoot 'runtime/logs/frontend-error.log') -PassThru
  $records += @{id=$frontend.Id;start=$frontend.StartTime.ToUniversalTime().ToString('o');name='frontend'}
  $records | ConvertTo-Json -AsArray | Set-Content runtime/processes.json -Encoding utf8
  $ready = $false
  for ($attempt=0; $attempt -lt 90; $attempt++) {
    try {
      $null = Invoke-RestMethod http://127.0.0.1:5173/health -TimeoutSec 2
      $ready = $true; break
    } catch { Start-Sleep -Seconds 1 }
  }
  if (!$ready) { throw 'Startup timeout. See runtime/logs.' }
  Write-Output 'RoadClear: http://127.0.0.1:5173'
  Write-Output 'Model availability: http://127.0.0.1:5173/models/status'
} catch {
  & (Join-Path $PSScriptRoot 'stop.ps1')
  throw
}
