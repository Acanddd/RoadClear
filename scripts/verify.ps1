param([string]$BaseUrl = "")
$ErrorActionPreference='Stop'
$projectRoot=Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$pythonExe=Join-Path $projectRoot '.venv-local/Scripts/python.exe'
if ($BaseUrl) {
  & $pythonExe scripts/validate.py --base-url $BaseUrl --report runtime/validation-http.json
} else {
  & $pythonExe scripts/validate.py
}
if ($LASTEXITCODE -ne 0) { throw 'End-to-end validation failed' }
& $pythonExe -m pytest backend/tests/test_local_profile.py -q
if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
