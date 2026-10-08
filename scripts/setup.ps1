param([string]$PythonExe = "python", [string]$NodeExe = "node", [string]$PnpmCmd = "pnpm")
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$pythonPath = (Get-Command $PythonExe -ErrorAction Stop).Source
$nodePath = (Get-Command $NodeExe -ErrorAction Stop).Source
$pnpmPath = (Get-Command $PnpmCmd -ErrorAction Stop).Source
& $pythonPath -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 required; pass -PythonExe with its path' }
& $nodePath -e "const [a,b]=process.versions.node.split('.').map(Number); if(a<22 || (a===22 && b<13))process.exit(1)"
if ($LASTEXITCODE -ne 0) { throw 'Node >=22.13 required' }
$pnpmVersion = & $pnpmPath --version
if ($pnpmVersion -ne '11.19.0') { throw 'Use pnpm 11.19.0 (npm install -g pnpm@11.19.0)' }
if (!(Test-Path '.venv-local/Scripts/python.exe')) { & $pythonPath -m venv .venv-local }
& ./.venv-local/Scripts/python.exe -m pip install -r backend/requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
& ./.venv-local/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Python dependency conflict' }
Push-Location frontend/vue
try {
  & $pnpmPath install --frozen-lockfile
  if ($LASTEXITCODE -ne 0) { throw 'Frontend installation failed' }
  & $pnpmPath build
  if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed' }
} finally { Pop-Location }
New-Item -ItemType Directory -Force runtime | Out-Null
@{node=$nodePath;pnpm=$pnpmPath;python=$pythonPath} | ConvertTo-Json | Set-Content runtime/toolchain.json -Encoding utf8
& ./.venv-local/Scripts/python.exe scripts/check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Model assets missing or mismatched: consult docs/model-manifest.json' }
Write-Output 'Setup complete. Run ./scripts/start.ps1'
