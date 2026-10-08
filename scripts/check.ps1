$projectRoot = Split-Path $PSScriptRoot -Parent
& (Join-Path $projectRoot '.venv-local/Scripts/python.exe') (Join-Path $PSScriptRoot 'check_environment.py')
exit $LASTEXITCODE
