$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$recordPath = Join-Path $projectRoot 'runtime/processes.json'
if (!(Test-Path -LiteralPath $recordPath)) { Write-Output 'No managed processes.'; exit 0 }
$records = @(Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json)
foreach ($record in $records) {
  $process = Get-Process -Id $record.id -ErrorAction SilentlyContinue
  if ($process) {
    $expected = ([datetime]$record.start).ToUniversalTime()
    if ([Math]::Abs(($process.StartTime.ToUniversalTime()-$expected).TotalSeconds) -gt 1) { throw "PID $($record.id) was reused; refusing to stop it." }
    & taskkill.exe /PID $record.id /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to stop managed process $($record.id)" }
  }
}
Remove-Item -LiteralPath $recordPath
Write-Output 'RoadClear stopped. Outputs remain in runtime.'
