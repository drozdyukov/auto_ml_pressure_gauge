param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command = @("summary")
)

$ErrorActionPreference = "Stop"

$Docker = "docker"
if (-not (Get-Command $Docker -ErrorAction SilentlyContinue)) {
  $Docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
}
if (-not (Test-Path $Docker) -and -not (Get-Command $Docker -ErrorAction SilentlyContinue)) {
  throw "Docker CLI not found. Add Docker Desktop to PATH or install Docker Desktop."
}

& $Docker build -t pressure-gauge-ml:latest .
& $Docker run --rm --gpus all --shm-size=8g `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/runs:/app/runs" `
  -v "${PWD}/reports:/app/reports" `
  pressure-gauge-ml:latest `
  python3 -m pressure_gauge_ml.cli --config configs/train.yaml @Command
