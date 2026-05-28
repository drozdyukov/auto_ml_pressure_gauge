param(
    [string]$ArchivePath = "C:\Users\drozd\Downloads\guage_read.v1i.coco.zip",
    [string]$OutputDir = "data\guage_read_coco"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ArchivePath)) {
    throw "Archive not found: $ArchivePath"
}

if (Test-Path $OutputDir) {
    Write-Host "Dataset directory already exists: $OutputDir"
    Write-Host "Remove it first if you need to re-import the archive."
    exit 0
}

New-Item -ItemType Directory -Force -Path (Split-Path $OutputDir) | Out-Null
Expand-Archive -LiteralPath $ArchivePath -DestinationPath $OutputDir
Write-Host "Dataset extracted to $OutputDir"
