$ErrorActionPreference = "Stop"

.\scripts\docker_run.ps1 prepare
.\scripts\docker_run.ps1 monitor --output reports/monitoring_snapshot.json
.\scripts\docker_run.ps1 train
