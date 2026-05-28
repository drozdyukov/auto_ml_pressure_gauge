#!/usr/bin/env bash
set -euo pipefail

scripts/docker_run.sh prepare
scripts/docker_run.sh monitor --output reports/monitoring_snapshot.json
scripts/docker_run.sh train
