#!/usr/bin/env bash
set -euo pipefail

docker build -t pressure-gauge-ml:latest .
docker run --rm --gpus all --shm-size=8g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/reports:/app/reports" \
  pressure-gauge-ml:latest \
  python3 -m pressure_gauge_ml.cli --config configs/train.yaml "$@"
