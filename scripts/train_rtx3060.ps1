$ErrorActionPreference = "Stop"

python -m pressure_gauge_ml.cli --config configs/train.yaml validate-data
python -m pressure_gauge_ml.cli --config configs/train.yaml prepare
python -m pressure_gauge_ml.cli --config configs/train.yaml monitor --output reports/monitoring_snapshot.json
python -m pressure_gauge_ml.cli --config configs/train.yaml train
