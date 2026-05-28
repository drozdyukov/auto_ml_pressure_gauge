param(
    [string]$ModelPath = "runs\rtdetr\gauge_rtdetr_l_rtx3060\weights\best.pt"
)

$ErrorActionPreference = "Stop"

python -m pressure_gauge_ml.cli --config configs/train.yaml prepare
python -m pressure_gauge_ml.cli --config configs/train.yaml evaluate `
  --model $ModelPath `
  --data-yaml data\gauge_rtdetr_detection\holdout_data.yaml `
  --output reports\holdout_metrics.json
