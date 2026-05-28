param(
    [string]$ModelPath = "runs\rtdetr\gauge_rtdetr_l_rtx3060\weights\best.pt"
)

$ErrorActionPreference = "Stop"

python -m pressure_gauge_ml.cli --config configs/train.yaml prepare
python -m pressure_gauge_ml.cli --config configs/train.yaml infer-demo `
  --model $ModelPath `
  --output-dir runs\predict\holdout_demo `
  --predictions-json reports\holdout_predictions.json
