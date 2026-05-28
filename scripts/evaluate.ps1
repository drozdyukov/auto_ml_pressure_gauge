param(
    [string]$ModelPath = "runs\rtdetr\gauge_rtdetr_l_rtx3060\weights\best.pt",
    [string]$Output = "reports\test_metrics.json"
)

$ErrorActionPreference = "Stop"

python -m pressure_gauge_ml.cli --config configs/train.yaml prepare
python -m pressure_gauge_ml.cli --config configs/train.yaml evaluate --model $ModelPath --output $Output
