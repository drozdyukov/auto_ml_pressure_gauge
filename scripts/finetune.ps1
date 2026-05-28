param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$DatasetRoot = "data\guage_read_coco",
    [string]$PreparedRoot = "data\gauge_rtdetr_finetune",
    [int]$Epochs = 20,
    [int]$Batch = 4,
    [string]$Name = "finetune_gauge_rtdetr"
)

$ErrorActionPreference = "Stop"

python -m pressure_gauge_ml.cli --config configs/train.yaml `
  --dataset-root $DatasetRoot `
  --prepared-root $PreparedRoot `
  --epochs $Epochs `
  --batch $Batch `
  --name $Name `
  finetune --model $ModelPath
