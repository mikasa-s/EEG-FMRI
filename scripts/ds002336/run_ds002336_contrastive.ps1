param(
    [string]$Config = "configs/train_ds002336.yaml"
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

Write-Host "Running contrastive training for ds002336 binary block dataset..."
& $python run_train.py --config $Config