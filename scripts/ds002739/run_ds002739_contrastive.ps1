param(
    [string]$Config = "configs/train_ds002739.yaml"
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

Write-Host "Running contrastive training for ds002739 subject-packed dataset..."
& $python run_train.py --config $Config
