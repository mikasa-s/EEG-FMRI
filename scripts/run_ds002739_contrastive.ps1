param(
    [string]$Config = "configs/train_contrastive_ds002739_subject_packed.yaml"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "D:\anaconda3\envs\mamba\python.exe"

Write-Host "Running contrastive training for ds002739 subject-packed dataset..."
& $python run_train.py --config $Config
