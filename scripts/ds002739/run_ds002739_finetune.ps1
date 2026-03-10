param(
    [string]$Config = "configs/finetune_ds002739.yaml",
    [string]$ContrastiveCheckpoint = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

Write-Host "Running finetuning for ds002739 subject-packed dataset..."
if ([string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    Write-Host "No contrastive checkpoint provided. Finetuning will use random initialization unless modality-specific checkpoints are configured."
    & $python run_finetune.py --config $Config
}
else {
    & $python run_finetune.py --config $Config --contrastive-checkpoint $ContrastiveCheckpoint
}