param(
    [string]$Config = "configs/finetune_classifier_ds002739_subject_packed.yaml",
    [string]$ContrastiveCheckpoint = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "D:\anaconda3\envs\mamba\python.exe"

$args = @("run_finetune.py", "--config", $Config)
if (-not [string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    $args += @("--contrastive-checkpoint", $ContrastiveCheckpoint)
}

Write-Host "Running finetuning for ds002739 subject-packed dataset..."
if ([string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    Write-Host "No contrastive checkpoint provided. Finetuning will use random initialization unless modality-specific checkpoints are configured."
}
& $python @args