param(
    [string]$ContrastiveConfig = "configs/train_contrastive_ds002739_subject_packed.yaml",
    [string]$FinetuneConfig = "configs/finetune_classifier_ds002739_subject_packed.yaml",
    [string]$ContrastiveCheckpoint = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "D:\anaconda3\envs\mamba\python.exe"

Write-Host "Running contrastive training for ds002739 subject-packed dataset..."
& $python run_train.py --config $ContrastiveConfig

if ([string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    $ContrastiveCheckpoint = "outputs/contrastive_ds002739_subject_packed/checkpoints/best.pth"
}

Write-Host "Running finetuning for ds002739 subject-packed dataset..."
& $python run_finetune.py --config $FinetuneConfig --contrastive-checkpoint $ContrastiveCheckpoint