param(
    [string]$Config = "configs/finetune_ds002336.yaml",
    [string]$ContrastiveCheckpoint = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

$commandLine = @("run_finetune.py", "--config", $Config)
if (-not [string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    $commandLine += @("--contrastive-checkpoint", $ContrastiveCheckpoint)
}

Write-Host "Running finetuning for ds002336 binary block dataset..."
if ([string]::IsNullOrWhiteSpace($ContrastiveCheckpoint)) {
    Write-Host "No contrastive checkpoint provided. Finetuning will use random initialization unless checkpoints are configured in YAML."
}
& $python @commandLine