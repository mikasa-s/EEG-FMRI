param(
    [string]$DatasetRoot = "cache/ds002739_subject_packed_eeg2_fmri6",
    [int]$TargetChannels = 0
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "D:\anaconda3\envs\mamba\python.exe"

$args = @(
    "preprocess/align_ds002739_eeg_channels.py",
    "--dataset-root", $DatasetRoot
)

if ($TargetChannels -gt 0) {
    $args += @("--target-channels", $TargetChannels.ToString())
}

Write-Host ("Aligning ds002739 EEG channels under: " + $DatasetRoot)
& $python @args