param(
    [string]$DsRoot = "D:\OpenNeuro\ds002739",
    [string]$OutputRoot = "cache/ds002739_subject_packed_eeg2_fmri6",
    [string[]]$Subjects = @(),
    [string[]]$Runs = @(),
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "subject",
    [int]$TrainSubjects = 7,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [double]$EegWindowSec = 2.0,
    [double]$FmriWindowSec = 6.0
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

$python = "D:\anaconda3\envs\mamba\python.exe"

$args = @(
    "preprocess/prepare_ds002739.py",
    "--ds-root", $DsRoot,
    "--output-root", $OutputRoot,
    "--fmri-mode", "volume",
    "--pack-subject-files",
    "--eeg-window-sec", $EegWindowSec.ToString(),
    "--fmri-window-sec", $FmriWindowSec.ToString(),
    "--eeg-seq-len", "2",
    "--eeg-patch-len", "200",
    "--eeg-target-sfreq", "200",
    "--eeg-lfreq", "0.5",
    "--eeg-hfreq", "40",
    "--tr", "2.0",
    "--fmri-max-shape", "48", "48", "48",
    "--split-mode", $SplitMode,
    "--train-subjects", $TrainSubjects.ToString(),
    "--val-subjects", $ValSubjects.ToString(),
    "--test-subjects", $TestSubjects.ToString()
)

if ($Subjects.Count -gt 0) {
    $args += "--subjects"
    $args += $Subjects
}

if ($Runs.Count -gt 0) {
    $args += "--runs"
    $args += $Runs
}

if ($Subjects.Count -gt 0) {
    Write-Host ("Preparing ds002739 for subjects: " + ($Subjects -join ", "))
}
else {
    Write-Host "Preparing ds002739 for all subjects..."
}

if ($Runs.Count -gt 0) {
    Write-Host ("Restricting to runs: " + ($Runs -join ", "))
}

Write-Host ("Output root: " + $OutputRoot)
Write-Host ("Split mode: " + $SplitMode)
& $python @args
