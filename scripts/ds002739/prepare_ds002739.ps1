param(
    [string]$DsRoot = "../ds002739",
    [string]$OutputRoot = "cache/ds002739",
    [string[]]$Subjects = @(),
    [string[]]$Runs = @(),
    [int]$NumWorkers = 2,
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "loso",
    [int]$TrainSubjects = 21,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [double]$EegWindowSec = 2.0,
    [double]$FmriWindowSec = 6.0,
    [bool]$TrainingReady = $true
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"

if ($SplitMode -ne "none" -and $Subjects.Count -gt 0) {
    $requiredSubjects = if ($SplitMode -eq "loso") { $ValSubjects + 1 } else { $TrainSubjects + $ValSubjects + $TestSubjects }
    if ($Subjects.Count -lt $requiredSubjects) {
        Write-Warning "Provided subject subset is smaller than the requested split sizes; disabling split generation for this run."
        $SplitMode = "none"
    }
}

$cliArgs = @(
    "preprocess/prepare_ds002739.py",
    "--ds-root", $DsRoot,
    "--output-root", $OutputRoot,
    "--fmri-mode", "volume",
    "--fmri-voxel-size", "2.0", "2.0", "2.0",
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
    "--num-workers", $NumWorkers.ToString(),
    "--split-mode", $SplitMode,
    "--train-subjects", $TrainSubjects.ToString(),
    "--val-subjects", $ValSubjects.ToString(),
    "--test-subjects", $TestSubjects.ToString()
)

if ($Subjects.Count -gt 0) {
    $cliArgs += "--subjects"
    $cliArgs += $Subjects
}

if ($Runs.Count -gt 0) {
    $cliArgs += "--runs"
    $cliArgs += $Runs
}

if ($TrainingReady) {
    $cliArgs += "--training-ready"
}
else {
    $cliArgs += "--no-training-ready"
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
Write-Host ("Worker processes: " + $NumWorkers)
Write-Host ("Split mode: " + $SplitMode)
Write-Host ("Training ready: " + $TrainingReady)
Write-Host "fMRI preprocessing: resample to 2.0x2.0x2.0 mm, then center-crop to 48x48x48"
& $python @cliArgs