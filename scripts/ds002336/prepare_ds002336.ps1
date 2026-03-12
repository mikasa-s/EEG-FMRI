param(
    [string]$DsRoot = "../ds002336",
    [string]$OutputRoot = "cache/ds002336",
    [string[]]$Subjects = @(),
    [string[]]$Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"),
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "loso",
    [int]$TrainSubjects = 7,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [int]$EegSeqLen = 20,
    [int]$EegPatchLen = 200,
    [bool]$DropEcg = $true,
    [bool]$TrainingReady = $true,
    [ValidateSet("raw", "spm_unsmoothed", "spm_smoothed")]
    [string]$FmriSource = "spm_smoothed"
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
    "preprocess/prepare_ds002336.py",
    "--ds-root", $DsRoot,
    "--output-root", $OutputRoot,
    "--sample-mode", "block",
    "--label-mode", "binary_rest_task",
    "--fmri-mode", "volume",
    "--pack-subject-files",
    "--eeg-mode", "patched",
    "--eeg-seq-len", $EegSeqLen.ToString(),
    "--eeg-patch-len", $EegPatchLen.ToString(),
    "--tr", "2.0",
    "--fmri-max-shape", "48", "48", "48",
    "--split-mode", $SplitMode,
    "--train-subjects", $TrainSubjects.ToString(),
    "--val-subjects", $ValSubjects.ToString(),
    "--test-subjects", $TestSubjects.ToString()
)

if ($FmriSource -ne "raw") {
    $cliArgs += @(
        "--fmri-source", $FmriSource,
        "--discard-initial-trs", "0",
        "--protocol-offset-sec", "0.0"
    )
}
else {
    $cliArgs += @(
        "--discard-initial-trs", "1",
        "--protocol-offset-sec", "2.0"
    )
}

$cliArgs += "--tasks"
$cliArgs += $Tasks

if ($DropEcg) {
    $cliArgs += "--drop-ecg"
}

if ($TrainingReady) {
    $cliArgs += "--training-ready"
}
else {
    $cliArgs += "--no-training-ready"
}

if ($Subjects.Count -gt 0) {
    $cliArgs += "--subjects"
    $cliArgs += $Subjects
}

Write-Host "Preparing ds002336 dataset..."
Write-Host ("Output root: " + $OutputRoot)
Write-Host "Packing one subject directory of NPY arrays per subject."
Write-Host ("Tasks: " + ($Tasks -join ", "))
Write-Host ("Split mode: " + $SplitMode)
Write-Host ("fMRI source: " + $FmriSource)
Write-Host ("Training ready: " + $TrainingReady)
if ($Subjects.Count -gt 0) {
    Write-Host ("Subjects: " + ($Subjects -join ", "))
}

& $python @cliArgs