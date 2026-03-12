param(
    [string]$TrainConfig = "configs/train_ds002739.yaml",
    [string]$FinetuneConfig = "configs/finetune_ds002739.yaml",
    [string]$DataRoot = "cache/ds002739",
    [string]$LosoDir = "cache/ds002739/loso_subjectwise",
    [string]$OutputRoot = "outputs/ds002739",
    [switch]$TestOnly,
    [switch]$SkipContrastive,
    [switch]$SkipFinetune,
    [switch]$ForceCpu,
    [int]$TrainEpochs = 0,
    [int]$FinetuneEpochs = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1
)

function Get-MetricValue {
    param(
        [Parameter(Mandatory = $true)]$Metrics,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $property = $Metrics.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) {
        return 0.0
    }
    return [double]$property.Value
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Write-ContrastiveSummary {
    param(
        [Parameter(Mandatory = $true)][string]$ContrastiveRoot,
        [Parameter(Mandatory = $true)]$FoldNameMap
    )

    Write-Host "Summarizing LOSO contrastive metrics..."
    $summaryRows = @()
    foreach ($foldDir in (Get-ChildItem -Path $ContrastiveRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name)) {
        $displayFoldName = $FoldNameMap[$foldDir.Name]
        $metricsPath = Join-Path $foldDir.FullName "final_metrics.json"
        if (!(Test-Path $metricsPath)) {
            continue
        }

        $metrics = Read-JsonFile -Path $metricsPath
        $bestValMetrics = $metrics.best_val_metrics
        if ($null -eq $bestValMetrics) {
            $bestValMetrics = $metrics.last_val_metrics
        }
        $summaryRows += [pscustomobject]@{
            fold               = $displayFoldName
            fold_dir           = $foldDir.Name
            mean_r1            = Get-MetricValue -Metrics $bestValMetrics -Name "mean_r1"
            mean_r1_std        = Get-MetricValue -Metrics $bestValMetrics -Name "mean_r1_std"
            mean_r5            = Get-MetricValue -Metrics $bestValMetrics -Name "mean_r5"
            mean_r5_std        = Get-MetricValue -Metrics $bestValMetrics -Name "mean_r5_std"
            eeg_to_fmri_r1     = Get-MetricValue -Metrics $bestValMetrics -Name "eeg_to_fmri_r1"
            eeg_to_fmri_r5     = Get-MetricValue -Metrics $bestValMetrics -Name "eeg_to_fmri_r5"
            fmri_to_eeg_r1     = Get-MetricValue -Metrics $bestValMetrics -Name "fmri_to_eeg_r1"
            fmri_to_eeg_r5     = Get-MetricValue -Metrics $bestValMetrics -Name "fmri_to_eeg_r5"
            loss               = Get-MetricValue -Metrics $bestValMetrics -Name "loss"
            best_epoch         = Get-MetricValue -Metrics $metrics -Name "best_epoch"
            selection_mode     = [string]$metrics.selection_mode
        }
    }

    if ($summaryRows.Count -eq 0) {
        throw "No fold final_metrics.json files found under $ContrastiveRoot"
    }

    $meanR1Values = @($summaryRows | ForEach-Object { [double]$_.mean_r1 })
    $meanR5Values = @($summaryRows | ForEach-Object { [double]$_.mean_r5 })
    $eegR1Values = @($summaryRows | ForEach-Object { [double]$_.eeg_to_fmri_r1 })
    $eegR5Values = @($summaryRows | ForEach-Object { [double]$_.eeg_to_fmri_r5 })
    $fmriR1Values = @($summaryRows | ForEach-Object { [double]$_.fmri_to_eeg_r1 })
    $fmriR5Values = @($summaryRows | ForEach-Object { [double]$_.fmri_to_eeg_r5 })
    $lossValues = @($summaryRows | ForEach-Object { [double]$_.loss })

    $summaryRows += [pscustomobject]@{
        fold               = "CROSS_FOLD_MEAN_STD"
        fold_dir           = ""
        mean_r1            = Get-MeanValue -Values $meanR1Values
        mean_r1_std        = Get-StdValue -Values $meanR1Values
        mean_r5            = Get-MeanValue -Values $meanR5Values
        mean_r5_std        = Get-StdValue -Values $meanR5Values
        eeg_to_fmri_r1     = Get-MeanValue -Values $eegR1Values
        eeg_to_fmri_r5     = Get-MeanValue -Values $eegR5Values
        fmri_to_eeg_r1     = Get-MeanValue -Values $fmriR1Values
        fmri_to_eeg_r5     = Get-MeanValue -Values $fmriR5Values
        loss               = Get-MeanValue -Values $lossValues
        best_epoch         = 0
        selection_mode     = "val_mean_r1"
    }

    $summaryPath = Join-Path $ContrastiveRoot "loso_contrastive_summary.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    $summaryRows | Format-Table -AutoSize | Out-String | Write-Host

    $aggregate = $summaryRows | Where-Object { $_.fold -eq "CROSS_FOLD_MEAN_STD" } | Select-Object -First 1
    Write-Host "---"
    Write-Host ([string]::Format("Cross-fold contrastive mean_r1: {0:F4} ({1:F4})", $aggregate.mean_r1, $aggregate.mean_r1_std))
    Write-Host ([string]::Format("Cross-fold contrastive mean_r5: {0:F4} ({1:F4})", $aggregate.mean_r5, $aggregate.mean_r5_std))
    Write-Host ("Saved summary to " + $summaryPath)
}

function Write-FinetuneSummary {
    param(
        [Parameter(Mandatory = $true)][string]$FinetuneRoot,
        [Parameter(Mandatory = $true)]$FoldNameMap
    )

    Write-Host "Summarizing LOSO finetune metrics..."
    $summaryRows = @()
    foreach ($foldDir in (Get-ChildItem -Path $FinetuneRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name)) {
        $displayFoldName = $FoldNameMap[$foldDir.Name]
        $metricsPath = Join-Path $foldDir.FullName "test_metrics.json"
        if (!(Test-Path $metricsPath)) {
            continue
        }

        $metrics = Read-JsonFile -Path $metricsPath
        $summaryRows += [pscustomobject]@{
            fold         = $displayFoldName
            fold_dir     = $foldDir.Name
            accuracy     = Get-MetricValue -Metrics $metrics -Name "accuracy"
            accuracy_std = Get-MetricValue -Metrics $metrics -Name "accuracy_std"
            macro_f1     = Get-MetricValue -Metrics $metrics -Name "macro_f1"
            macro_f1_std = Get-MetricValue -Metrics $metrics -Name "macro_f1_std"
            loss         = Get-MetricValue -Metrics $metrics -Name "loss"
        }
    }

    if ($summaryRows.Count -eq 0) {
        throw "No fold test_metrics.json files found under $FinetuneRoot"
    }

    $accuracyValues = @($summaryRows | ForEach-Object { [double]$_.accuracy })
    $macroF1Values = @($summaryRows | ForEach-Object { [double]$_.macro_f1 })
    $lossValues = @($summaryRows | ForEach-Object { [double]$_.loss })

    $summaryRows += [pscustomobject]@{
        fold         = "CROSS_FOLD_MEAN_STD"
        fold_dir     = ""
        accuracy     = Get-MeanValue -Values $accuracyValues
        accuracy_std = Get-StdValue -Values $accuracyValues
        macro_f1     = Get-MeanValue -Values $macroF1Values
        macro_f1_std = Get-StdValue -Values $macroF1Values
        loss         = Get-MeanValue -Values $lossValues
    }

    $summaryPath = Join-Path $FinetuneRoot "loso_finetune_summary.csv"
    $summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
    $summaryRows | Format-Table -AutoSize | Out-String | Write-Host

    $aggregate = $summaryRows | Where-Object { $_.fold -eq "CROSS_FOLD_MEAN_STD" } | Select-Object -First 1
    Write-Host "---"
    Write-Host ([string]::Format("Cross-fold accuracy: {0:F4} ({1:F4})", $aggregate.accuracy, $aggregate.accuracy_std))
    Write-Host ([string]::Format("Cross-fold macro_f1: {0:F4} ({1:F4})", $aggregate.macro_f1, $aggregate.macro_f1_std))
    Write-Host ("Saved summary to " + $summaryPath)
}

function Get-MeanValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) {
        return 0.0
    }
    return [double](($Values | Measure-Object -Average).Average)
}

function Get-StdValue {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -le 1) {
        return 0.0
    }

    $mean = Get-MeanValue -Values $Values
    $sum = 0.0
    foreach ($value in $Values) {
        $sum += [math]::Pow(([double]$value - $mean), 2)
    }
    return [math]::Sqrt($sum / $Values.Count)
}

function Invoke-PythonOrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExe,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$StepName
    )

    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
$losoPath = Resolve-Path $LosoDir
$dataPath = Resolve-Path $DataRoot
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputRoot)) {
    [System.IO.Path]::GetFullPath($OutputRoot)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputRoot))
}
$contrastiveRoot = Join-Path $outputPath "contrastive"
$finetuneRoot = Join-Path $outputPath "finetune"

New-Item -ItemType Directory -Force -Path $contrastiveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $finetuneRoot | Out-Null

$foldDirs = Get-ChildItem -Path $losoPath -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name
if ($foldDirs.Count -eq 0) {
    throw "No LOSO fold directories found under $LosoDir"
}

if ($SkipContrastive -and $SkipFinetune) {
    throw "SkipContrastive and SkipFinetune cannot both be set"
}

if ($TestOnly -and $SkipFinetune) {
    throw "TestOnly requires the finetune stage and cannot be combined with SkipFinetune"
}

$foldNameMap = @{}
for ($foldIndex = 0; $foldIndex -lt $foldDirs.Count; $foldIndex++) {
    $foldNameMap[$foldDirs[$foldIndex].Name] = "fold$($foldIndex + 1)"
}

Write-Host "Running ds002739 LOSO cross-validation..."
Write-Host ("LOSO dir: " + $losoPath)
Write-Host ("Output root: " + $outputPath)
Write-Host ("Fold labels: " + (($foldDirs | ForEach-Object { $foldNameMap[$_.Name] + "=" + $_.Name }) -join ", "))
if ($TestOnly) {
    Write-Host "Mode: test-only"
}

foreach ($foldDir in $foldDirs) {
    $foldName = $foldDir.Name
    $displayFoldName = $foldNameMap[$foldName]
    $foldStart = Get-Date
    $trainManifest = Join-Path $foldDir.FullName "manifest_train.csv"
    $valManifest = Join-Path $foldDir.FullName "manifest_val.csv"
    $testManifest = Join-Path $foldDir.FullName "manifest_test.csv"
    $contrastiveOutputDir = Join-Path $contrastiveRoot $foldName
    $finetuneOutputDir = Join-Path $finetuneRoot $foldName
    $contrastiveCheckpoint = Join-Path $contrastiveOutputDir "checkpoints\best.pth"
    $finetuneCheckpoint = Join-Path $finetuneOutputDir "checkpoints\best.pth"

    if (!(Test-Path $trainManifest) -or !(Test-Path $valManifest) -or !(Test-Path $testManifest)) {
        throw "Missing LOSO manifest(s) under $($foldDir.FullName)"
    }

    if ($TestOnly) {
        if (!(Test-Path $finetuneCheckpoint)) {
            throw "Expected finetune checkpoint not found for test-only: $finetuneCheckpoint"
        }

        $finetuneArgs = @(
            "run_finetune.py",
            "--config", $FinetuneConfig,
            "--train-manifest", $trainManifest,
            "--val-manifest", $valManifest,
            "--test-manifest", $testManifest,
            "--root-dir", $dataPath,
            "--output-dir", $finetuneOutputDir,
            "--finetune-checkpoint", $finetuneCheckpoint,
            "--test-only"
        )
        if ($BatchSize -gt 0) {
            $finetuneArgs += "--batch-size"
            $finetuneArgs += $BatchSize.ToString()
        }
        if ($EvalBatchSize -gt 0) {
            $finetuneArgs += "--eval-batch-size"
            $finetuneArgs += $EvalBatchSize.ToString()
        }
        if ($NumWorkers -ge 0) {
            $finetuneArgs += "--num-workers"
            $finetuneArgs += $NumWorkers.ToString()
        }
        if ($ForceCpu) {
            $finetuneArgs += "--force-cpu"
        }

        Write-Host ("[" + $displayFoldName + "] test-only using finetune checkpoint: " + $finetuneCheckpoint)
        Invoke-PythonOrThrow -PythonExe $python -Args $finetuneArgs -StepName ("[" + $displayFoldName + "] test-only")
        $foldElapsed = (Get-Date) - $foldStart
        Write-Host ([string]::Format("[{0}] fold_time={1:N1}s", $displayFoldName, $foldElapsed.TotalSeconds))
        continue
    }

    if (-not $SkipContrastive) {
        $trainArgs = @(
            "run_train.py",
            "--config", $TrainConfig,
            "--train-manifest", $trainManifest,
            "--val-manifest", $valManifest,
            "--root-dir", $dataPath,
            "--output-dir", $contrastiveOutputDir
        )
        if ($TrainEpochs -gt 0) {
            $trainArgs += "--epochs"
            $trainArgs += $TrainEpochs.ToString()
        }
        if ($BatchSize -gt 0) {
            $trainArgs += "--batch-size"
            $trainArgs += $BatchSize.ToString()
        }
        if ($EvalBatchSize -gt 0) {
            $trainArgs += "--eval-batch-size"
            $trainArgs += $EvalBatchSize.ToString()
        }
        if ($NumWorkers -ge 0) {
            $trainArgs += "--num-workers"
            $trainArgs += $NumWorkers.ToString()
        }
        if ($ForceCpu) {
            $trainArgs += "--force-cpu"
        }

        Write-Host ("[" + $displayFoldName + "] contrastive")
        Invoke-PythonOrThrow -PythonExe $python -Args $trainArgs -StepName ("[" + $displayFoldName + "] contrastive")
    }

    $useContrastiveCheckpoint = Test-Path $contrastiveCheckpoint
    if ((-not $SkipContrastive) -and (-not $useContrastiveCheckpoint)) {
        throw "Expected contrastive checkpoint not found: $contrastiveCheckpoint"
    }

    if (-not $SkipFinetune) {
        $finetuneArgs = @(
            "run_finetune.py",
            "--config", $FinetuneConfig,
            "--train-manifest", $trainManifest,
            "--val-manifest", $valManifest,
            "--test-manifest", $testManifest,
            "--root-dir", $dataPath,
            "--output-dir", $finetuneOutputDir
        )
        if ($useContrastiveCheckpoint) {
            $finetuneArgs += @("--contrastive-checkpoint", $contrastiveCheckpoint)
        }
        $effectiveFinetuneEpochs = $FinetuneEpochs
        if ($effectiveFinetuneEpochs -le 0 -and $TrainEpochs -gt 0) {
            $effectiveFinetuneEpochs = $TrainEpochs
        }
        if ($effectiveFinetuneEpochs -gt 0) {
            $finetuneArgs += "--epochs"
            $finetuneArgs += $effectiveFinetuneEpochs.ToString()
        }
        if ($BatchSize -gt 0) {
            $finetuneArgs += "--batch-size"
            $finetuneArgs += $BatchSize.ToString()
        }
        if ($EvalBatchSize -gt 0) {
            $finetuneArgs += "--eval-batch-size"
            $finetuneArgs += $EvalBatchSize.ToString()
        }
        if ($NumWorkers -ge 0) {
            $finetuneArgs += "--num-workers"
            $finetuneArgs += $NumWorkers.ToString()
        }
        if ($ForceCpu) {
            $finetuneArgs += "--force-cpu"
        }

        if ($useContrastiveCheckpoint) {
            Write-Host ("[" + $displayFoldName + "] using contrastive checkpoint: " + $contrastiveCheckpoint)
        }
        elseif ($SkipContrastive) {
            Write-Host ("[" + $displayFoldName + "] no contrastive checkpoint found; finetuning from random initialization")
        }

        Write-Host ("[" + $displayFoldName + "] finetune")
        Invoke-PythonOrThrow -PythonExe $python -Args $finetuneArgs -StepName ("[" + $displayFoldName + "] finetune")
    }

    $foldElapsed = (Get-Date) - $foldStart
    Write-Host ([string]::Format("[{0}] fold_time={1:N1}s", $displayFoldName, $foldElapsed.TotalSeconds))
}

if (-not $SkipContrastive) {
    Write-ContrastiveSummary -ContrastiveRoot $contrastiveRoot -FoldNameMap $foldNameMap
}

if (-not $SkipFinetune) {
    Write-FinetuneSummary -FinetuneRoot $finetuneRoot -FoldNameMap $foldNameMap
}