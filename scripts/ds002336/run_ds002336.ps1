param(
    [string]$TrainConfig = "configs/train_ds002336.yaml",
    [string]$FinetuneConfig = "configs/finetune_ds002336.yaml",
    [string]$DataRoot = "cache/ds002336",
    [string]$LosoDir = "cache/ds002336/loso_subjectwise",
    [string]$OutputRoot = "outputs/ds002336",
    [switch]$SkipContrastive,
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

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$python = "python"
$losoPath = Resolve-Path $LosoDir
$dataPath = Resolve-Path $DataRoot
$outputPath = Join-Path (Get-Location) $OutputRoot
$contrastiveRoot = Join-Path $outputPath "contrastive"
$finetuneRoot = Join-Path $outputPath "finetune"

New-Item -ItemType Directory -Force -Path $contrastiveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $finetuneRoot | Out-Null

$foldDirs = Get-ChildItem -Path $losoPath -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name
if ($foldDirs.Count -eq 0) {
    throw "No LOSO fold directories found under $LosoDir"
}

Write-Host "Running ds002336 LOSO cross-validation..."
Write-Host ("LOSO dir: " + $losoPath)
Write-Host ("Output root: " + $outputPath)

foreach ($foldDir in $foldDirs) {
    $foldName = $foldDir.Name
    $trainManifest = Join-Path $foldDir.FullName "manifest_train.csv"
    $valManifest = Join-Path $foldDir.FullName "manifest_val.csv"
    $testManifest = Join-Path $foldDir.FullName "manifest_test.csv"
    $contrastiveOutputDir = Join-Path $contrastiveRoot $foldName
    $finetuneOutputDir = Join-Path $finetuneRoot $foldName
    $contrastiveCheckpoint = Join-Path $contrastiveOutputDir "checkpoints\best.pth"

    if (!(Test-Path $trainManifest) -or !(Test-Path $valManifest) -or !(Test-Path $testManifest)) {
        throw "Missing LOSO manifest(s) under $($foldDir.FullName)"
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

        Write-Host ("[" + $foldName + "] contrastive")
        & $python @trainArgs
    }

    if (!(Test-Path $contrastiveCheckpoint)) {
        throw "Expected contrastive checkpoint not found: $contrastiveCheckpoint"
    }

    $finetuneArgs = @(
        "run_finetune.py",
        "--config", $FinetuneConfig,
        "--train-manifest", $trainManifest,
        "--val-manifest", $valManifest,
        "--test-manifest", $testManifest,
        "--root-dir", $dataPath,
        "--output-dir", $finetuneOutputDir,
        "--contrastive-checkpoint", $contrastiveCheckpoint
    )
    if ($FinetuneEpochs -gt 0) {
        $finetuneArgs += "--epochs"
        $finetuneArgs += $FinetuneEpochs.ToString()
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

    Write-Host ("[" + $foldName + "] finetune")
    & $python @finetuneArgs
}

Write-Host "Summarizing LOSO finetune metrics..."
$summaryRows = @()
foreach ($foldDir in (Get-ChildItem -Path $finetuneRoot -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name)) {
    $metricsPath = Join-Path $foldDir.FullName "test_metrics.json"
    if (!(Test-Path $metricsPath)) {
        continue
    }

    $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
    $summaryRows += [pscustomobject]@{
        fold         = $foldDir.Name
        accuracy     = Get-MetricValue -Metrics $metrics -Name "accuracy"
        accuracy_std = Get-MetricValue -Metrics $metrics -Name "accuracy_std"
        macro_f1     = Get-MetricValue -Metrics $metrics -Name "macro_f1"
        macro_f1_std = Get-MetricValue -Metrics $metrics -Name "macro_f1_std"
        loss         = Get-MetricValue -Metrics $metrics -Name "loss"
    }
}

if ($summaryRows.Count -eq 0) {
    throw "No fold test_metrics.json files found under $finetuneRoot"
}

$accuracyValues = @($summaryRows | ForEach-Object { [double]$_.accuracy })
$macroF1Values = @($summaryRows | ForEach-Object { [double]$_.macro_f1 })
$lossValues = @($summaryRows | ForEach-Object { [double]$_.loss })

$summaryRows += [pscustomobject]@{
    fold         = "CROSS_FOLD_MEAN_STD"
    accuracy     = Get-MeanValue -Values $accuracyValues
    accuracy_std = Get-StdValue -Values $accuracyValues
    macro_f1     = Get-MeanValue -Values $macroF1Values
    macro_f1_std = Get-StdValue -Values $macroF1Values
    loss         = Get-MeanValue -Values $lossValues
}

$summaryPath = Join-Path $finetuneRoot "loso_finetune_summary.csv"
$summaryRows | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8
$summaryRows | Format-Table -AutoSize | Out-String | Write-Host

$aggregate = $summaryRows | Where-Object { $_.fold -eq "CROSS_FOLD_MEAN_STD" } | Select-Object -First 1
Write-Host "---"
Write-Host ([string]::Format("Cross-fold accuracy: {0:F4} ± {1:F4}", $aggregate.accuracy, $aggregate.accuracy_std))
Write-Host ([string]::Format("Cross-fold macro_f1: {0:F4} ± {1:F4}", $aggregate.macro_f1, $aggregate.macro_f1_std))
Write-Host ("Saved summary to " + $summaryPath)