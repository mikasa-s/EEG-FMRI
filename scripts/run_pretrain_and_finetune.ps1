param(
    [string]$PretrainDatasets = "ds002336,ds002338,ds002739",
    [ValidateSet("ds002336", "ds002338", "ds002739")]
    [string]$TargetDataset = "ds002739",
    [string]$JointTrainConfig = "configs/train_joint_contrastive.yaml",
    [string]$Ds002336FinetuneConfig = "configs/finetune_ds002336.yaml",
    [string]$Ds002338FinetuneConfig = "configs/finetune_ds002338.yaml",
    [string]$Ds002739FinetuneConfig = "configs/finetune_ds002739.yaml",
    [string]$Ds002336Root = "../ds002336",
    [string]$Ds002338Root = "../ds002338",
    [string]$Ds002739Root = "../ds002739",
    [string]$JointCacheRoot = "cache/joint_contrastive",
    [string]$Ds002336CacheRoot = "cache/ds002336",
    [string]$Ds002338CacheRoot = "cache/ds002338",
    [string]$Ds002739CacheRoot = "cache/ds002739",
    [string]$JointOutputRoot = "outputs/joint_contrastive",
    [string]$Ds002336OutputRoot = "outputs/ds002336",
    [string]$Ds002338OutputRoot = "outputs/ds002338",
    [string]$Ds002739OutputRoot = "outputs/ds002739",
    [string]$PretrainedWeightsDir = "pretrained_weights",
    [double]$JointEegWindowSec = 8.0,
    [int]$PretrainEpochs = 0,
    [int]$FinetuneEpochs = 0,
    [int]$PretrainBatchSize = 0,
    [int]$FinetuneBatchSize = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [int]$GpuCount = 1,
    [string]$GpuIds = "",
    [switch]$SkipPretrain,
    [switch]$SkipFinetune,
    [switch]$TestOnly,
    [switch]$ForceCpu,
    [string]$PythonExe = ""
)

function Invoke-CommandOrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Args,
        [Parameter(Mandatory = $true)][string]$StepName
    )

    & $Executable @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Test-JointCacheReady {
    param([Parameter(Mandatory = $true)][string]$CacheRoot)

    $manifestPath = Join-Path $CacheRoot "manifest_all.csv"
    return (Test-Path $manifestPath)
}

function Test-TargetCacheReady {
    param([Parameter(Mandatory = $true)][string]$CacheRoot)

    $losoRoot = Join-Path $CacheRoot "loso_subjectwise"
    if (!(Test-Path $losoRoot)) {
        return $false
    }

    $foldDirs = Get-ChildItem -Path $losoRoot -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "fold_*" }
    if ($null -eq $foldDirs -or $foldDirs.Count -eq 0) {
        return $false
    }

    foreach ($foldDir in $foldDirs) {
        $trainManifest = Join-Path $foldDir.FullName "manifest_train.csv"
        $valManifest = Join-Path $foldDir.FullName "manifest_val.csv"
        $testManifest = Join-Path $foldDir.FullName "manifest_test.csv"
        if (!(Test-Path $trainManifest) -or !(Test-Path $valManifest) -or !(Test-Path $testManifest)) {
            return $false
        }
    }

    return $true
}

function Test-ConfigUsesEegBaseline {
    param([Parameter(Mandatory = $true)][string]$ConfigPath)

    $config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Yaml
    if ($null -eq $config -or $null -eq $config.finetune -or $null -eq $config.finetune.eeg_baseline) {
        return $false
    }
    return [bool]$config.finetune.eeg_baseline.enabled
}

function Resolve-DatasetRoot {
    param(
        [Parameter(Mandatory = $true)][string]$CurrentRoot,
        [Parameter(Mandatory = $true)][string]$DatasetName,
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    if (Test-Path $CurrentRoot) {
        return $CurrentRoot
    }

    $candidates = @(
        (Join-Path $RepoRoot ("data/" + $DatasetName)),
        (Join-Path $RepoRoot ("../data/" + $DatasetName)),
        (Join-Path $RepoRoot ("../" + $DatasetName))
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    return $CurrentRoot
}

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

$Ds002336Root = Resolve-DatasetRoot -CurrentRoot $Ds002336Root -DatasetName "ds002336" -RepoRoot $repoRoot
$Ds002338Root = Resolve-DatasetRoot -CurrentRoot $Ds002338Root -DatasetName "ds002338" -RepoRoot $repoRoot
$Ds002739Root = Resolve-DatasetRoot -CurrentRoot $Ds002739Root -DatasetName "ds002739" -RepoRoot $repoRoot

$allowedPretrainDatasets = @("ds002336", "ds002338", "ds002739")
$pretrainDatasetList = @($PretrainDatasets.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($pretrainDatasetList.Count -eq 0) {
    $pretrainDatasetList = @("ds002336", "ds002338", "ds002739")
}
$invalidPretrain = @($pretrainDatasetList | Where-Object { $_ -notin $allowedPretrainDatasets })
if ($invalidPretrain.Count -gt 0) {
    throw "Invalid PretrainDatasets values: $($invalidPretrain -join ', '). Allowed: $($allowedPretrainDatasets -join ', ')"
}

if ($SkipPretrain -and $SkipFinetune) {
    throw "SkipPretrain and SkipFinetune cannot both be set"
}

if ($TestOnly -and $SkipFinetune) {
    throw "TestOnly requires the finetune stage and cannot be combined with SkipFinetune"
}
if ($GpuCount -le 0) {
    throw "GpuCount must be >= 1"
}

$gpuIdList = @($GpuIds.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($gpuIdList.Count -gt 0 -and $gpuIdList.Count -lt $GpuCount) {
    throw "GpuIds count ($($gpuIdList.Count)) must be >= GpuCount ($GpuCount)"
}
if ($gpuIdList.Count -gt 0) {
    $env:CUDA_VISIBLE_DEVICES = ($gpuIdList -join ',')
    Write-Host ("Using CUDA_VISIBLE_DEVICES=" + $env:CUDA_VISIBLE_DEVICES)
}

$useMultiGpu = (-not $ForceCpu) -and ($GpuCount -gt 1)

if ($PretrainBatchSize -le 0 -and $BatchSize -gt 0) {
    $PretrainBatchSize = $BatchSize
}
if ($FinetuneBatchSize -le 0 -and $BatchSize -gt 0) {
    $FinetuneBatchSize = $BatchSize
}

if ($PythonExe.Trim()) {
    $python = $PythonExe
}
elseif ($env:CONDA_PREFIX) {
    $python = Join-Path $env:CONDA_PREFIX "python.exe"
}
else {
    $python = "python"
}
$jointManifestPath = Join-Path $JointCacheRoot "manifest_all.csv"
$jointChannelManifest = Join-Path $JointCacheRoot "eeg_channels_target.csv"
$jointTrainingCheckpointPath = Join-Path $JointOutputRoot "contrastive\checkpoints\best.pth"
$jointCheckpointPath = Join-Path $PretrainedWeightsDir "contrastive_best.pth"
$jointCheckpointSourcePath = ""

if (!$SkipPretrain) {
    if (Test-JointCacheReady -CacheRoot $JointCacheRoot) {
        Write-Host ("Found existing joint cache under " + $JointCacheRoot + "; skipping preprocessing.")
    }
    else {
        $jointPrepareArgs = @(
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $repoRoot "scripts/prepare_joint_contrastive.ps1"),
            "-OutputRoot", $JointCacheRoot,
            "-EegWindowSec", $JointEegWindowSec.ToString(),
            "-PythonExe", $python
        )
        if ($pretrainDatasetList.Count -gt 0) {
            $jointPrepareArgs += "-Datasets"
            $jointPrepareArgs += ($pretrainDatasetList -join ",")
        }
        if ($NumWorkers -ge 1) {
            $jointPrepareArgs += @("-NumWorkers", $NumWorkers.ToString())
        }
        $jointPrepareArgs += @("-Ds002336Root", $Ds002336Root, "-Ds002338Root", $Ds002338Root, "-Ds002739Root", $Ds002739Root)

        Write-Host "Preparing joint contrastive cache..."
        Invoke-CommandOrThrow -Executable "powershell" -Args $jointPrepareArgs -StepName "joint preprocessing"
    }

    $trainScript = (Join-Path $repoRoot "run_train.py")
    $trainArgs = @(
        "--config", $JointTrainConfig,
        "--manifest", $jointManifestPath,
        "--root-dir", $JointCacheRoot,
        "--output-dir", (Join-Path $JointOutputRoot "contrastive")
    )
    if ($PretrainEpochs -gt 0) {
        $trainArgs += @("--epochs", $PretrainEpochs.ToString())
    }
    if ($PretrainBatchSize -gt 0) {
        $trainArgs += @("--batch-size", $PretrainBatchSize.ToString())
    }
    if ($NumWorkers -ge 0) {
        $trainArgs += @("--num-workers", $NumWorkers.ToString())
    }
    if ($ForceCpu) {
        $trainArgs += "--force-cpu"
    }
    elseif ($GpuCount -gt 0) {
        $trainArgs += @("--set", "train.gpu_count=$GpuCount")
        if ($gpuIdList.Count -gt 0) {
            $trainArgs += @("--set", "train.gpu_ids=$($gpuIdList -join ',')")
        }
    }

    Write-Host "Running joint contrastive pretraining..."
    if ($useMultiGpu) {
        $ddpTrainArgs = @("-m", "torch.distributed.run", "--nproc_per_node", $GpuCount.ToString(), $trainScript) + $trainArgs
        Invoke-CommandOrThrow -Executable $python -Args $ddpTrainArgs -StepName "joint pretraining"
    }
    else {
        Invoke-CommandOrThrow -Executable $python -Args (@($trainScript) + $trainArgs) -StepName "joint pretraining"
    }

    if (!(Test-Path $jointTrainingCheckpointPath)) {
        throw "Pretrain checkpoint not found after pretraining: $jointTrainingCheckpointPath"
    }
    $jointCheckpointParent = Split-Path -Parent $jointCheckpointPath
    if ($jointCheckpointParent) {
        New-Item -ItemType Directory -Force -Path $jointCheckpointParent | Out-Null
    }
    Copy-Item -Path $jointTrainingCheckpointPath -Destination $jointCheckpointPath -Force
    Write-Host ("Synced pretrain best checkpoint to: " + $jointCheckpointPath)
}

if (Test-Path $jointCheckpointPath) {
    $jointCheckpointSourcePath = $jointCheckpointPath
}
elseif (Test-Path $jointTrainingCheckpointPath) {
    $jointCheckpointSourcePath = $jointTrainingCheckpointPath
    Write-Host ("Using fallback pretrain checkpoint path: " + $jointCheckpointSourcePath)
}

if ($SkipFinetune) {
    return
}

$targetPrepareScript = if ($TargetDataset -eq "ds002336" -or $TargetDataset -eq "ds002338") { "scripts/ds00233x/prepare_ds00233x.ps1" } else { "scripts/ds002739/prepare_ds002739.ps1" }
$targetFinetuneConfig = if ($TargetDataset -eq "ds002336") { $Ds002336FinetuneConfig } elseif ($TargetDataset -eq "ds002338") { $Ds002338FinetuneConfig } else { $Ds002739FinetuneConfig }
$targetCacheRoot = if ($TargetDataset -eq "ds002336") { $Ds002336CacheRoot } elseif ($TargetDataset -eq "ds002338") { $Ds002338CacheRoot } else { $Ds002739CacheRoot }
$targetOutputRoot = if ($TargetDataset -eq "ds002336") { $Ds002336OutputRoot } elseif ($TargetDataset -eq "ds002338") { $Ds002338OutputRoot } else { $Ds002739OutputRoot }

if (Test-TargetCacheReady -CacheRoot $targetCacheRoot) {
    Write-Host ("Found existing target cache under " + $targetCacheRoot + "; skipping preprocessing.")
}
else {
    $targetPrepareArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $repoRoot $targetPrepareScript),
        "-OutputRoot", $targetCacheRoot,
        "-SplitMode", "loso",
        "-TrainingReady",
        "-EegOnly",
        "-PythonExe", $python
    )
    if (Test-Path $jointChannelManifest) {
        $targetPrepareArgs += @("-TargetChannelManifest", $jointChannelManifest)
    }
    if ($TargetDataset -eq "ds002336") {
        $targetPrepareArgs += @("-DatasetName", "ds002336", "-DsRoot", $Ds002336Root)
    }
    elseif ($TargetDataset -eq "ds002338") {
        $targetPrepareArgs += @("-DatasetName", "ds002338", "-DsRoot", $Ds002338Root)
    }
    else {
        $targetPrepareArgs += @("-DsRoot", $Ds002739Root)
    }

    Write-Host ("Preparing target finetune cache for " + $TargetDataset + "...")
    Invoke-CommandOrThrow -Executable "powershell" -Args $targetPrepareArgs -StepName "target preprocessing"
}

$losoDir = Join-Path $targetCacheRoot "loso_subjectwise"
$foldDirs = Get-ChildItem -Path $losoDir -Directory | Where-Object { $_.Name -like "fold_*" } | Sort-Object Name
if ($foldDirs.Count -eq 0) {
    throw "No LOSO fold directories found under $losoDir"
}

$finetuneRoot = Join-Path $targetOutputRoot "finetune"
New-Item -ItemType Directory -Force -Path $finetuneRoot | Out-Null

$baselineEnabled = Test-ConfigUsesEegBaseline -ConfigPath $targetFinetuneConfig

if ($jointCheckpointSourcePath -and -not $baselineEnabled) {
    Write-Host ("Using pretrain best checkpoint for finetune from: " + $jointCheckpointSourcePath)
}
elseif (-not $baselineEnabled) {
    Write-Host ("Pretrain checkpoint not found at expected paths: " + $jointCheckpointPath + " or " + $jointTrainingCheckpointPath + "; finetune will run without contrastive checkpoint unless you pass --contrastive-checkpoint manually.")
}

$finetuneScript = (Join-Path $repoRoot "run_finetune.py")
$finetuneArgs = @(
    "--config", $targetFinetuneConfig,
    "--loso",
    "--root-dir", $targetCacheRoot,
    "--output-dir", $finetuneRoot
)
if ($jointCheckpointSourcePath -and -not $baselineEnabled) {
    $finetuneArgs += @("--contrastive-checkpoint", $jointCheckpointSourcePath)
}
if ($TestOnly) {
    $finetuneArgs += "--test-only"
}
if ($FinetuneEpochs -gt 0) {
    $finetuneArgs += @("--epochs", $FinetuneEpochs.ToString())
}
if ($FinetuneBatchSize -gt 0) {
    $finetuneArgs += @("--batch-size", $FinetuneBatchSize.ToString())
}
if ($EvalBatchSize -gt 0) {
    $finetuneArgs += @("--eval-batch-size", $EvalBatchSize.ToString())
}
if ($NumWorkers -ge 0) {
    $finetuneArgs += @("--num-workers", $NumWorkers.ToString())
}
if ($ForceCpu) {
    $finetuneArgs += "--force-cpu"
}
elseif ($GpuCount -gt 0) {
    $finetuneArgs += @("--set", "train.gpu_count=$GpuCount")
    if ($gpuIdList.Count -gt 0) {
        $finetuneArgs += @("--set", "train.gpu_ids=$($gpuIdList -join ',')")
    }
}

Write-Host "Running LOSO finetune..."
if ($useMultiGpu) {
    $ddpFinetuneArgs = @("-m", "torch.distributed.run", "--nproc_per_node", $GpuCount.ToString(), $finetuneScript) + $finetuneArgs
    Invoke-CommandOrThrow -Executable $python -Args $ddpFinetuneArgs -StepName "LOSO finetune"
}
else {
    Invoke-CommandOrThrow -Executable $python -Args (@($finetuneScript) + $finetuneArgs) -StepName "LOSO finetune"
}
