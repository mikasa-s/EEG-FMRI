param(
    [string]$FinetuneConfig = "configs/finetune_ds009999.yaml",
    [string]$CacheRoot = "cache/ds009999",
    [string]$OutputRoot = "outputs/ds009999",
    [int]$Epochs = 0,
    [int]$BatchSize = 0,
    [int]$EvalBatchSize = 0,
    [int]$NumWorkers = -1,
    [string]$PythonExe = "",
    [switch]$ForceCpu
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

$python = if ($PythonExe.Trim()) { $PythonExe } else { "python" }
$argsList = @(
    "run_finetune.py",
    "--config", $FinetuneConfig,
    "--loso",
    "--root-dir", $CacheRoot,
    "--output-dir", (Join-Path $OutputRoot "finetune")
)
if ($Epochs -gt 0) {
    $argsList += @("--epochs", $Epochs.ToString())
}
if ($BatchSize -gt 0) {
    $argsList += @("--batch-size", $BatchSize.ToString())
}
if ($EvalBatchSize -gt 0) {
    $argsList += @("--eval-batch-size", $EvalBatchSize.ToString())
}
if ($NumWorkers -ge 0) {
    $argsList += @("--num-workers", $NumWorkers.ToString())
}
if ($ForceCpu) {
    $argsList += "--force-cpu"
}

& $python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "LOSO finetune failed with exit code $LASTEXITCODE"
}
