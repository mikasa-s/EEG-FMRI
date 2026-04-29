param(
    [int]$Count = 5,
    [string]$DatasetName = "",
    [string]$Config = "configs\train_joint_contrastive.yaml",
    [string]$Checkpoint = "pretrained_weights\pretrain_full\contrastive\checkpoints\best.pth",
    [string]$OutputDir = "outputs\visualizations\contrastive",
    [int]$BatchSize = 128,
    [int]$MaxSamples = 256,
    [int]$TsneMaxPoints = 256,
    [int]$HeatmapMaxPoints = 128,
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"

if ($Count -lt 1) {
    throw "Count must be >= 1."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    for ($i = 1; $i -le $Count; $i++) {
        Write-Host "[$i/$Count] Running contrastive visualization..."
        $argsList = @(
            "run_visualize.py",
            "contrastive",
            "--config", $Config,
            "--checkpoint", $Checkpoint,
            "--output-dir", $OutputDir,
            "--batch-size", $BatchSize,
            "--max-samples", $MaxSamples,
            "--tsne-max-points", $TsneMaxPoints,
            "--heatmap-max-points", $HeatmapMaxPoints
        )
        if ($DatasetName.Trim()) {
            $argsList += @("--dataset-name", $DatasetName.Trim())
        }
        if ($Device.Trim()) {
            $argsList += @("--device", $Device.Trim())
        }
        python @argsList
        if ($LASTEXITCODE -ne 0) {
            throw "contrastive visualization failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
