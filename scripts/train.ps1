#!/usr/bin/env pwsh
<#
Full ML pipeline: dataset -> train -> evaluate.
Usage: scripts/train.ps1 [-Experiment v1_resnet18] [-Epochs 30]
#>
param(
    [string]$Experiment = "v1_resnet18",
    [int]$Epochs = 0
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = $root

Write-Host "== 1/3 Preparing dataset ==" -ForegroundColor Cyan
& $py ml\data\prepare.py --config ml\configs\train_v1.yaml
if ($LASTEXITCODE -ne 0) { throw "dataset preparation failed" }

Write-Host "== 2/3 Training ==" -ForegroundColor Cyan
$trainArgs = @("ml\training\trainer.py", "--config", "ml\configs\train_v1.yaml", "--name", $Experiment)
if ($Epochs -gt 0) {
    # quick override run
    $tmp = New-TemporaryFile
    Copy-Item ml\configs\train_v1.yaml $tmp.FullName -Force
    & $py ml\training\trainer.py --config ml\configs\train_v1.yaml --name $Experiment
} else {
    & $trainArgs | Tee-Object -FilePath "train_$Experiment.log"
}
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host "== 3/3 Evaluating on test split ==" -ForegroundColor Cyan
& $py ml\evaluation\evaluate.py --config ml\configs\train_v1.yaml --experiment $Experiment
if ($LASTEXITCODE -ne 0) { throw "evaluation failed" }

Write-Host "Done. Artifacts in models\experiments\$Experiment" -ForegroundColor Green
