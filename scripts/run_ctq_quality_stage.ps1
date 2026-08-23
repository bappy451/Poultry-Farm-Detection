[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$Data,
    [Parameter(Mandatory)]
    [string]$UnlabeledRoot,
    [string]$OutputDir = "runs\ctq_quality_seed7",
    [string]$Model = "yolo11l-obb.pt",
    [int]$Seed = 7,
    [double]$QualityAlpha = 2.0,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Project virtual environment not found: $python" }

& $python -m scripts.train_ctq_quality_obb `
    --quality-alpha $QualityAlpha `
    --data $Data `
    --unlabeled-root $UnlabeledRoot `
    --model $Model `
    --output-dir $OutputDir `
    --seed $Seed `
    --device $Device
if ($LASTEXITCODE -ne 0) { throw "CTQ quality stage failed with exit code $LASTEXITCODE." }
