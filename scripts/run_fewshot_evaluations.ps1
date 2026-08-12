<#
.SYNOPSIS
Runs the 10% few-shot ablation multi-seed comparison pipeline for Poultry, GRSD, and DOTA.
This executes the exact same LP-FT SparK multi-seed evaluation, but uses fraction=0.1
to train on only 10% of each dataset.

.DESCRIPTION
Executes the downstream_scratch_compare.py script sequentially for all three fewshot configurations.
#>

$ErrorActionPreference = 'Stop'

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptPath
$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path (Join-Path $venvPath "Scripts") "python.exe"

Set-Location $projectRoot

if (-not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment Python not found at $pythonExe"
    exit 1
}

$configs = @(
    "configs/poultry_fewshot_compare.yaml",
    "configs/grsd_fewshot_compare.yaml",
    "configs/dota_fewshot_compare.yaml"
)

foreach ($config in $configs) {
    Write-Host ""
    Write-Host "==========================================================================" -ForegroundColor Cyan
    Write-Host "   Starting 10% Few-Shot Multi-Seed Ablation: $config " -ForegroundColor Cyan
    Write-Host "==========================================================================" -ForegroundColor Cyan

    # Run the few-shot evaluation pipeline
    & $pythonExe scripts/downstream_scratch_compare.py --config $config

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Few-Shot ablation pipeline failed for $config with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "==========================================================================" -ForegroundColor Green
Write-Host "   All Few-Shot Ablations completed successfully!                         " -ForegroundColor Green
Write-Host "   Check results in runs/*_fewshot_compare/comparison_results.csv         " -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green
