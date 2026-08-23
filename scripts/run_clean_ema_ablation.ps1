[CmdletBinding()]
param(
    [string]$OutputDir = "runs\ema_consistency_ablation_clean",
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
$script = Join-Path $PSScriptRoot "run_clean_ema_ablation.py"

Write-Host "Starting clean 10% EMA consistency-weight ablation (0, 0.01, 0.02, 0.05; seeds 0, 7, 42)..." -ForegroundColor Cyan
& $python $script --output-dir $OutputDir --device $Device
if ($LASTEXITCODE -ne 0) { throw "EMA consistency ablation failed with exit code $LASTEXITCODE." }
