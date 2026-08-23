<#
.SYNOPSIS
Runs fair 10% few-shot comparisons for Poultry, GRSD, and DOTA using the
DINO-to-YOLO distilled backbone, Ultralytics pretrained weights, and random init.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptPath
. (Join-Path $scriptPath "windows_common.ps1")
Set-Location $projectRoot
$pythonExe = Resolve-PreferredPython -CodeDir $projectRoot
$configs = @("configs/poultry_fewshot_compare.yaml", "configs/grsd_fewshot_compare.yaml", "configs/dota_fewshot_compare.yaml")

foreach ($config in $configs) {
    Write-Host "`n=========================================================================" -ForegroundColor Cyan
    Write-Host "Starting DINO-distilled 10% few-shot comparison: $config" -ForegroundColor Cyan
    Write-Host "=========================================================================" -ForegroundColor Cyan
    & $pythonExe scripts/downstream_distill_compare.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Few-shot comparison failed for $config (exit $LASTEXITCODE)." }
}
Write-Host "`nAll few-shot evaluations completed successfully." -ForegroundColor Green
