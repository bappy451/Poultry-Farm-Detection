<# Runs fair DINO-distilled YOLO evaluations for Poultry, GRSD, and DOTA. #>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptPath
. (Join-Path $scriptPath "windows_common.ps1")
Set-Location $projectRoot
$pythonExe = Resolve-PreferredPython -CodeDir $projectRoot
$configs = @(
    "configs/downstream_scratch_compare.yaml",
    "configs/grsd_scratch_compare.yaml",
    "configs/dota_scratch_compare.yaml"
)

for ($index = 0; $index -lt $configs.Count; $index++) {
    $config = $configs[$index]
    Write-Host "`n[$($index + 1)/$($configs.Count)] Running DINO-distilled comparison: $config" -ForegroundColor Cyan
    & $pythonExe scripts/downstream_distill_compare.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for $config (exit $LASTEXITCODE)." }
}
Write-Host "`nAll evaluations completed successfully." -ForegroundColor Green
