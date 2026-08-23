[CmdletBinding()]
param(
    [string]$Config = "configs\ema_pseudo_obb_clean_10pct.yaml",
    [string]$OutputDir = "",
    [string]$UnlabeledRoot = "E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg",
    [switch]$ReuseExisting,
    [switch]$SkipTest,
    [switch]$PreflightOnly
)

$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("scripts/run_ema_pseudo_obb.py", "--config", $Config, "--unlabeled-root", $UnlabeledRoot)
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($ReuseExisting) { $arguments += "--reuse-existing" }
if ($SkipTest) { $arguments += "--skip-test" }
if ($PreflightOnly) { $arguments += "--preflight-only" }

Write-Host "Starting detector-level EMA pseudo-OBB experiment..." -ForegroundColor Cyan
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "EMA pseudo-OBB experiment failed with exit code $LASTEXITCODE."
}
Write-Host "EMA pseudo-OBB experiment complete." -ForegroundColor Green