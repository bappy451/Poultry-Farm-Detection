[CmdletBinding()]
param(
    [string]$Config = "configs\ema_pseudo_obb_burnin_clean_10pct.yaml",
    [string]$OutputDir = "",
    [switch]$ReuseExisting,
    [switch]$SkipTest,
    [switch]$PreflightOnly,
    [switch]$NoPlots,
    [switch]$AllowTestOverlap,
    [switch]$SkipMatchedControl
)

$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("scripts/run_pseudo_obb_burnin.py", "--config", $Config)
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($ReuseExisting) { $arguments += "--reuse-existing" }
if ($SkipTest) { $arguments += "--skip-test" }
if ($PreflightOnly) { $arguments += "--preflight-only" }
if ($NoPlots) { $arguments += "--no-plots" }
if ($AllowTestOverlap) { $arguments += "--allow-test-overlap" }
if ($SkipMatchedControl) { $arguments += "--skip-matched-control" }

Write-Host "Starting supervised-burn-in pseudo-OBB benchmark..." -ForegroundColor Cyan
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Burn-in pseudo-OBB benchmark failed with exit code $LASTEXITCODE."
}
Write-Host "Burn-in pseudo-OBB benchmark complete." -ForegroundColor Green
