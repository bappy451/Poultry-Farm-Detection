[CmdletBinding()]
param(
    [string]$Config = "configs\ssl_publication_study_clean.yaml",
    [string]$OutputDir = "",
    [ValidateSet("all", "pilot", "confirmation", "curve")]
    [string]$Stage = "all",
    [switch]$PreflightOnly,
    [switch]$Plots
)

$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("scripts/run_ssl_publication_study.py", "--config", $Config, "--stage", $Stage)
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($PreflightOnly) { $arguments += "--preflight-only" }
if ($Plots) { $arguments += "--plots" }

Write-Host "Starting validation-gated SSL publication study..." -ForegroundColor Cyan
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "SSL publication study failed with exit code $LASTEXITCODE."
}
Write-Host "SSL publication study complete." -ForegroundColor Green
