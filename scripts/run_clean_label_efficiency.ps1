[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$ReuseExisting,
    [string]$OutputDir = "runs\label_efficiency_clean",
    [string]$Fractions = "1,5,10,25,50,100",
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python -ErrorAction Stop).Source

$arguments = @(
    (Join-Path $PSScriptRoot "run_clean_label_efficiency.py"),
    "--output-dir", $OutputDir,
    "--fractions", $Fractions,
    "--device", $Device
)
if (-not $Full) { $arguments += "--pilot" }
if ($ReuseExisting) { $arguments += "--reuse-existing" }

Write-Host "Starting clean raw-train-only label-efficiency experiment..." -ForegroundColor Cyan
if (-not $Full) {
    Write-Host "Pilot mode: one seed at all fractions. Use -Full after reviewing it for seeds 0, 7 and 42. Add -ReuseExisting to extend the same output directory." -ForegroundColor Yellow
}
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Clean label-efficiency experiment failed with exit code $LASTEXITCODE." }


