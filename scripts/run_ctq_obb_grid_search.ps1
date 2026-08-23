[CmdletBinding()]
param(
    [string]$Config = "configs\ctq_obb_grid_search.yaml",
    [string]$OutputDir = "",
    [switch]$ReuseExisting,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @("scripts/run_ctq_obb_grid_search.py", "--config", $Config)
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($ReuseExisting) { $arguments += "--reuse-existing" }
if ($PreflightOnly) { $arguments += "--preflight-only" }

& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "CTQ-OBB grid search failed with exit code $LASTEXITCODE." }
