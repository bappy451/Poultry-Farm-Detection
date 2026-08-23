param(
    [string]$Config = "configs/ctq_publication_protocol_v1.yaml",
    [string]$OutputDir = "runs/ctq_publication_v1",
    [switch]$ReuseExisting,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$arguments = @("-m", "scripts.run_ctq_publication_protocol", "--config", $Config, "--output-dir", $OutputDir)
if ($ReuseExisting) { $arguments += "--reuse-existing" }
if ($PreflightOnly) { $arguments += "--preflight-only" }
& $python @arguments
exit $LASTEXITCODE
