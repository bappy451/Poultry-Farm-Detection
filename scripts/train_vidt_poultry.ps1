Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure UTF-8 console encoding so modern progress bars and symbols never mangle into ΓÇª
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\train_vidt_poultry.ps1 [--config PATH] [--dry-run] [options]

What it does:
  - Runs the standalone ViDT+ fine-tuning specifically tailored to the Poultry House dataset
  - Displays a modern live Rich progress bar with ETA and proxy metrics
  - Saves best and last checkpoints to runs/vidt_poultry_standalone/weights/

Options:
  --config PATH   Path to training YAML config. Default: configs/vidt_poultry_standalone.yaml
  --dry-run       Run only 1 epoch/batch per experiment to verify pipeline correctness
  --help, -h      Show usage help

Examples:
  .\scripts\train_vidt_poultry.ps1 --dry-run
  .\scripts\train_vidt_poultry.ps1 --config configs/vidt_poultry_standalone.yaml
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$configPath = "configs/vidt_poultry_standalone.yaml"
$dryRun = $false
$forwardArgs = @()

for ($i = 0; $i -lt $args.Length; $i++) {
    $token = $args[$i]
    if ($token -in @("-h", "--help", "-?")) {
        Show-Usage
        exit 0
    }
    elseif ($token -eq "--config") {
        if ($i + 1 -ge $args.Length) { throw "Missing value for --config" }
        $configPath = $args[$i + 1]
        $i++
    }
    elseif ($token -eq "--dry-run") {
        $dryRun = $true
    }
    else {
        $forwardArgs += $token
    }
}

$pythonExe = Resolve-PreferredPython -CodeDir $repoDir
$commandArgs = @("-m", "downstream_finetune", "--train-vidt", "--config", $configPath)

if ($dryRun) {
    $commandArgs += "--dry-run"
}
$commandArgs += $forwardArgs

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host " Starting Standalone ViDT+ Downstream Fine-Tuning" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor DarkGray
Write-Host "Config: $configPath" -ForegroundColor Yellow

$exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs
exit $exitCode
