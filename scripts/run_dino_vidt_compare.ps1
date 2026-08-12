Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Ensure UTF-8 console encoding so modern progress bars and symbols never mangle into ΓÇª
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\run_dino_vidt_compare.ps1 [--config PATH] [--dry-run] [options]

What it does:
  - Runs the 9-experiment DINO ViDT+ Downstream Comparison Suite sequentially
  - Evaluates 3 initialization strategies (DINO pretrained, ImageNet supervised, Random scratch)
  - Across 3 target datasets (Poultry House Detection, GRSD Greenhouse, DOTA Aerial)
  - Displays a modern live Rich progress bar with ETA and summary tables
  - Generates comparison_results.csv incrementally in runs/dino_vit_compare/

Options:
  --config PATH   Path to suite YAML config. Default: configs/dino_vit_downstream_compare.yaml
  --dry-run       Run only 1 epoch/batch per experiment to verify pipeline correctness
  --help, -h      Show usage help

Examples:
  .\scripts\run_dino_vidt_compare.ps1 --dry-run
  .\scripts\run_dino_vidt_compare.ps1 --config configs/dino_vit_downstream_compare.yaml
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$configPath = "configs/dino_vit_downstream_compare.yaml"
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
$commandArgs = @("-m", "downstream_finetune", "--dino-vidt-compare", "--config", $configPath)

if ($dryRun) {
    $commandArgs += "--dry-run"
}
$commandArgs += $forwardArgs

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host " Starting DINO ViDT+ Downstream 9-Experiment Comparison Suite" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Python: $pythonExe" -ForegroundColor DarkGray
Write-Host "Config: $configPath" -ForegroundColor Yellow

$exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs
exit $exitCode
