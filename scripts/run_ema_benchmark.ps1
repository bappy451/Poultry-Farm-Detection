Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
. (Join-Path $scriptDir "windows_common.ps1")

$forwardArgs = New-Object System.Collections.Generic.List[string]
$requestedPython = ""
$showHelp = $false
$consumeNext = $false

foreach ($arg in $args) {
    if ($arg -eq "--help" -or $arg -eq "-h") {
        $showHelp = $true
        continue
    }
    if ($arg -eq "--python") {
        $consumeNext = $true
        continue
    }
    if ($consumeNext) {
        $requestedPython = $arg
        $consumeNext = $false
        continue
    }
    if ($arg.StartsWith("--python=")) {
        $requestedPython = $arg.Substring("--python=".Length)
        continue
    }
    $forwardArgs.Add($arg)
}

if ($showHelp) {
    Write-Host "Matched multi-seed EMA benchmark"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host '  .\scripts\run_ema_benchmark.ps1 --unlabeled-root "E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg" --include-random'
    Write-Host ""
    Write-Host "Defaults: fraction=0.1, seeds=0 7 42, test split=held-out test."
    Write-Host "Use --help after the Python options to see the full benchmark arguments."
    exit 0
}

if ($consumeNext) {
    throw "--python requires a path to python.exe."
}

$pythonExe = Resolve-PreferredPython -CodeDir $projectRoot -RequestedPython $requestedPython
$pythonScript = Join-Path $scriptDir "run_ema_benchmark.py"
$pythonArgs = @($pythonScript) + @($forwardArgs)
$exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $pythonArgs
if ($exitCode -ne 0) {
    throw "EMA benchmark failed with exit code $exitCode."
}
