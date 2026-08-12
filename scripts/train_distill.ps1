$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

# Define the path to the virtual environment python
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-Not (Test-Path $pythonExe)) {
    Write-Error "Virtual environment not found at $pythonExe. Please set it up first."
    exit 1
}

$cliScript = Join-Path $scriptDir "poultry_geo.py"

# Pass all arguments to the python script
& $pythonExe $cliScript train-distill @args
