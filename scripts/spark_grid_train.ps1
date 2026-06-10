Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\spark_grid_train.ps1 --training-name NAME [options]

What it does:
  - runs stateful resume-aware SparK (SSL) grid search training
  - appends every run to one project log file by default
  - forwards the remaining arguments directly to `scripts/poultry_geo.py`

Wrapper options:
  --training-name NAME  Run identifier.
  --config PATH         Path to the SparK grid-search YAML config. Default: configs/spark_grid_search.yaml
  --epochs INT          Override runtime.epochs from the config.
  --root-dir PATH       Root directory for saved runs. Default: trained_models
  --verbose             Enable rich multi-line progress reporting with live metrics.
  --python PATH         Python executable to use.
  --log-file PATH       Explicit log file path.
  --no-log              Disable automatic append logging for this run
  -h, --help            Show this wrapper help

Examples:
  .\scripts\spark_grid_train.ps1 --training-name my-spark-grid --verbose
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$invocationCwd = (Get-Location).Path
$loggingEnabled = $true
$logFile = Join-Path $codeDir "command.log"
$pythonExe = ""
$trainingName = ""
$configPath = "configs/spark_grid_search.yaml"
$epochs = $null
$rootDir = "trained_models"
$forwardArgs = New-Object System.Collections.Generic.List[string]

$originalArgs = $args
$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    switch -Regex ($token) {
        '^--training-name=(.*)$' {
            $trainingName = $Matches[1]
            $index += 1
            continue
        }
        '^--training-name$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--training-name requires a value." }
            $trainingName = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--config=(.*)$' {
            $configPath = $Matches[1]
            $index += 1
            continue
        }
        '^--config$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--config requires a path." }
            $configPath = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--epochs=(.*)$' {
            $epochs = $Matches[1]
            $index += 1
            continue
        }
        '^--epochs$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--epochs requires an integer." }
            $epochs = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--root-dir=(.*)$' {
            $rootDir = $Matches[1]
            $index += 1
            continue
        }
        '^--root-dir$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--root-dir requires a path." }
            $rootDir = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--python=(.*)$' {
            $pythonExe = $Matches[1]
            $index += 1
            continue
        }
        '^--python$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--python requires a path." }
            $pythonExe = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--log-file=(.*)$' {
            $logFile = $Matches[1]
            $index += 1
            continue
        }
        '^--log-file$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--log-file requires a path." }
            $logFile = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--no-log$' { $loggingEnabled = $false; $index += 1; continue }
        '^(-h|--help)$' { Show-Usage; exit 0 }
        default {
            if ([string]::IsNullOrWhiteSpace($trainingName)) { $trainingName = $token }
            else { $forwardArgs.Add($token) | Out-Null }
            $index += 1
            continue
        }
    }
}

if ([string]::IsNullOrWhiteSpace($trainingName)) {
    Write-Error "--training-name is required."
    Show-Usage
    exit 1
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
    if ($loggingEnabled) {
        $invocationTokens = @($MyInvocation.MyCommand.Path) + $originalArgs
        Start-ProjectLog -LogFile $logFile -CommandText (Format-CommandString $invocationTokens) -InvocationCwd $invocationCwd
    }

    $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
    $pythonScript = Join-Path $scriptDir "poultry_geo.py"
    $commandArgs = @($pythonScript, "--no-log", "spark-grid-train", $trainingName, "--config", $configPath, "--root-dir", $rootDir)
    if ($null -ne $epochs) { $commandArgs += @("--epochs", $epochs) }
    $commandArgs += $forwardArgs.ToArray()

    if ($loggingEnabled) {
        Write-ProjectLogLine -LogFile $logFile -Line ("Running: " + (Format-CommandString (@($pythonExe) + $commandArgs)))
        $exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs -LogFile $logFile -CaptureToLog
    }
    else {
        $exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs
    }
}
catch {
    $exitCode = 1
    if ($loggingEnabled) {
        Write-ProjectLogLine -LogFile $logFile -Line ("Fatal error: " + $_.Exception.Message)
        Write-ProjectException -LogFile $logFile -ErrorRecord $_
    }
    else {
        Write-Error $_
    }
}
finally {
    if ($loggingEnabled) {
        Finish-ProjectLog -LogFile $logFile -ExitCode $exitCode -Stopwatch $stopwatch
    }
}

exit $exitCode
