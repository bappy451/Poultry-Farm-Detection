Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\train_ssl_grid.ps1 --training-name NAME --config PATH [options]

What it does:
  - runs stateful resume-aware SSL (Masked Image Modeling) grid search training.
  - uses the `ssl-grid-train` CLI command.

Examples:
  .\scripts\train_ssl_grid.ps1 --training-name my-ssl-grid --config ssl_config.yaml --epochs 100
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)
$pythonExe = ""
$logFile = $defaultLogFile
$loggingEnabled = $true
$showUsage = $false
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$forwardArgs = New-Object System.Collections.Generic.List[string]
$trainingName = ""

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    switch -Regex ($token) {
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
            $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $Matches[1]
            $index += 1
            continue
        }
        '^--log-file$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--log-file requires a path." }
            $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--no-log$' {
            $loggingEnabled = $false
            $index += 1
            continue
        }
        '^(-h|--help|-\\?)$' {
            $showUsage = $true
            $index += 1
            continue
        }
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
        default {
            if ([string]::IsNullOrWhiteSpace($trainingName) -and -not $token.StartsWith('-')) {
                $trainingName = $token
            }
            else {
                $forwardArgs.Add($token) | Out-Null
            }
            $index += 1
            continue
        }
    }
}

if ($loggingEnabled) {
    $invocationTokens = @($MyInvocation.MyCommand.Path) + $originalArgs
    Start-ProjectLog -LogFile $logFile -CommandText (Format-CommandString $invocationTokens) -InvocationCwd $invocationCwd
}

$exitCode = 0
try {
    if ($showUsage) {
        Show-Usage
    }
    elseif ([string]::IsNullOrWhiteSpace($trainingName)) {
        throw "--training-name is required."
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $pythonScript = Join-Path $scriptDir "poultry_geo.py"
        $commandArgs = @($pythonScript, "--no-log", "ssl-grid-train", $trainingName) + $forwardArgs

        if ($loggingEnabled) {
            Write-ProjectLogLine -LogFile $logFile -Line ("Running: " + (Format-CommandString (@($pythonExe) + $commandArgs)))
            $exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs -LogFile $logFile -CaptureToLog
        }
        else {
            $exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $commandArgs
        }
    }
}
catch {
    $exitCode = 1
    if ($loggingEnabled) {
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
