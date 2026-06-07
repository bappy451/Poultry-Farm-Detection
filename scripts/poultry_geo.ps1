Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\poultry_geo.ps1 [wrapper-options] <command> [command-options]

What it does:
  - runs the Python poultry-geo CLI from Windows PowerShell
  - appends every run to one project log file by default
  - forwards the remaining arguments directly to `python -m poultry_geo`

Wrapper options:
  --python PATH         Python executable to use. Default: .venv\Scripts\python.exe, then py, then python
  --log-file PATH       Explicit log file path. Default: Code\command.log
  --no-log              Disable automatic append logging for this run
  -h, --help, -?        Show this wrapper help

Examples:
  .\scripts\poultry_geo.ps1 --help
  .\scripts\poultry_geo.ps1 slice ..\Data\Test_ms_ga .\tmp\tiles --tile-size 256 --format png
  .\scripts\poultry_geo.ps1 infer-yolo ..\Data\Test_ms_ga .\weights\best.pt .\detections.csv --window-size 3000 --stride 3000
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
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    if ($token -eq "--no-log") {
        $loggingEnabled = $false
        $index += 1
        continue
    }
    if ($token -eq "--log-file") {
        if ($index + 1 -ge $originalArgs.Count) {
            throw "--log-file requires a path."
        }
        $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $originalArgs[$index + 1]
        $index += 2
        continue
    }
    if ($token.StartsWith("--log-file=")) {
        $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $token.Substring("--log-file=".Length)
        $index += 1
        continue
    }
    if ($token -eq "--python") {
        if ($index + 1 -ge $originalArgs.Count) {
            throw "--python requires a path."
        }
        $pythonExe = $originalArgs[$index + 1]
        $index += 2
        continue
    }
    if ($token.StartsWith("--python=")) {
        $pythonExe = $token.Substring("--python=".Length)
        $index += 1
        continue
    }
    $index += 1
}

if ($loggingEnabled) {
    $invocationTokens = @($MyInvocation.MyCommand.Path) + $originalArgs
    Start-ProjectLog -LogFile $logFile -CommandText (Format-CommandString $invocationTokens) -InvocationCwd $invocationCwd
}

$exitCode = 0

try {
    $forwardArgs = New-Object System.Collections.Generic.List[string]
    $index = 0
    while ($index -lt $originalArgs.Count) {
        $token = $originalArgs[$index]
        switch -Regex ($token) {
            '^--no-log$' {
                $index += 1
                continue
            }
            '^--log-file$' {
                $index += 2
                continue
            }
            '^--log-file=' {
                $index += 1
                continue
            }
            '^--python$' {
                $index += 2
                continue
            }
            '^--python=' {
                $index += 1
                continue
            }
            default {
                $forwardArgs.Add($token)
                $index += 1
                continue
            }
        }
    }

    if ($forwardArgs.Count -eq 0 -or @("--help", "-h", "-?").Contains($forwardArgs[0])) {
        Show-Usage
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $commandArgs = @("-m", "poultry_geo", "--no-log") + $forwardArgs.ToArray()
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
