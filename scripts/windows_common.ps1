# Set strict mode and error preference
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Silence distracting HuggingFace symlink warnings on Windows
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

function Resolve-UserPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BaseDir,
        [Parameter(Mandatory = $true)]
        [string]$RawPath
    )

    if ([System.IO.Path]::IsPathRooted($RawPath)) {
        return [System.IO.Path]::GetFullPath($RawPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $RawPath))
}

function Format-CommandToken {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Token
    )

    if ($Token -match '[\s"`]') {
        return '"' + $Token.Replace('"', '\"') + '"'
    }
    return $Token
}

function Format-CommandString {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Tokens
    )

    return (($Tokens | ForEach-Object { Format-CommandToken $_ }) -join ' ')
}

function Write-ProjectLogLine {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Line
    )

    try {
        $Line | Tee-Object -FilePath $LogFile -Append | Out-Host
    }
    catch [System.IO.IOException] {
        $Line | Out-Host
        Write-Warning "[logging] Could not write to $LogFile (file locked). Continuing without log file entry."
    }
    catch {
        $Line | Out-Host
        throw $_
    }
}

function Start-ProjectLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [string]$CommandText,
        [Parameter(Mandatory = $true)]
        [string]$InvocationCwd
    )

    $parentDir = Split-Path -Parent $LogFile
    if (-not [string]::IsNullOrWhiteSpace($parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    Write-ProjectLogLine -LogFile $LogFile -Line ""
    Write-ProjectLogLine -LogFile $LogFile -Line ("=" * 80)
    Write-ProjectLogLine -LogFile $LogFile -Line "Command Log"
    Write-ProjectLogLine -LogFile $LogFile -Line ("=" * 80)
    Write-ProjectLogLine -LogFile $LogFile -Line ("Started   : " + (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz"))
    Write-ProjectLogLine -LogFile $LogFile -Line ("Command   : " + $CommandText)
    Write-ProjectLogLine -LogFile $LogFile -Line ("CWD       : " + $InvocationCwd)
    Write-ProjectLogLine -LogFile $LogFile -Line ("Log file  : " + $LogFile)
    Write-ProjectLogLine -LogFile $LogFile -Line ("=" * 80)
    Write-ProjectLogLine -LogFile $LogFile -Line ("[logging] Appending command output to " + $LogFile)
}

function Write-ProjectException {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    Write-ProjectLogLine -LogFile $LogFile -Line "[logging] Wrapper failed with an exception:"
    Write-ProjectLogLine -LogFile $LogFile -Line ("Exception type : " + $ErrorRecord.Exception.GetType().FullName)
    Write-ProjectLogLine -LogFile $LogFile -Line ("Message        : " + $ErrorRecord.Exception.Message)
    Write-ProjectLogLine -LogFile $LogFile -Line ("Category       : " + $ErrorRecord.CategoryInfo.ToString())
    if ($null -ne $ErrorRecord.TargetObject) {
        Write-ProjectLogLine -LogFile $LogFile -Line ("Target          : " + $ErrorRecord.TargetObject.ToString())
    }
    if (
        $null -ne $ErrorRecord.InvocationInfo -and
        -not [string]::IsNullOrWhiteSpace($ErrorRecord.InvocationInfo.PositionMessage)
    ) {
        Write-ProjectLogLine -LogFile $LogFile -Line "Position:"
        Write-ProjectLogLine -LogFile $LogFile -Line $ErrorRecord.InvocationInfo.PositionMessage
    }
    if (-not [string]::IsNullOrWhiteSpace($ErrorRecord.ScriptStackTrace)) {
        Write-ProjectLogLine -LogFile $LogFile -Line "PowerShell stack trace:"
        Write-ProjectLogLine -LogFile $LogFile -Line $ErrorRecord.ScriptStackTrace
    }
    if ($null -ne $ErrorRecord.Exception.InnerException) {
        Write-ProjectLogLine -LogFile $LogFile -Line ("Inner exception: " + $ErrorRecord.Exception.InnerException.ToString())
    }
}

function Finish-ProjectLog {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogFile,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode,
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Stopwatch]$Stopwatch
    )

    $Stopwatch.Stop()
    $elapsed = $Stopwatch.Elapsed
    $duration = "{0:00}:{1:00}:{2:00}.{3:000}" -f [math]::Floor($elapsed.TotalHours), $elapsed.Minutes, $elapsed.Seconds, $elapsed.Milliseconds
    $elapsedSeconds = $elapsed.TotalSeconds.ToString("F2", [System.Globalization.CultureInfo]::InvariantCulture)
    Write-ProjectLogLine -LogFile $LogFile -Line (
        "[logging] Finished (exit=" + $ExitCode +
        ", finished=" + (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz") +
        ", duration=" + $duration +
        ", elapsed_seconds=" + $elapsedSeconds + ")"
    )
}

function Invoke-ProjectNativeCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$LogFile = "",
        [switch]$CaptureToLog
    )

    $commandLine = (Format-CommandString (@($Executable) + $Arguments)) + " 2>&1"
    $previousProgressLog = $env:POULTRY_PROGRESS_LOG
    $progressEventLog = ""
    try {
        if ($CaptureToLog) {
            if ([string]::IsNullOrWhiteSpace($LogFile)) {
                throw "LogFile is required when CaptureToLog is enabled."
            }
            $progressEventLog = $LogFile + ".progress." + $PID + ".tmp"
            $env:POULTRY_PROGRESS_LOG = $progressEventLog
            
            # Try to use Tee-Object, fall back to direct output if locked
            try {
                & cmd.exe /d /c $commandLine |
                    Tee-Object -FilePath $LogFile -Append |
                    Out-Host
            }
            catch [System.IO.IOException] {
                & cmd.exe /d /c $commandLine | Out-Host
                Write-Warning "[logging] Could not write command output to $LogFile (file locked)."
            }
        }
        else {
            & cmd.exe /d /c $commandLine | Out-Host
        }
        return $LASTEXITCODE
    }
    finally {
        $env:POULTRY_PROGRESS_LOG = $previousProgressLog
        if (
            $CaptureToLog -and
            -not [string]::IsNullOrWhiteSpace($progressEventLog) -and
            (Test-Path -LiteralPath $progressEventLog)
        ) {
            Get-Content -LiteralPath $progressEventLog | Add-Content -LiteralPath $LogFile
            Remove-Item -LiteralPath $progressEventLog
        }
    }
}

function Resolve-PreferredPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CodeDir,
        [string]$RequestedPython = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($RequestedPython)) {
        $expandedPython = $ExecutionContext.InvokeCommand.ExpandString($RequestedPython)
        if ([System.IO.Path]::IsPathRooted($expandedPython)) {
            return [System.IO.Path]::GetFullPath($expandedPython)
        }
        return [System.IO.Path]::GetFullPath((Join-Path (Get-Location).Path $expandedPython))
    }

    $venvPython = Join-Path $CodeDir ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }

    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return $condaPython
        }
    }

    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($null -ne $condaCommand) {
        try {
            $condaBase = (& $condaCommand.Source info --base 2>$null | Select-Object -First 1).Trim()
            if (-not [string]::IsNullOrWhiteSpace($condaBase)) {
                $projectEnvPython = Join-Path $condaBase "envs\poultry-geo\python.exe"
                if (Test-Path $projectEnvPython) {
                    return $projectEnvPython
                }
            }
        }
        catch {
        }
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        return $pyLauncher.Source
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return $python.Source
    }

    throw "No Python interpreter was found. Create .venv first or pass --python PATH."
}
