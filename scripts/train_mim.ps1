Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\train_mim.ps1 --data-root PATH --output-dir PATH [options]

What it does:
  - builds NAIP train and validation loaders
  - trains a MAE-style masked image modeling run
  - supports fp32, fp16, bf16, and nf4/QLoRA-style runs
  - appends every run to one project log file by default

Examples:
  .\scripts\train_mim.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\mim-smoke --batch-size 4 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision bf16 --device gpu
  .\scripts\train_mim.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\mim-nf4 --batch-size 4 --crop-size 224 --channels 4 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --epochs 1 --precision nf4 --use-lora --freeze-backbone
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
$dataRoot = ""
$outputDir = ""

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    switch -Regex ($token) {
        '^--python$' {
            if ($index + 1 -ge $originalArgs.Count) { throw "--python requires a path." }
            $pythonExe = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--python=' {
            $pythonExe = $token.Substring("--python=".Length)
            $index += 1
            continue
        }
        '^--log-file$' {
            if ($index + 1 -ge $originalArgs.Count) { throw "--log-file requires a path." }
            $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--log-file=' {
            $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $token.Substring("--log-file=".Length)
            $index += 1
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
        '^--data-root$' {
            if ($index + 1 -ge $originalArgs.Count) { throw "--data-root requires a path." }
            $dataRoot = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--output-dir$' {
            if ($index + 1 -ge $originalArgs.Count) { throw "--output-dir requires a path." }
            $outputDir = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        default {
            $forwardArgs.Add($token) | Out-Null
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
    elseif ([string]::IsNullOrWhiteSpace($dataRoot) -or [string]::IsNullOrWhiteSpace($outputDir)) {
        throw "--data-root and --output-dir are required."
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $pythonScript = Join-Path $scriptDir "poultry_geo.py"
        $commandArgs = @($pythonScript, "--no-log", "train-mim", $dataRoot, $outputDir) + $forwardArgs

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
