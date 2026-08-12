Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\train_spark.ps1 --data-root PATH --output-dir PATH [options]

What it does:
  - builds NAIP train and validation loaders
  - trains a Sparse Masked Modeling (SparK) SSL run
  - supports hierarchical CNN backbones (ResNet50, YOLO CSPDarkNet)
  - appends every run to one project log file by default

Options:
  --verbose             Enable rich multi-line progress reporting with live metrics.

Examples:
  .\scripts\train_spark.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\spark-smoke --batch-size 4 --backbone-type resnet50 --precision bf16 --device gpu --verbose
  .\scripts\train_spark.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\spark-yolo --batch-size 4 --backbone-type cspdarknet --mask-ratio 0.75 --epochs 10 --verbose
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
$dataRoot = "E:\NAIP_Dataset_PH"
$outputDir = "runs\spark_pretrain_yolo11l"

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
        '^--data-root=(.*)$' {
            $dataRoot = $Matches[1]
            $index += 1
            continue
        }
        '^--data-root$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--data-root requires a path." }
            $dataRoot = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--output-dir=(.*)$' {
            $outputDir = $Matches[1]
            $index += 1
            continue
        }
        '^--output-dir$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--output-dir requires a path." }
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

$defaultArgs = @(
    "--backbone-type", "yolo11l",
    "--batch-size", "128",
    "--epochs", "300",
    "--epoch-size", "2500",
    "--precision", "bf16",
    "--learning-rate", "0.0002",
    "--mask-ratio", "0.65",
    "--num-workers", "16"
)

for ($i = 0; $i -lt $defaultArgs.Count; $i += 2) {
    $argName = $defaultArgs[$i]
    $argValue = $defaultArgs[$i+1]
    if ($originalArgs -notcontains $argName) {
        $forwardArgs.Add($argName) | Out-Null
        $forwardArgs.Add($argValue) | Out-Null
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
        $commandArgs = @($pythonScript, "--no-log", "train-spark", $dataRoot, $outputDir) + $forwardArgs

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
