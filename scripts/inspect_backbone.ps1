Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\inspect_backbone.ps1 --data-root PATH [options]

What it does:
  - loads one NAIP batch from nested image tiles
  - adapts n-band imagery to 3 channels with a learnable 1x1 projection
  - runs the RemoteSensingViT backbone and prints shape and precision metadata
  - appends every run to one project log file by default

Required:
  --data-root PATH      Root directory containing nested JPG, PNG, TIFF, or NPY tiles

Options:
  --batch-size N                Batch size. Default: 2
  --crop-size N                 Crop size passed to the dataloader. Default: 224
  --channels N                  Expected input channel count. Default: infer
  --precision MODE              fp32, fp16, bf16, nf4. Default: fp32
  --patch-size N                ViT patch size. Default: 16
  --embed-dim N                 ViT embedding dimension. Default: 192
  --depth N                     Number of transformer blocks. Default: 12
  --num-heads N                 Attention heads. Default: 3
  --mlp-ratio N                 Feed-forward expansion ratio. Default: 4.0
  --num-workers N               DataLoader workers. Default: 4
  --device NAME                 cpu, cuda, or cuda:0. Default: auto
  --gradient-checkpointing      Enable checkpointed transformer blocks
  --python PATH                 Python executable to use. Default: .venv\Scripts\python.exe, then py, then python
  --log-file PATH               Explicit log file path. Default: Code\command.log
  --no-log                      Disable automatic append logging for this run
  --no-progress                 Disable progress output for this run
  -h, --help, -?                Show this help text

Examples:
  .\scripts\inspect_backbone.ps1 --data-root .\tmp\test_jpg --batch-size 2 --crop-size 224 --channels 3 --patch-size 16 --embed-dim 192 --depth 12 --num-heads 3
  .\scripts\inspect_backbone.ps1 --data-root .\tmp\test_jpg --batch-size 2 --crop-size 224 --channels 4 --precision bf16 --patch-size 16 --embed-dim 384 --depth 8 --num-heads 6 --device cpu
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)

$dataRoot = ""
$batchSize = "2"
$cropSize = "224"
$channels = ""
$precision = "fp32"
$patchSize = "16"
$embedDim = "192"
$depth = "12"
$numHeads = "3"
$mlpRatio = "4.0"
$numWorkers = "4"
$deviceName = ""
$enableCheckpointing = $false
$pythonExe = ""
$logFile = $defaultLogFile
$loggingEnabled = $true
$progressEnabled = $true
$showUsage = $false
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    if ($token -eq "--no-log") {
        $loggingEnabled = $false
        $index += 1
        continue
    }
    if ($token -eq "--no-progress") {
        $progressEnabled = $false
        $index += 1
        continue
    }
    if ($token -eq "--log-file") {
        if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) {
            throw "--log-file requires a path."
        }
        $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $originalArgs[$index + 1]
        $index += 2
        continue
    }
    if ($token -like "--log-file=*") {
        $logFile = Resolve-UserPath -BaseDir $invocationCwd -RawPath $token.Substring("--log-file=".Length)
        $index += 1
        continue
    }
    if ($token -eq "--python") {
        if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) {
            throw "--python requires a path."
        }
        $pythonExe = $originalArgs[$index + 1]
        $index += 2
        continue
    }
    if ($token -like "--python=*") {
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
    $index = 0
    while ($index -lt $originalArgs.Count) {
        $token = $originalArgs[$index]
        switch -Regex ($token) {
            '^--data-root=(.*)$' { $dataRoot = $Matches[1]; $index += 1; continue }
            '^--data-root$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--data-root requires a path." }
                $dataRoot = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--batch-size=(.*)$' { $batchSize = $Matches[1]; $index += 1; continue }
            '^--batch-size$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--batch-size requires a value." }
                $batchSize = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--crop-size=(.*)$' { $cropSize = $Matches[1]; $index += 1; continue }
            '^--crop-size$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--crop-size requires a value." }
                $cropSize = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--channels=(.*)$' { $channels = $Matches[1]; $index += 1; continue }
            '^--channels$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--channels requires a value." }
                $channels = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--precision=(.*)$' { $precision = $Matches[1]; $index += 1; continue }
            '^--precision$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--precision requires a value." }
                $precision = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--patch-size=(.*)$' { $patchSize = $Matches[1]; $index += 1; continue }
            '^--patch-size$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--patch-size requires a value." }
                $patchSize = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--embed-dim=(.*)$' { $embedDim = $Matches[1]; $index += 1; continue }
            '^--embed-dim$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--embed-dim requires a value." }
                $embedDim = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--depth=(.*)$' { $depth = $Matches[1]; $index += 1; continue }
            '^--depth$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--depth requires a value." }
                $depth = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--num-heads=(.*)$' { $numHeads = $Matches[1]; $index += 1; continue }
            '^--num-heads$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--num-heads requires a value." }
                $numHeads = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--mlp-ratio=(.*)$' { $mlpRatio = $Matches[1]; $index += 1; continue }
            '^--mlp-ratio$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--mlp-ratio requires a value." }
                $mlpRatio = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--num-workers=(.*)$' { $numWorkers = $Matches[1]; $index += 1; continue }
            '^--num-workers$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--num-workers requires a value." }
                $numWorkers = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--device=(.*)$' { $deviceName = $Matches[1]; $index += 1; continue }
            '^--device$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--device requires a value." }
                $deviceName = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--gradient-checkpointing$' {
                $enableCheckpointing = $true
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
            '^--log-file$' {
                $index += 2
                continue
            }
            '^--log-file=' {
                $index += 1
                continue
            }
            '^--no-log$' {
                $index += 1
                continue
            }
            '^--no-progress$' {
                $progressEnabled = $false
                $index += 1
                continue
            }
            '^(-h|--help|-\\?)$' {
                $showUsage = $true
                $index += 1
                continue
            }
            default {
                throw "Unknown argument: $token"
            }
        }
    }

    if ($showUsage) {
        Show-Usage
    }
    elseif ([string]::IsNullOrWhiteSpace($dataRoot)) {
        throw "--data-root is required."
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $pythonScript = Join-Path $scriptDir "poultry_geo.py"
        $commandArgs = @(
            $pythonScript, "--no-log", "inspect-backbone",
            $dataRoot,
            "--batch-size", $batchSize,
            "--crop-size", $cropSize,
            "--precision", $precision,
            "--patch-size", $patchSize,
            "--embed-dim", $embedDim,
            "--depth", $depth,
            "--num-heads", $numHeads,
            "--mlp-ratio", $mlpRatio,
            "--num-workers", $numWorkers
        )
        if (-not [string]::IsNullOrWhiteSpace($channels)) {
            $commandArgs += @("--channels", $channels)
        }
        if (-not [string]::IsNullOrWhiteSpace($deviceName)) {
            $commandArgs += @("--device", $deviceName)
        }
        if ($enableCheckpointing) {
            $commandArgs += "--gradient-checkpointing"
        }
        if (-not $progressEnabled) {
            $commandArgs += "--no-progress"
        }

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
