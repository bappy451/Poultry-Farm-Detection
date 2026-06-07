Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\inspect_segmentation.ps1 --data-root PATH [options]

What it does:
  - loads one NAIP batch from nested image tiles
  - adapts n-band imagery to 3 channels
  - runs the ViT backbone with optional LoRA adapters
  - decodes patch embeddings back into a binary roof mask
  - appends every run to one project log file by default

Required:
  --data-root PATH      Root directory containing nested JPG, PNG, TIFF, or NPY tiles

Options:
  --batch-size N                Batch size. Default: 2
  --crop-size N                 Crop size and mask size. Default: 256
  --channels N                  Expected input channel count. Default: infer
  --precision MODE              fp32, fp16, bf16, nf4. Default: fp32
  --patch-size N                ViT patch size. Default: 16
  --embed-dim N                 ViT embedding dimension. Default: 192
  --depth N                     Number of transformer blocks. Default: 8
  --num-heads N                 Attention heads. Default: 6
  --mlp-ratio N                 Feed-forward expansion ratio. Default: 4.0
  --num-workers N               DataLoader workers. Default: 4
  --device NAME                 cpu, gpu, cuda:0, or mps. Default: auto
  --gradient-checkpointing      Enable checkpointed transformer blocks
  --use-lora                    Inject LoRA adapters into ViT attention projections
  --freeze-backbone             Freeze non-LoRA backbone parameters
  --lora-rank N                 LoRA rank. Default: 4
  --lora-alpha N                LoRA scaling alpha. Default: 8.0
  --lora-dropout N              LoRA dropout. Default: 0.0
  --decoder-hidden-channels N   Decoder hidden channels. Default: 128
  --python PATH                 Python executable to use. Default: .venv\Scripts\python.exe, then conda env, then py, then python
  --log-file PATH               Explicit log file path. Default: Code\command.log
  --no-log                      Disable automatic append logging for this run
  --no-progress                 Disable progress output for this run
  -h, --help, -?                Show this help text

Examples:
  .\scripts\inspect_segmentation.ps1 --data-root .\tmp\test_jpg --batch-size 2 --crop-size 256 --channels 3 --patch-size 16 --embed-dim 192 --depth 8 --num-heads 6 --use-lora --freeze-backbone
  .\scripts\inspect_segmentation.ps1 --data-root .\tmp\test_jpg --batch-size 2 --crop-size 512 --channels 4 --precision bf16 --patch-size 16 --embed-dim 384 --depth 12 --num-heads 6 --use-lora --lora-rank 8 --decoder-hidden-channels 128 --device gpu
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)

$dataRoot = ""
$batchSize = "2"
$cropSize = "256"
$channels = ""
$precision = "fp32"
$patchSize = "16"
$embedDim = "192"
$depth = "8"
$numHeads = "6"
$mlpRatio = "4.0"
$numWorkers = "4"
$deviceName = ""
$enableCheckpointing = $false
$useLora = $false
$freezeBackbone = $false
$loraRank = "4"
$loraAlpha = "8.0"
$loraDropout = "0.0"
$decoderHiddenChannels = "128"
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
    $index = 0
    while ($index -lt $originalArgs.Count) {
        $token = $originalArgs[$index]
        switch -Regex ($token) {
            '^--data-root$' { $dataRoot = $originalArgs[$index + 1]; $index += 2; continue }
            '^--batch-size$' { $batchSize = $originalArgs[$index + 1]; $index += 2; continue }
            '^--crop-size$' { $cropSize = $originalArgs[$index + 1]; $index += 2; continue }
            '^--channels$' { $channels = $originalArgs[$index + 1]; $index += 2; continue }
            '^--precision$' { $precision = $originalArgs[$index + 1]; $index += 2; continue }
            '^--patch-size$' { $patchSize = $originalArgs[$index + 1]; $index += 2; continue }
            '^--embed-dim$' { $embedDim = $originalArgs[$index + 1]; $index += 2; continue }
            '^--depth$' { $depth = $originalArgs[$index + 1]; $index += 2; continue }
            '^--num-heads$' { $numHeads = $originalArgs[$index + 1]; $index += 2; continue }
            '^--mlp-ratio$' { $mlpRatio = $originalArgs[$index + 1]; $index += 2; continue }
            '^--num-workers$' { $numWorkers = $originalArgs[$index + 1]; $index += 2; continue }
            '^--device$' { $deviceName = $originalArgs[$index + 1]; $index += 2; continue }
            '^--gradient-checkpointing$' { $enableCheckpointing = $true; $index += 1; continue }
            '^--use-lora$' { $useLora = $true; $index += 1; continue }
            '^--freeze-backbone$' { $freezeBackbone = $true; $index += 1; continue }
            '^--lora-rank$' { $loraRank = $originalArgs[$index + 1]; $index += 2; continue }
            '^--lora-alpha$' { $loraAlpha = $originalArgs[$index + 1]; $index += 2; continue }
            '^--lora-dropout$' { $loraDropout = $originalArgs[$index + 1]; $index += 2; continue }
            '^--decoder-hidden-channels$' { $decoderHiddenChannels = $originalArgs[$index + 1]; $index += 2; continue }
            '^--python$' { $index += 2; continue }
            '^--python=' { $index += 1; continue }
            '^--log-file$' { $index += 2; continue }
            '^--log-file=' { $index += 1; continue }
            '^--no-log$' { $index += 1; continue }
            '^--no-progress$' { $progressEnabled = $false; $index += 1; continue }
            '^(-h|--help|-\\?)$' { $showUsage = $true; $index += 1; continue }
            default { throw "Unknown argument: $token" }
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
            $pythonScript, "--no-log", "inspect-segmentation",
            $dataRoot,
            "--batch-size", $batchSize,
            "--crop-size", $cropSize,
            "--precision", $precision,
            "--patch-size", $patchSize,
            "--embed-dim", $embedDim,
            "--depth", $depth,
            "--num-heads", $numHeads,
            "--mlp-ratio", $mlpRatio,
            "--num-workers", $numWorkers,
            "--lora-rank", $loraRank,
            "--lora-alpha", $loraAlpha,
            "--lora-dropout", $loraDropout,
            "--decoder-hidden-channels", $decoderHiddenChannels
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
        if ($useLora) {
            $commandArgs += "--use-lora"
        }
        if ($freezeBackbone) {
            $commandArgs += "--freeze-backbone"
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
