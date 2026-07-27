Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\train_dino.ps1 --data-root PATH --output-dir PATH [options]

What it does:
  - builds DINO multi-crop train and validation loaders (2 global + 8 local crops)
  - trains a Domain-Adaptive DINO self-supervised Vision Transformer
  - updates EMA teacher with cosine momentum schedule and prototype centering vector
  - saves both full training checkpoint and extracted clean backbone weights
  - appends every run to one project log file by default

Options:
  --data-root PATH      Directory containing NAIP or RGB tiles. (Required)
  --output-dir PATH     Directory to store metrics.csv and checkpoint .pt files. (Required)
  --batch-size INT      Batch size for student/teacher crops. Default: 16.
  --model-name STR      ViT architecture size: S, B, L, or H. Default: B.
  --precision STR       fp32, fp16, bf16, or nf4. Default: fp32.
  --dino-out-dim INT    Prototype logit dimension K. Default: 65536.
  --epochs INT          Total training epochs. Default: 100.
  --resume              Resume training from last.pt if available in output-dir.
  --verbose             Enable rich multi-line progress reporting.

Examples:
  .\scripts\train_dino.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\dino-smoke --batch-size 4 --model-name S --precision bf16 --device gpu --verbose
  .\scripts\train_dino.ps1 --data-root .\tmp\test_jpg --output-dir .\runs\dino-vitb --batch-size 16 --model-name B --epochs 100 --resume
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$showUsage = $false
$dataRoot = ""
$outputDir = ""
$catalogPath = ""
$configPath = ""
$pythonExe = ""
$loggingEnabled = $true
$logFile = $defaultLogFile
$forwardArgs = [System.Collections.Generic.List[string]]::new()

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    switch -Regex ($token) {
        '^(-h|--help)$' {
            $showUsage = $true
            $index += 1
            continue
        }
        '^--no-log$' {
            $loggingEnabled = $false
            $index += 1
            continue
        }
        '^--log-file$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--log-file requires a path." }
            $logFile = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--python$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--python requires a path." }
            $pythonExe = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--data-root$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--data-root requires a path." }
            $dataRoot = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--output-dir$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--output-dir requires a path." }
            $outputDir = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^(--catalog|--catalog-path|-Catalog|-CatalogPath)$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--catalog requires a path." }
            $catalogPath = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^(--config|-c|-Config)$' {
            if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--config requires a path." }
            $configPath = $originalArgs[$index + 1]
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
    elseif ([string]::IsNullOrWhiteSpace($configPath) -and ([string]::IsNullOrWhiteSpace($dataRoot) -or [string]::IsNullOrWhiteSpace($outputDir))) {
        throw "Either --config or both --data-root and --output-dir are required."
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $pythonScript = Join-Path $scriptDir "poultry_geo.py"
        $commandArgs = @($pythonScript, "--no-log", "train-dino")
        if (-not [string]::IsNullOrWhiteSpace($dataRoot)) {
            $commandArgs += $dataRoot
        }
        if (-not [string]::IsNullOrWhiteSpace($outputDir)) {
            $commandArgs += $outputDir
        }
        if (-not [string]::IsNullOrWhiteSpace($configPath)) {
            $commandArgs += @("--config", $configPath)
        }
        if (-not [string]::IsNullOrWhiteSpace($catalogPath)) {
            $commandArgs += @("--catalog", $catalogPath)
        }
        $commandArgs += $forwardArgs

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
