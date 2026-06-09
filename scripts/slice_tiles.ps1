Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\slice_tiles.ps1 --input-dir PATH --output-dir PATH [options]

What it does:
  - slices one GeoTIFF or a directory tree of GeoTIFFs into square tiles
  - defaults to non-overlapping windows (stride = tile-size)
  - snaps the last window to the image edge instead of zero padding
  - writes tif, png, jpg, or jpeg tiles
  - appends every run to one project log file by default

Required:
  --input-dir PATH      GeoTIFF file or directory of GeoTIFFs to slice
  --output-dir PATH     Directory where sliced tiles will be written

Options:
  --tile-size N         Square tile size in pixels. Default: 256
  --stride N            Sliding-window stride in pixels. Default: tile size
  --format FMT          Output tile format: tif, png, jpg, jpeg. Default: png
  --jpeg-quality N      JPEG quality when format is jpg or jpeg. Default: 95
  --workers N|auto      Number of file-level worker processes. Default: auto
  --python PATH         Python executable to use. Default: .venv\Scripts\python.exe, then py, then python
  --log-file PATH       Explicit log file path. Default: Code\command.log
  --no-log              Disable automatic append logging for this run
  --no-progress         Disable progress output for this run
  -h, --help, -?        Show this help text

Examples:
  .\scripts\slice_tiles.ps1 --input-dir ..\Data\Test_ms_ga --output-dir .\tmp\tiles_256_png --tile-size 256 --format png
  .\scripts\slice_tiles.ps1 --input-dir ..\Data\Test_ms_ga --output-dir .\tmp\tiles_64_jpg --tile-size 64 --stride 32 --format jpg --workers 4
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$defaultLogFile = Join-Path $codeDir "command.log"
$invocationCwd = (Get-Location).Path
$originalArgs = @($args)

$inputDir = ""
$outputDir = ""
$tileSize = "256"
$stride = ""
$format = "png"
$jpegQuality = "95"
$workers = "auto"
$pythonExe = ""
$logFile = $defaultLogFile
$loggingEnabled = $true
$showUsage = $false
$progressEnabled = $true
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
            '^--input-dir=(.*)$' { $inputDir = $Matches[1]; $index += 1; continue }
            '^--input-dir$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--input-dir requires a path." }
                $inputDir = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--output-dir=(.*)$' { $outputDir = $Matches[1]; $index += 1; continue }
            '^--output-dir$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--output-dir requires a path." }
                $outputDir = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--tile-size=(.*)$' { $tileSize = $Matches[1]; $index += 1; continue }
            '^--tile-size$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--tile-size requires a value." }
                $tileSize = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--stride=(.*)$' { $stride = $Matches[1]; $index += 1; continue }
            '^--stride$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--stride requires a value." }
                $stride = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--format=(.*)$' { $format = $Matches[1]; $index += 1; continue }
            '^--format$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--format requires a value." }
                $format = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--jpeg-quality=(.*)$' { $jpegQuality = $Matches[1]; $index += 1; continue }
            '^--jpeg-quality$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--jpeg-quality requires a value." }
                $jpegQuality = $originalArgs[$index + 1]
                $index += 2
                continue
            }
            '^--workers=(.*)$' { $workers = $Matches[1]; $index += 1; continue }
            '^--workers$' {
                if ($index + 1 -ge $originalArgs.Count -or $originalArgs[$index + 1].StartsWith('-')) { throw "--workers requires a value." }
                $workers = $originalArgs[$index + 1]
                $index += 2
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
    elseif ([string]::IsNullOrWhiteSpace($inputDir) -or [string]::IsNullOrWhiteSpace($outputDir)) {
        throw "--input-dir and --output-dir are required."
    }
    else {
        $pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
        $commandArgs = @(
            "-m", "poultry_geo", "--no-log", "slice",
            $inputDir,
            $outputDir,
            "--tile-size", $tileSize,
            "--format", $format,
            "--jpeg-quality", $jpegQuality,
            "--workers", $workers
        )
        if (-not [string]::IsNullOrWhiteSpace($stride)) {
            $commandArgs += @("--stride", $stride)
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
