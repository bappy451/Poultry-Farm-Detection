Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @'
Usage:
  .\scripts\plot_run.ps1 RUN_DIR [options]

What it does:
  - Generates professional-grade training plots for a single run (loss curves, etc).

Options:
  RUN_DIR               Path to the run directory containing metrics.csv/json.
  --output-dir DIR      Directory to save plots. Default: RUN_DIR\plots.
  --python PATH         Python executable to use.
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDir = ""
$outputDir = ""
$pythonExe = ""
$forwardArgs = New-Object System.Collections.Generic.List[string]

$index = 0
while ($index -lt $args.Count) {
    $token = $args[$index]
    switch -Regex ($token) {
        '^--output-dir=(.*)$' {
            $outputDir = $Matches[1]
            $index += 1
            continue
        }
        '^--output-dir$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--output-dir requires a path." }
            $outputDir = $args[$index + 1]
            $index += 2
            continue
        }
        '^--python=(.*)$' {
            $pythonExe = $Matches[1]
            $index += 1
            continue
        }
        '^--python$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--python requires a path." }
            $pythonExe = $args[$index + 1]
            $index += 2
            continue
        }
        '^(-h|--help)$' {
            Show-Usage
            exit 0
        }
        default {
            if ([string]::IsNullOrWhiteSpace($runDir) -and -not $token.StartsWith('-')) {
                $runDir = $token
            }
            else {
                $forwardArgs.Add($token) | Out-Null
            }
            $index += 1
            continue
        }
    }
}

if ([string]::IsNullOrWhiteSpace($runDir)) {
    Write-Error "RUN_DIR is required."
    Show-Usage
    exit 1
}

if ([string]::IsNullOrWhiteSpace($outputDir)) {
    $outputDir = Join-Path $runDir "plots"
}

$poultryGeo = Join-Path $scriptDir "poultry_geo.ps1"
$commandArgs = @()
if (-not [string]::IsNullOrWhiteSpace($pythonExe)) {
    $commandArgs += @("--python", $pythonExe)
}
$commandArgs += @("plot-run", $runDir, "--output-dir", $outputDir)
$commandArgs += $forwardArgs.ToArray()

& $poultryGeo $commandArgs
