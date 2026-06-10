Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @'
Usage:
  .\scripts\plot_ssl_grid.ps1 --training-name NAME [options]

What it does:
  - Generates a professional-grade summary plot for an SSL grid search campaign.
  - Automatically finds the grid_search_results.csv in the trained_models folder.

Options:
  --training-name NAME  Name of the training campaign (folder in trained_models).
  --root-dir DIR        Root directory for trained models. Default: trained_models.
  --output-dir DIR      Directory to save plots. Default: <run_dir>\figures.
  --python PATH         Python executable to use.
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$trainingName = ""
$rootDir = "trained_models"
$outputDir = ""
$pythonExe = ""
$forwardArgs = New-Object System.Collections.Generic.List[string]

$index = 0
while ($index -lt $args.Count) {
    $token = $args[$index]
    switch -Regex ($token) {
        '^--training-name=(.*)$' {
            $trainingName = $Matches[1]
            $index += 1
            continue
        }
        '^--training-name$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--training-name requires a value." }
            $trainingName = $args[$index + 1]
            $index += 2
            continue
        }
        '^--root-dir=(.*)$' {
            $rootDir = $Matches[1]
            $index += 1
            continue
        }
        '^--root-dir$' {
            if ($index + 1 -ge $args.Count -or $args[$index + 1].StartsWith('-')) { throw "--root-dir requires a path." }
            $rootDir = $args[$index + 1]
            $index += 2
            continue
        }
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

if ([string]::IsNullOrWhiteSpace($trainingName)) {
    Write-Error "--training-name is required."
    Show-Usage
    exit 1
}

$resultsCsv = Join-Path $codeDir (Join-Path $rootDir (Join-Path $trainingName "grid_search_results.csv"))

if (-not (Test-Path $resultsCsv)) {
    Write-Error "Could not find results CSV at $resultsCsv"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($outputDir)) {
    $outputDir = Join-Path $codeDir (Join-Path $rootDir (Join-Path $trainingName "figures"))
}

$poultryGeo = Join-Path $scriptDir "poultry_geo.ps1"
$commandArgs = @()
if (-not [string]::IsNullOrWhiteSpace($pythonExe)) {
    $commandArgs += @("--python", $pythonExe)
}
$commandArgs += @("ssl-grid-plot", $resultsCsv, "--output-dir", $outputDir)
$commandArgs += $forwardArgs.ToArray()

& $poultryGeo @commandArgs
