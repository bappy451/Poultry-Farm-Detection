Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    @'
Usage:
  .\scripts\visualize_dino_similarity.ps1 --image PATH [options]

What it does:
  - Loads a DINO / DINOv3 Vision Transformer (pretrained or from --checkpoint)
  - Computes Query-Point Cosine Similarity Map, Salience Map, and PCA Segmentation
  - Saves a 4-panel comparison heatmap figure to --output-dir

Options:
  --image PATH        Path to one or more image tile files. (Required)
  --checkpoint PATH   Optional path to trained .pt checkpoint file.
  --output-dir PATH   Folder to save generated figures. Default: runs/dino_visualizations
  --model-name STR    ViT model size: S, B, L, or H. Default: B
  --patch-size INT    ViT patch size. Default: 16
  --query-x INT       Optional X pixel coordinate for query point.
  --query-y INT       Optional Y pixel coordinate for query point.

Example:
  .\scripts\visualize_dino_similarity.ps1 --image "E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg\tile_1.jpg" --checkpoint "runs/dino_pretrain/last.pt" --output-dir "runs/dino_vis"
'@ | Write-Output
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$originalArgs = @($args)

$showUsage = $false
$imagePath = ""
$checkpointPath = ""
$outputDir = "runs/dino_visualizations"
$modelName = "B"
$patchSize = 16
$queryX = $null
$queryY = $null
$pythonExe = ""

$index = 0
while ($index -lt $originalArgs.Count) {
    $token = $originalArgs[$index]
    switch -Regex ($token) {
        '^(-h|--help|-\?)$' { $showUsage = $true; $index += 1; continue }
        '^(--image|-i)$' {
            $imagePath = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--checkpoint$' {
            $checkpointPath = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--output-dir$' {
            $outputDir = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--model-name$' {
            $modelName = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        '^--python$' {
            $pythonExe = $originalArgs[$index + 1]
            $index += 2
            continue
        }
        default {
            $index += 1
            continue
        }
    }
}

if ($showUsage -or [string]::IsNullOrWhiteSpace($imagePath)) {
    Show-Usage
    exit 0
}

$pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
$pythonScript = Join-Path $scriptDir "poultry_geo.py"
$cmdArgs = @($pythonScript, "--no-log", "visualize-dino-similarity", $imagePath, "--output-dir", $outputDir, "--model-name", $modelName)

if (-not [string]::IsNullOrWhiteSpace($checkpointPath)) {
    $cmdArgs += @("--checkpoint", $checkpointPath)
}

Write-Host "Running DINO cosine similarity visualization..." -ForegroundColor Cyan
& $pythonExe @cmdArgs
