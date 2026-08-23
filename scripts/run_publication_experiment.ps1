param(
    [string]$UnlabeledRoot = "E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg",
    [string]$Data = "configs\poultry_80_10_10.yaml",
    [string]$FinalTestData = "temp\poultry_houses_test\data.yaml",
    [string]$OutputDir = "runs\publication_ema_10pct_fixed",
    [double]$Fraction = 0.1,
    [int[]]$Seeds = @(0, 7, 42),
    [int]$AdaptEpochs = 60,
    [int]$FinetuneEpochs = 100,
    [int]$Batch = 16,
    [int]$UnlabeledBatch = 16,
    [int]$UnlabeledLimit = 250000,
    [int]$TestBatch = 32,
    [int]$Workers = 12,
    [int]$UnlabeledWorkers = 2,
    [int]$PrefetchFactor = 2,
    [int]$Device = 0,
    [switch]$NoRandom,
    [switch]$ResumeEMA,
    [switch]$ReuseExisting,
    [switch]$SkipTraining,
    [switch]$NoPlots,
    [switch]$NoProgress,
    [switch]$AllowTestOverlap,
    [string]$Python = "",
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
. (Join-Path $scriptDir "windows_common.ps1")

if ($Help) {
    Write-Host "Publication EMA benchmark"
    Write-Host "Usage: .\scripts\run_publication_experiment.ps1"
    Write-Host "Default: 10% labels, seeds 0/7/42, COCO + repaired EMA + random, final test on temp\poultry_houses_test."
    Write-Host "Use -Data configs\poultry_clean_raw_90_10.yaml for the raw-train-only clean protocol."
    Write-Host "Throughput preset: batch=16, unlabeled-batch=16, test-batch=32, workers=12 labeled + 2 unlabeled; bounded NAIP pool=250000."
    Write-Host "Useful switches: -NoRandom -ResumeEMA -ReuseExisting -SkipTraining -NoPlots -NoProgress -AllowTestOverlap"
    exit 0
}

$trainingData = Resolve-UserPath -BaseDir $projectRoot -RawPath $Data
$finalTestData = Resolve-UserPath -BaseDir $projectRoot -RawPath $FinalTestData
if (-not (Test-Path -LiteralPath $trainingData)) { throw "Training dataset YAML not found: $trainingData" }
if (-not (Test-Path -LiteralPath $finalTestData)) { throw "Final test dataset YAML not found: $finalTestData" }
if (-not $SkipTraining -and -not (Test-Path -LiteralPath $UnlabeledRoot)) { throw "Unlabeled root not found: $UnlabeledRoot" }
if ($Fraction -le 0 -or $Fraction -gt 1) { throw "Fraction must be greater than 0 and at most 1." }
if ($Seeds.Count -eq 0) { throw "At least one seed is required." }

$resolvedOutput = Resolve-UserPath -BaseDir $projectRoot -RawPath $OutputDir
$pythonExe = Resolve-PreferredPython -CodeDir $projectRoot -RequestedPython $Python
$pythonScript = Join-Path $scriptDir "run_ema_benchmark.py"

$pythonArgs = New-Object System.Collections.Generic.List[string]
foreach ($token in @(
    $pythonScript,
    "--data", $trainingData,
    "--final-test-data", $finalTestData,
    "--final-test-split", "test",
    "--unlabeled-root", $UnlabeledRoot,
    "--model", "yolo11l-obb.pt",
    "--random-model", "yolo11l-obb.yaml",
    "--output-dir", $resolvedOutput,
    "--fraction", $Fraction.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--adapt-epochs", $AdaptEpochs,
    "--finetune-epochs", $FinetuneEpochs,
    "--batch", $Batch,
    "--unlabeled-batch", $UnlabeledBatch,
    "--unlabeled-limit", $UnlabeledLimit,
    "--test-batch", $TestBatch,
    "--imgsz", 640,
    "--unlabeled-imgsz", 448,
    "--workers", $Workers,
    "--unlabeled-workers", $UnlabeledWorkers,
    "--prefetch-factor", $PrefetchFactor,
    "--device", $Device,
    "--seeds"
)) { $pythonArgs.Add([string]$token) | Out-Null }
foreach ($seed in $Seeds) { $pythonArgs.Add([string]$seed) | Out-Null }
if (-not $NoRandom) { $pythonArgs.Add("--include-random") | Out-Null }
if ($ResumeEMA) { $pythonArgs.Add("--resume-ema") | Out-Null }
if ($ReuseExisting) { $pythonArgs.Add("--reuse-existing") | Out-Null }
if ($SkipTraining) { $pythonArgs.Add("--skip-training") | Out-Null }
if ($NoPlots) { $pythonArgs.Add("--no-plots") | Out-Null }
if ($NoProgress) { $pythonArgs.Add("--no-progress") | Out-Null }
if ($AllowTestOverlap) { $pythonArgs.Add("--allow-test-overlap") | Out-Null }

Write-Host "Publication experiment"
Write-Host "  Training data : $trainingData"
Write-Host "  Final test    : $finalTestData (split=test)"
Write-Host "  Fraction      : $Fraction"
Write-Host "  Seeds         : $($Seeds -join ', ')"
Write-Host "  Output        : $resolvedOutput"
Write-Host ""

$exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments $pythonArgs
if ($exitCode -ne 0) { throw "Publication experiment failed with exit code $exitCode." }
