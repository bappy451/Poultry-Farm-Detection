# run_all_evaluations.ps1
# Sequentially executes downstream evaluation for three datasets.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host "Starting Unified 3-Dataset Downstream Evaluation"
Write-Host "Datasets: Poultry House (50 ep), GRSD (50 ep), DOTA (20 ep)"
Write-Host "============================================================"

# 1. Poultry House
Write-Host "`n[1/3] Running Poultry House Evaluation..."
.venv\Scripts\python.exe scripts\downstream_scratch_compare.py --config configs\downstream_scratch_compare.yaml

# 2. GRSD
Write-Host "`n[2/3] Running GRSD Evaluation..."
.venv\Scripts\python.exe scripts\downstream_scratch_compare.py --config configs\grsd_scratch_compare.yaml

# 3. DOTA
Write-Host "`n[3/3] Running DOTA Evaluation..."
.venv\Scripts\python.exe scripts\downstream_scratch_compare.py --config configs\dota_scratch_compare.yaml

Write-Host "`n============================================================"
Write-Host "All Evaluations Completed Successfully!"
Write-Host "============================================================"
