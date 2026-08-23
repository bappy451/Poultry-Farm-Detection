# Publication EMA–YOLO experiment protocol

Run the complete experiment from the project root after activating `ag-foundation`:

```powershell
conda activate ag-foundation
.\scripts\run_publication_experiment.ps1
```

The launcher shows a live Rich progress bar for every seed/method/final-test phase and fixes the protocol. The throughput preset uses batch=16, unlabeled-batch=16, test-batch=32, and 12 workers for each loader on the RTX 4090/32-thread workstation. to three seeds (`0`, `7`, `42`), 10% labeled-data fraction, 60 EMA adaptation epochs, 100 supervised fine-tuning epochs, and matched YOLO11l-OBB COCO, EMA-adapted, and random-initialized controls. The independent final report uses `temp/poultry_houses_test/data.yaml`, split `test`, only after all training is complete.

Before training, the runner performs a SHA-256 overlap audit. The current `poultry_houses_test` set contains 702 exact duplicates in `poultry_80_10_10`; the default safe mode creates reproducible train/validation manifests excluding those images. This is required for a valid final comparison. `-AllowTestOverlap` exists only for exploratory analysis and must not be used for publication claims.

Outputs are written under `runs/publication_ema_10pct/`, including the isolated YAML/manifests, overlap audit, per-seed final-test metrics, aggregate mean/std metrics, paired EMA-minus-COCO deltas, plots, and `benchmark_summary.json`.

The primary publication result is the paired difference in `paired_ema_minus_coco.csv`. Report mean ± sample standard deviation across the three seeds, together with every per-seed result; do not select a checkpoint using the final test set.
