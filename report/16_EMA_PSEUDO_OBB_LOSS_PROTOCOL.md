# Detector-Level EMA Pseudo-OBB Loss Protocol

## Purpose

This is a separate experiment from the existing feature-only EMA adaptation. It adds a detector-level teacher pseudo-label loss for YOLO11-OBB while preserving the original feature-consistency code and outputs.

The primary claim to test is whether high-confidence teacher OBB predictions provide useful unlabeled supervision at 10% labels.

## What is implemented

- Frozen-for-the-step EMA teacher on weak unlabeled views.
- Rotated Ultralytics NMS with class-specific confidence thresholds.
- Native normalized `xywhr` targets passed to Ultralytics `v8OBBLoss`.
- Confidence weighting using the mean confidence of accepted pseudo boxes.
- Class ambiguity safeguard: reject detections whose top-1/top-2 class-score margin is below `pseudo_class_margin`.
- NMS overlap suppression and minimum 2-pixel box filtering.
- Pseudo-loss warmup and ramp: disabled for the first 5 epochs, then linearly ramped for 10 epochs.
- Optional foreground-weighted feature cosine loss, disabled in the primary experiment.
- Stage 2 supervised fine-tuning and validation-only checkpoint selection.
- External test is run only after every seed has finished; it cannot select checkpoints.

The custom loss preserves Ultralytics' four-channel OBB loss contract. The feature regularizer is added once to the vector before the trainer sums channels, avoiding accidental four-fold broadcasting.

## Primary 10% experiment

Configuration: `configs/ema_pseudo_obb_clean_10pct.yaml`

- Clean raw-train-only 10% subset, fixed validation split.
- External test: `temp/poultry_houses_test/data.yaml`.
- Seeds: 0, 7, 42.
- Pseudo weight: 0.05.
- Class thresholds: 0.90, 0.90.
- Class margin: 0.05.
- IoU NMS threshold: 0.50.
- EMA decay: 0.9998.
- Adaptation: 60 epochs; supervised fine-tune: 100 epochs.

## Commands

Activate the project environment, then run the read-only preflight first:

```powershell
conda activate ag-foundation
.\scripts\run_ema_pseudo_obb.ps1 -PreflightOnly
```

Run all three seeds and the final external test:

```powershell
.\scripts\run_ema_pseudo_obb.ps1
```

If all checkpoints already exist and their handoff markers are valid, reuse them and only run the final test:

```powershell
.\scripts\run_ema_pseudo_obb.ps1 -ReuseExisting
```

Training-only mode (useful before the external test):

```powershell
.\scripts\run_ema_pseudo_obb.ps1 -SkipTest
```

## Controls

Foreground-feature control:

```powershell
.\scripts\run_ema_pseudo_obb.ps1 -Config configs\ema_foreground_feature_clean_10pct.yaml
```

Combined pseudo-OBB plus feature consistency:

```powershell
.\scripts\run_ema_pseudo_obb.ps1 -Config configs\ema_pseudo_obb_combined_clean_10pct.yaml
```

These controls have isolated output directories and do not modify the existing feature-only EMA pipeline.

## Interpretation and selection

Select the stage-2 `best.pt` only by the fixed validation metric. Report per-seed mean, standard deviation, and paired differences against the same-seed Ultralytics baseline. Do not tune thresholds or weights from the external test set. The recommended order is:

1. Finish the existing consistency ablation (0, 0.01, 0.02, 0.05).
2. Run the primary pseudo-OBB experiment at 10% for all three seeds.
3. Run the foreground-feature control only if the primary result is competitive.
4. After choosing the loss, run the nested 1%, 5%, 10%, 25%, 50%, and 100% label-efficiency curve.

A positive result requires a consistent improvement over the same-seed Ultralytics baseline, not a single favorable seed. If pseudo boxes are sparse or unstable, report pseudo-box counts and lower the claim rather than relaxing the external-test protocol.