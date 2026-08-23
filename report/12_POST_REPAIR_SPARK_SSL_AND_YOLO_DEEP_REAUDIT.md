# Post-Repair SparK SSL and YOLO Deep Re-Audit

**Date:** 2026-08-13  
**Scope:** Fresh, read-only inspection of the active SparK SSL, sparse ResNet conversion, checkpoints/resume, grid search, downstream transfer surfaces, package and standalone YOLO GeoTIFF inference, configuration, documentation, and tests. No application code was changed. This report supersedes only the findings rechecked below; earlier reports remain historical evidence.

## Executive conclusion

The codebase is safer than it was after the prior fixes: unsupported SparK-to-YOLO transfer now fails closed, the primary SparK CLI exposes only `resnet50`, inference handles boxes/masks/OBB records, and the focused regression suite currently passes. Those are valuable safeguards.

It is nevertheless **not valid to claim a working SparK pretraining pipeline, a transferable SparK backbone, or a trustworthy multi-raster YOLO inference result**. Four blocking defects remain:

1. The direct trainer writes no transformer/adapter state, although its resume path requires it. Every newly produced direct checkpoint is therefore non-resumable.
2. SparK grid-search raises `NameError: shutil is not defined` at the first improved epoch, after saving its latest checkpoint but before producing its advertised best artifacts.
3. The sparse-ResNet conversion copies dense `OIHW` convolution weights into spconv 2.x `KRSC` weights without conversion and silently ignores the shape error. Its dense export makes the reverse mistake. Thus ImageNet initialization and exported dense backbones are not trustworthy.
4. Package-level polygon NMS compares all files together using unprojected native coordinates and only class labels. It can suppress a prediction from a different GeoTIFF or CRS.

The installed project virtual environment contains Torch, TorchVision, Ultralytics, Shapely, and pytest, but **does not contain `spconv`**. The actual sparse forward/backward path is therefore untested and cannot run there.

## What I examined

- `src/poultry_geo/spark.py`, `spark_models.py`, `train_spark_ssl.py`, `spark_grid_search.py`, `cli.py`, and `checkpoint_manager.py`
- `src/poultry_geo/inference.py`, `pipeline.py`, and `scripts/run_naip_detection.py`
- SparK/downstream scripts and YAML configs
- Focused tests: `test_spark_models.py`, `test_spark_pipeline.py`, `test_spark_training.py`, `test_spark_yolo_regressions.py`, `test_inference.py`, `test_inference_results.py`, and `test_cli.py`
- Official [SparK repository](https://github.com/keyu-tian/SparK), [spconv usage documentation](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md), [spconv CUDA package documentation](https://pypi.org/project/spconv-cu120/), and [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/).

## Confirmed blockers

### C1. Direct SparK checkpoints cannot be resumed

**Evidence:** `cli.py` constructs `transformer`, trains it, and its resume branch requires `checkpoint['transformer_state_dict']` to be present. But its call to `save_spark_checkpoint()` at the end of the epoch does not pass `transformer=transformer`. The saving function consequently writes `transformer_state_dict: null`.

**Effect:** A fresh direct `train-spark` run completes an epoch and writes a checkpoint; `train-spark --resume` then raises `RuntimeError: Checkpoint lacks transformer state and cannot be resumed safely.` The learned `BandAdapter` is also absent from the direct checkpoint.

**Required correction:** Pass and persist the transformer on every direct CLI save, then add a real one-epoch direct train/resume test which asserts adapter parameters and optimizer state continue correctly.

### C2. Grid-search best-artifact creation crashes on the first improvement

**Evidence:** `spark_grid_search.py` executes `shutil.copyfile(...)` when `avg_loss <= best_loss`, but does not import `shutil`.

**Effect:** The first finite epoch loss normally meets that condition, so the run is caught by the broad `except`, recorded as `status=error`, and fails to create `spark_unified_best.pt` / `backbone_dense_best.pt`. It may leave partial latest/model checkpoint artifacts, making the run directory look more complete than it is.

**Required correction:** Import the module, use atomic best-artifact handling, and add an integration test that performs one grid epoch with a lightweight real sparse runtime or a verified functional test fixture.

### C3. Dense-to-spconv weight copying is wrong and errors are silently discarded

**Evidence:** `replace_with_sparse_layers()` and the sparse stem call `_copy_parameter_if_available(new_layer, child.weight)`, which attempts a raw `copy_`. TorchVision `Conv2d` uses `OIHW` weights. spconv 2.x documents that its weights are `KRSC`; the project’s pinned package is spconv 2.3.8. The shapes/layouts are incompatible. `_copy_parameter_if_available()` catches `RuntimeError` and returns without reporting failure.

The same issue occurs in reverse: `translate_state_dict()` writes sparse weight tensors directly into a nominal dense ResNet state dict; it neither permutes `KRSC` back to `OIHW` nor validates exact expected keys/shapes.

**Effect:** `pretrained=True` does not reliably initialize the sparse encoder from ImageNet; layer conversion can silently retain random spconv weights. `backbone_dense_latest.pt` and derived best files are not demonstrated to load into a standard dense ResNet and should not be considered usable exports.

**Required correction:** Implement explicit, version-pinned weight transforms with shape assertions; fail on any unmatched expected parameter; test dense→sparse numerical parity on a fully populated input and sparse→dense load/forward parity. Do not call the current artifacts “dense backbone” exports until this passes.

### C4. Sparse ResNet execution has no real-runtime validation and likely residual-index hazards

**Evidence:** `SparKNetwork` invokes `require_spconv()`, and the project `.venv` has no `spconv`. Existing tests use sentinels/mocks and never construct a real `SparseConvTensor`, run a ResNet block, backpropagate, or resume. Standard torchvision residual blocks still use ordinary `identity + out` addition after their layers are recursively replaced. Strided sparse convolutions can change index sets.

spconv’s official documentation says regular addition requires matching indices; `Fsp.sparse_add` is provided for sparse tensors whose indices differ. It also specifies the sparse tensor feature/coordinate contract.

**Effect:** The primary supported sparse ResNet may fail in residual/downsample blocks or produce semantically invalid features as soon as it runs with real spconv. Current passing tests cannot establish that the implementation is SparK.

**Required correction:** Test one real spconv forward/backward on CPU/GPU supported by the selected package, including all ResNet stages and a residual downsample. Replace residual operations with a proven index-safe sparse operation where necessary. Record the exact Torch/CUDA/spconv versions used.

### C5. The stated dependency is not installable in the current Python environment

**Evidence:** The active `.venv` is Python 3.12 and `importlib.util.find_spec('spconv')` is false. The project pins `spconv-cu120==2.3.8`. The official spconv package page documents prebuilt Windows/Linux support for Python 3.7–3.11, CUDA-specific packages, and a CUDA/driver requirement.

**Effect:** `pip install .[spark]` cannot be assumed to yield a runnable sparse environment on the project’s current Python 3.12 setup. The sparse command fails closed, which is preferable to fake results, but the declared workflow is not reproducible as configured.

**Required correction:** Publish a tested environment matrix and installation procedure (Python, OS, Torch, CUDA driver/runtime, spconv package); enforce it at startup. Use an actual compatible environment in CI or state that SparK is unavailable.

### C6. Global package NMS corrupts multi-raster/multi-CRS results

**Evidence:** `_dedupe_records()` in `inference.py` groups candidates only by `class`, constructs polygons from `geometry_native`, and suppresses based on IoU. It ignores source file, dataset identity, and `source_crs`; it does not transform records to a common CRS first.

A read-only synthetic check gave two same-class, identical polygons from `a.tif` / `EPSG:3857` and `b.tif` / `EPSG:4326`. `_dedupe_records(..., 0.5)` retained **one** record. Those coordinate systems are incomparable.

**Effect:** Predictions can disappear across independent source files, across duplicated file basenames, and across different CRSs. Suppression is only meaningful among tiles belonging to the same raster or after a correct common-CRS transformation with source identity retained.

**Required correction:** Partition NMS by a stable source identifier (full path or raster UUID) by default. If cross-raster deduplication is truly required, reproject geometries to a declared common CRS, define overlap eligibility, and test same/different raster and same/different CRS cases.

## High-severity problems

### H1. “Best” model selection uses training reconstruction loss, not validation or downstream quality

Both direct and grid paths choose best artifacts from the lowest per-epoch training reconstruction loss. The direct CLI builds `val_loader` but never uses it. This tends to reward fitting/noise rather than representation quality and is insufficient for a downstream-pretraining claim.

**Required correction:** define held-out SSL validation or a fixed representation probe as the selection criterion; write the metric, split identity, and provenance to a manifest.

### H2. Visualization pass mutates training-mode normalization state

After each direct training epoch, the CLI performs another model/transformer forward pass inside `torch.no_grad()` but leaves both modules in training mode. `no_grad()` prevents gradients, not BatchNorm running-stat updates. Sparse and adapter normalization statistics can therefore be updated on a single visualization batch immediately before checkpointing.

**Effect:** Saved models differ from the optimization state measured by the epoch loss; the effect depends on the last batch and can impair reproducibility.

**Required correction:** temporarily switch relevant modules to evaluation mode for visualization and restore their previous modes, or explicitly accept and document an additional training-statistics update.

### H3. The sparse stem is a new architecture, not pretrained ResNet50

The sparse encoder replaces ResNet’s max-pooling stem with a newly initialized stride-2 sparse convolution plus new normalization. Even if the first convolution conversion were correct, that second convolution has no ResNet pretrained counterpart. `translate_state_dict()` also has no valid mapping for it.

**Effect:** labels such as “ResNet50 pretrained” and any downstream dense checkpoint implication are misleading. This is a custom sparse ResNet-like encoder requiring its own validation and checkpoint format.

**Required correction:** name/version it as a distinct architecture; state exactly what is initialized; avoid claiming architecture or weight parity with torchvision ResNet50.

### H4. Grid search hides a programming failure inside normal result output

`_run_single_experiment()` catches every exception and returns an error record. This is useful for a multi-experiment campaign, but C2 means a code defect is converted to a CSV row while partially saved artifacts remain in the run folder.

**Required correction:** mark a failed run with an explicit failure manifest; prevent consumers from discovering its artifacts; surface programming errors prominently in the top-level command exit status.

### H5. Package inference is no longer memory-streaming at scale

`run_yolo_inference()` saves every record and full polygon JSON in `all_records` before NMS and CSV output. Large imagery, dense detections, masks, and OBBs can exhaust host memory. This is materially different from the standalone runner’s batch-level prediction iterator.

**Required correction:** partition by source raster and use a bounded spatial NMS strategy, an on-disk intermediate store, or tile-neighborhood suppression. Document the memory scaling.

### H6. Invalid-prediction metadata is inaccurate

`yolo_result_to_records()` catches `ValueError` for each invalid polygon and silently skips it. `run_yolo_inference()` increments `invalid_predictions` only if the whole conversion call raises, so the metadata normally reports zero even when individual predictions were discarded.

**Required correction:** return conversion counts/reasons or emit a structured warning/metadata counter per skipped prediction.

### H7. Sparse BatchNorm has an unhandled one-active-feature training failure

`SparseBatchNorm` applies `BatchNorm1d` to `x.features`. In training mode BatchNorm requires more than one value per channel. A small image/batch with high masking or a late sparse stage can have one active feature.

**Required correction:** guard the condition with a defined policy, or validate minimum active coordinates for each stage; add a real sparse test with the minimum allowed crop/batch/mask combination.

## Medium-severity implementation and research issues

### M1. Resume determinism is incomplete in the direct CLI

`--seed` only calls Torch seeding, while data order/worker behavior and Python/NumPy randomness are not seeded there. It also enables `cudnn.benchmark`, which prioritizes performance over reproducible algorithm selection. RNG restoration is better than before but cannot make the initial run deterministic under these settings.

### M2. The direct resume schedule cannot be extended as documented

The code restores the old `schedule_total_epochs` on resume even if `--epochs` is larger. Epochs beyond the original schedule then evaluate cosine progress above 1, causing a second upward segment rather than a normal extended decay. The comment says the curve is continuous when changing from 300 to 600, but the formula is not a valid extended cosine schedule.

### M3. Dead disabled YOLO sparse-conversion code remains after `NotImplementedError`

`SparseCSPDarkNetEncoder.__init__` raises immediately, leaving a large unreachable conversion/monkey-patching implementation below it. It contains architecture-specific assumptions and manual feature additions that must not be mistaken for maintained code.

**Required correction:** remove or move it to an explicitly experimental, non-imported module with tests; keep only the clear fail-closed adapter in production code.

### M4. Documentation, analysis scripts, and YAMLs still preserve invalid SparK-to-YOLO narratives

The training commands now fail closed, but multiple comparison scripts/configs still use `spark_pretrain_yolo11l` paths or describe SparK-pretrained YOLO comparisons. Older reports claim successful transferred YOLO results. Disabled config entries are not enough when surrounding scripts and paper figures can be run or cited independently.

**Required correction:** mark historical results invalid/superseded, remove unsupported SparK variants from executable campaigns, and ensure every report/figure states the architecture and verified transfer provenance.

### M5. Standalone and package inference are different products

The package runner serializes boxes, masks, and OBB polygons, while `scripts/run_naip_detection.py` only consumes `result.boxes` and performs axis-aligned-box NMS. Their device, image sizing, source identity, error accounting, metadata, output schema, and NMS semantics differ.

**Required correction:** designate one supported production path or factor shared geometry/NMS/normalization code and run identical task fixtures through both entry points.

### M6. Cross-tile NMS has no CLI policy control in package inference

The API defaults to NMS at 0.5 but the CLI does not expose an IoU threshold or a way to disable it. Given C6, users cannot safely choose per-raster-only behavior through the command line.

### M7. Input-band meaning is not captured in inference provenance

The package takes the first three bands (or replicates fewer) but metadata does not record source band indices/order, dtype, scale/offset, CRS policy, or model class mapping. Fixed normalization is an improvement, but it is not enough to reproduce an aerial-imagery inference result.

### M8. Tests assert source text and mocks rather than runtime contracts

The 13 focused SparK/regression tests pass, but several assertions inspect source strings or mock spconv availability. They do not validate the C1 transformer checkpoint contract, C2 grid artifact path, actual spconv weight layout, residual behavior, dense export loading, or multi-raster NMS isolation.

## Checks performed

- Project `.venv`: Python 3.12.13; Torch/TorchVision/Ultralytics/Shapely/pytest present; `spconv` absent.
- Focused SparK/YOLO regression run: **13 passed**.
- Focused inference/CLI run: **22 passed, 3 skipped**.
- Reviewed modules compile successfully.
- Synthetic different-file/different-CRS NMS check: **1 of 2 records retained**, confirming C6.

Passing checks demonstrate only the tested guard/synthetic behavior. They do not execute a real sparse model because its runtime is absent.

## Recommended repair order

1. Repair C1 and C2, then add direct and grid one-epoch/resume integration tests.
2. Establish a reproducible, compatible spconv environment. Do not use surrogate/mocked tests as evidence of sparse execution.
3. Replace raw/silent weight copying with explicit tested `OIHW ↔ KRSC` conversion and fail-closed coverage checks; prove numerical parity.
4. Validate the entire sparse ResNet forward/backward including residual additions, then decide whether its custom stem supports any pretrained claim.
5. Fix NMS partitioning/reprojection before using multi-file results; add source/CRS tests.
6. Define a valid downstream research path. Current code deliberately has no verified SparK-to-YOLO adapter, so no comparison should be labelled as such.
7. Consolidate inference entry points and make provenance, memory limits, model task, band policy, and NMS behavior explicit.

## Source notes

The official [spconv usage guide](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md) specifies sparse feature and coordinate formats and calls out `Fsp.sparse_add` for different sparse index sets. The official [spconv CUDA package page](https://pypi.org/project/spconv-cu120/) documents its CUDA/Python support constraints and `KRSC` layout for spconv 2.x weights. The [official SparK repository](https://github.com/keyu-tian/SparK) is the relevant baseline for sparse hierarchical masked modeling, while [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/) is the relevant result/task API reference.
