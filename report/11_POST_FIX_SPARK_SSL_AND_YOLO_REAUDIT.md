# Post-Fix SparK SSL and YOLO Re-Audit

**Date:** 2026-08-13  
**Scope:** Fresh read-only review of the current SparK SSL, downstream transfer, YOLO/OBB/segmentation inference, launchers, configuration, and focused tests. No implementation files were edited for this audit.

## Executive conclusion

The current code is safer than the first audit in important ways: the dense fake-SparK fallback is no longer used, unsupported YOLO sparse encoders are rejected by the main network, the grid checkpoint call now has the right parameter contract, and package inference can serialize boxes, masks, and OBB geometries.

However, the project is still **not ready to claim usable SparK pretraining or SparK-to-YOLO downstream results**. The active environment has no `spconv`, it is not declared as a project dependency, and there is no real-spconv forward/backward/resume test. More importantly, the only supported SparK encoder is ResNet50 while the research scripts and downstream comparators still expect YOLO11 weights. Several of those scripts directly perform a zero-coverage-tolerant `strict=False` load and can still train a normal YOLO model under a SparK label.

A separate CLI regression is severe: `args.normalization_min` / `args.normalization_max` are used 22 times, while only `infer-yolo` defines those options. At least 21 unrelated commands will fail with `AttributeError` if they reach these calls; many also pass an unsupported keyword to their target function.

Severity definitions:

- **Critical:** invalidates an experiment, silently uses the wrong weights/model, or makes a documented command unusable.
- **High:** likely runtime failure, materially wrong results, or a serious reproducibility problem.
- **Medium:** important robustness, documentation, performance, or maintainability weakness.

## Evidence and methods

Reviewed the active source and launch paths, including:

- `src/poultry_geo/spark.py`, `spark_models.py`, `train_spark_ssl.py`, `spark_grid_search.py`, and `cli.py`
- `src/poultry_geo/inference.py`, `src/downstream_finetune/finetune.py`
- `scripts/train_spark.ps1`, `train_spark.sh`, `downstream_compare.py`, `downstream_fewshot_poultry.py`, `downstream_scratch_compare.py`, `prepare_and_train_dota.py`, `prepare_and_train_grsd.py`, and `run_naip_detection.py`
- current SparK/downstream YAML files and focused tests

Read-only checks found:

```text
spconv_spec = None
pyproject.toml declares ultralytics>=8.3 but no spconv dependency
args.normalization_min uses in cli.py = 22
--normalization-min parser declarations = 1
focused SparK/inference/transfer tests = 14 passed
```

The 14 passing tests prove guard and serialization behavior with mocks/small synthetic data. They do **not** exercise a real `spconv` tensor, a real sparse ResNet forward/backward pass, a grid run, an actual GeoTIFF plus Ultralytics result, or a real downstream transfer.

Compared assumptions against the official [SparK repository](https://github.com/keyu-tian/SparK), [spconv usage documentation](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md), and [Ultralytics prediction documentation](https://docs.ultralytics.com/modes/predict/). The spconv documentation specifies `[N, ndim + 1]` indices with batch first, documents sparse addition as requiring compatible indices, and explicitly distinguishes sparse convolution from submanifold convolution. The present transformer uses the correct 2-D index shape `[N, 3]`, but the remaining residual-conversion behavior is not validated.

## Critical findings

### C1. No usable sparse runtime is installed or declared

**Evidence:** `importlib.util.find_spec("spconv")` returns `None`; neither `pyproject.toml` nor the reviewed requirements files declare `spconv`. The main SparK constructor correctly fails closed, so every direct CLI/grid run currently stops before training.

**Impact:** No current environment can execute the advertised SparK workflow. This is safer than silently using dense zero-filled inputs, but it means there is no runnable experiment.

**Required correction:** Declare and pin a supported `spconv` distribution by OS, CUDA, PyTorch, and Python version; add installation validation; record the resolved package/CUDA versions in run metadata; and fail during CLI preflight before data loading.

### C2. The supported SparK encoder cannot be transferred into YOLO, but several launchers still try

**Evidence:** `SparKNetwork` only accepts `resnet50`. `load_spark_backbone_into_yolo()` correctly refuses it. In contrast, `downstream_fewshot_poultry.py`, `downstream_scratch_compare.py`, `prepare_and_train_dota.py`, and `prepare_and_train_grsd.py` still load a supposed YOLO11 SparK state dict directly into `YOLO("yolo11l-obb.pt")` with `strict=False`.

**Impact:** These scripts bypass the guarded transfer path. A ResNet dense export has no expected YOLO keys, so their `patch` can be empty; the script then continues with an ordinary pretrained YOLO and labels the result as SparK. This is the exact false-comparison failure the first audit identified.

**Required correction:** Remove or disable every direct injection helper until a version-pinned YOLO sparse adapter exists. Centralize transfer in one loader that requires an architecture identity and a nonzero, expected-module coverage threshold before training starts.

### C3. Multiple legacy training wrappers are immediately incompatible with the current CLI

**Evidence:** `scripts/train_spark.ps1` defaults to `--backbone-type yolo11l`, uses a `cspdarknet` example, and defaults its output folder to `spark_pretrain_yolo11l`. The CLI now only allows `resnet50`. `train_spark.sh` retains the invalid CSP example and its help claims support for YOLO CSPDarkNet.

**Impact:** A documented PowerShell launch fails argument parsing without starting a run. Users can also believe they are initiating a YOLO SparK experiment when the supported implementation intentionally does not provide one.

**Required correction:** Update wrappers, defaults, output naming, help text, and examples to ResNet50 only, or remove the wrappers while the workflow is unavailable.

### C4. Legacy downstream scripts still reference nonexistent `backbone_dense_ep299.pt` artifacts

**Evidence:** The direct downstream scripts above retain hard-coded paths ending `backbone_dense_ep299.pt`. The current trainer writes `backbone_dense_latest.pt`; the direct CLI copies `backbone_dense_best.pt` only when a best training-loss epoch is observed; grid training does not create `backbone_dense_best.pt` at all.

**Impact:** These scripts either fail before training or point at stale artifacts from the invalid former YOLO workflow. Changing YAML references alone did not repair these Python launchers.

**Required correction:** Define one artifact manifest and use it everywhere. Export a named final/best artifact in both direct and grid modes, with architecture/type metadata. Do not encode an epoch number in filenames unless those epoch artifacts are actually retained.

### C5. The CLI has a broad runtime regression from undefined normalization arguments

**Evidence:** `cli.py` contains 22 references to `args.normalization_min`, but only the `infer-yolo` parser declares `--normalization-min` and `--normalization-max`. The references were added to `sam2-visualize`, DINO visualization, Mississippi pretraining/evaluation, grid training, DETR, Mask2Former, model export, and other unrelated branches.

**Impact:** On those commands, `argparse.Namespace` has no such attributes. If the branch reaches the call, it raises `AttributeError`; several targets also do not accept `normalization_range`, which would then raise `TypeError`. This breaks CLI workflows beyond YOLO and makes the command dispatcher unreliable.

**Required correction:** Remove the unrelated arguments. For commands that genuinely need fixed raster scaling, define their own parser options and update the target function signature/tests together. Use `getattr(args, ..., None)` only as a temporary compatibility bridge, not as the final interface contract.

## High findings

### H1. There is no real-spconv integration test, and sparse residual behavior is unproven

**Evidence:** The focused tests pass with `spconv` absent or replaced by simple objects; none instantiate an actual `SparseConvTensor`. `replace_with_sparse_layers()` converts ResNet branches independently and residual blocks use ordinary `x + out` rather than a documented sparse-add operation. The official spconv documentation cautions that sparse tensors with different indices cannot be summed by regular addition.

**Impact:** After installing spconv, the first real ResNet forward can fail at residual additions or silently use incorrect coordinate behavior. This is a core correctness risk, not just test coverage.

**Required correction:** Build a pinned real-spconv smoke environment. Test every ResNet stage and residual downsample block for indices, shapes, gradients, mixed precision, and a short optimizer step. Use a deliberate coordinate-map/indice-key design and a sparse-add strategy when index sets differ.

### H2. Sparse ResNet is not architecture/weight equivalent to torchvision ResNet50

**Evidence:** The sparse stem replaces the original ResNet `MaxPool2d` with a learned `SparseConv2d`; it adds a second learned stride-2 convolution. Therefore the stem has different parameters and behavior. `SparseBatchNorm` improves on the former LayerNorm substitution, but it applies BatchNorm1d to active sites rather than the original dense spatial BatchNorm2d population.

**Impact:** A `pretrained: true` run is only partially initialized from ImageNet. Dense export cannot faithfully reconstruct a standard ResNet50 stem, and downstream comparisons must not call it an architecture-preserving checkpoint.

**Required correction:** Document it as a new sparse ResNet variant with its own checkpoint type. Provide an explicit, measured initialization policy and only export/load tensors whose keys and semantics are verified.

### H3. Direct CLI resume still overwrites restored LR schedule anchors

**Evidence:** The resume branch assigns `schedule_total_epochs` and `schedule_warmup_epochs` from the checkpoint. Later in the same command, the training loop unconditionally assigns both values from `args.epochs` and the newly computed warmup value.

**Impact:** Resumed runs change the learning-rate trajectory despite the code/comments claiming continuity. It invalidates comparisons across interrupted vs uninterrupted runs.

**Required correction:** Initialize schedule anchors before resume, restore them conditionally, and do not reassign them later. Persist/restore the exact scheduler state rather than reconstructing a schedule from inferred values.

### H4. Grid output and configured downstream artifact names still disagree

**Evidence:** Grid calls `save_spark_checkpoint`, which writes `backbone_dense_latest.pt`. It does not calculate a best epoch or write `backbone_dense_best.pt`. Several config files now point to `backbone_dense_best.pt`.

**Impact:** A grid-trained checkpoint cannot be consumed via the configured best-artifact path. This remains a deterministic integration failure, even after replacing `ep299` strings in some YAMLs.

**Required correction:** Either create `backbone_dense_best.pt` in grid mode based on a defined selection metric, or configure consumers to use a single `backbone_dense_latest.pt` only after a completed run. Prefer a manifest over hard-coded paths.

### H5. The sparse target adapter and checkpoint semantics remain under-specified

**Evidence:** `train_one_step()` uses adapter output as both the visible sparse input and a detached reconstruction target. The adapter still receives gradients through the visible-path encoder, but no direct target gradient. The checkpoint stores transformer state now, which is good; no test proves optimizer-state restoration has the same parameter ordering or that resumed output matches an uninterrupted run.

**Impact:** The learned band projection can drift jointly with the encoder while defining its own target representation. This may be intentional, but it is not documented or ablated. Resume reproducibility is unverified.

**Required correction:** Decide whether the adapter is a fixed preprocessing transform, a trainable encoder input projection, or a separately regularized component. Add a deterministic two-step resume equivalence test including optimizer and RNG state.

### H6. Package YOLO inference still has no cross-tile duplicate suppression

**Evidence:** `run_yolo_inference()` accepts independent windows and writes records per tile, but does not run NMS after aggregating overlapping tiles. The standalone script has class-aware global box NMS, while package inference has none.

**Impact:** Any `stride < window_size` run can emit repeated detections at tile boundaries. For OBB and segmentation, box-only NMS is insufficient anyway; task-aware geometry NMS is required.

**Required correction:** Add class-aware deduplication in a shared inference module: box IoU for detection, polygon IoU for masks/OBB, with tests over overlapping windows.

### H7. Invalid segmentation geometry aborts the whole package run

**Evidence:** `_record()` constructs a Shapely `Polygon` and raises `ValueError` if it is invalid. `run_yolo_inference()` does not catch a bad individual prediction.

**Impact:** One malformed/self-intersecting mask can terminate a large GeoTIFF campaign after substantial compute. Ultralytics masks and pixel contour transformations can legitimately produce problematic polygons.

**Required correction:** Skip and count invalid predictions, log tile/prediction identity, and continue. Keep a strict mode only when requested.

### H8. Non-uint8 input handling is safer but operationally incomplete

**Evidence:** Package and standalone inference now reject non-uint8 imagery unless a fixed range is supplied. The package CLI exposes min/max only for `infer-yolo`; the range is not persisted into the output CSV or an inference manifest.

**Impact:** This prevents per-tile contrast distortion, but a rerun cannot establish which raw-value mapping produced the predictions. A user may choose a range that clips most NAIP values without an audit trail.

**Required correction:** Store source dtype, selected bands, global range, clipping percentages, source CRS, model checksum, and command options in a sidecar JSON/CSV metadata file. Support fixed percentile/raster-statistics modes computed once per source dataset.

## Medium findings

### M1. YOLO version is not pinned

`pyproject.toml` permits `ultralytics>=8.3`; the earlier audit environment had 8.4.63. YOLO result APIs and architecture graphs are version-dependent. Pin a tested range/exact version and record it in checkpoints and inference metadata.

### M2. The code retains dead, unsafe sparse YOLO conversion code

`SparseCSPDarkNetEncoder` contains the old dynamic Ultralytics graph conversion and C2PSA patch logic after an immediate `NotImplementedError`. It cannot be reached via `SparKNetwork`, but leaving it in source invites direct use and obscures supported behavior.

### M3. SparK documentation is stale

Docstrings/help still say ResNet or CSPDarkNet/YOLO. `translate_state_dict()` also describes Ultralytics YOLO export even though it raises for YOLO. This makes the public interface misleading.

### M4. Direct CLI has no explicit `--resume` or seed contract

It resumes automatically when a latest checkpoint is found. It does not expose a seed, does not save/restore RNG state in `spark_unified_latest.pt`, and has no way to start fresh in an existing output directory. Grid `StateManager` has stronger RNG handling, making the two entry points semantically inconsistent.

### M5. Best checkpoint selection uses noisy training reconstruction loss only

The direct command creates validation data but does not evaluate it. The best model is selected from randomly masked training batches, so it is not a stable measure of representation quality.

### M6. Sparse dependency errors arrive late in the direct CLI

The direct CLI constructs dataloaders before building the sparse model. On a missing-spconv system it may scan/create dataset state before reporting the unavailable core dependency. A preflight check would be clearer and safer.

### M7. Package inference re-scans the raster directory for every progress update

It calls `find_geotiffs(geotiff_dir)` once for iteration and again inside the progress callback. This is unnecessary recursive filesystem work and can make progress totals change if files appear mid-run.

### M8. CSV field semantics remain ambiguous for legacy geometry consumers

The compatibility field `mask_points_gis` now stores coordinates in the source/native CRS, while legacy downstream exporters can have independent default CRS assumptions. Consumers need explicit `source_crs` handling rather than relying on column names.

### M9. Tile band selection is implicit

Both inference paths use the first three raster bands (or repeat bands) without recording their semantic order. NAIP conventions, BGR imagery, and data with NIR-first ordering can therefore be fed incorrectly to YOLO.

### M10. Existing broad tests do not establish repository health

The last full suite run in this workspace had 11 failures outside these focused files (progress interface, NAIP dataset ordering, an unrelated SSL loss test, Mississippi retrieval, and ResNet artifact expectations). The new 14 focused tests pass, but they should not be presented as a clean repository-wide validation.

## What improved since the first audit

- The missing sparse engine no longer silently falls back to a dense zero-filled tensor.
- `SparKNetwork` rejects unsupported YOLO/CSP names rather than silently using YOLO11-nano.
- Transformer state is now saved in direct/grid checkpoints.
- Grid training includes transformer parameters and uses dataloader length instead of hard-coded 4,000 batches.
- Package inference supports detection, segmentation, and OBB records and emits explicit native/WGS84 coordinate fields.
- Package inference writes results incrementally rather than retaining all records in memory.
- Standalone box NMS is class-aware.

## Required repair order

1. Decide the supported research claim. Either deliver a real, pinned sparse ResNet SparK workflow or remove SparK training commands until it can run.
2. Remove/disable every YOLO SparK downstream launcher and direct `strict=False` injection helper. Do not report prior results from those routes as SparK transfer.
3. Fix the 22 CLI normalization argument regressions immediately; add dispatcher tests for every affected command.
4. Add a real-spconv test environment and verify residual additions, coordinate maps, AMP, one optimizer step, direct resume, and one-epoch grid run.
5. Unify artifact naming/metadata across direct/grid training and consumers. Eliminate every epoch-specific stale path.
6. Repair direct schedule resume, add seed/RNG controls, and define a validation/selection metric.
7. Add shared task-aware cross-tile NMS and invalid-geometry tolerance to package inference.
8. Pin Ultralytics and persist model/version/band/radiometry/CRS provenance alongside every inference result.

## Verification status

This audit intentionally did not train models, install dependencies, download weights, or edit implementation code. Focused tests passed (`14 passed`), syntax checks passed for reviewed SparK/YOLO scripts, and static checks confirmed the absent sparse runtime, 22:1 CLI normalization mismatch, invalid wrappers, stale artifact references, and remaining direct transfer bypasses.