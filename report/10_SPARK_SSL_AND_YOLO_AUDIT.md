# SparK SSL and YOLO Audit

**Date:** 2026-08-13  
**Scope:** Read-only audit of SparK masking/encoder/decoder/training/grid-search/checkpoint/transfer code and the project’s Ultralytics YOLO training and GeoTIFF inference paths. No implementation code was changed for this audit.

## Executive conclusion

The repository has useful experimental building blocks, but the present SparK-to-YOLO workflow is not reliable enough to support claims that a downstream model was sparse-masked pretrained. The main blocker is fundamental: this environment has no `spconv`, so the code uses a dense zero-filled fallback rather than SparK’s sparse-convolution mechanism. There are also guaranteed grid-search failures, silent backbone substitutions, stale/incompatible checkpoint paths, and transfer loaders that can report success while loading no useful weights.

The two YOLO inference implementations are inconsistent. The package CLI inference only emits records for segmentation masks, while the separate NAIP detection script emits only axis-aligned boxes. Both normalize each tile independently; this can materially change imagery presented to a trained model. The package inference additionally labels arbitrary projected coordinates as latitude/longitude.

Severity definitions:

- **Critical:** an advertised workflow fails, silently uses the wrong model/weights, or invalidates an experiment.
- **High:** likely wrong behavior or a substantial reproducibility/measurement risk.
- **Medium:** robustness, portability, or maintainability problem that can undermine results.

## Evidence and verification

Reviewed:

- `src/poultry_geo/spark.py`, `spark_models.py`, `train_spark_ssl.py`, `spark_grid_search.py`
- `src/poultry_geo/cli.py`, `checkpoint_manager.py`, `models.py`
- `src/downstream_finetune/finetune.py`, `lpft_llrd_trainer.py`
- `src/poultry_geo/inference.py`, `training.py`, `ultralytics_runtime.py`
- `scripts/run_naip_detection.py`, `scripts/run_downstream_finetune.py`, SparK/YOLO configs and wrappers

Focused runtime checks on the active environment:

```text
spconv = None
ultralytics = 8.4.63
```

The checked grid-search call is:

```text
train_spark_ssl.save_spark_checkpoint(model, spark_config, str(manager.run_dir), epoch_index)
```

but the actual function signature is:

```text
save_spark_checkpoint(model, optimizer, scaler, config, path, epoch, ...)
```

Focused test collection fails before SparK behavior can be exercised:

```text
tests/test_spark_models.py: IndentationError at line 106
tests/test_spark_pipeline.py: ImportError: PatchMasker is not exported by poultry_geo.spark
```

Compared algorithm and framework assumptions with the official [SparK implementation](https://github.com/keyu-tian/SparK), the [SparK paper](https://arxiv.org/abs/2301.03580), [spconv usage documentation](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md), and the official [Ultralytics architecture guide](https://docs.ultralytics.com/guides/yolo-architecture). Official SparK is specifically sparse/hierarchical masked modeling for convolutional networks; spconv requires indices of shape `[N, ndim + 1]` with batch index in the first column; Ultralytics architectures are version/YAML-dependent backbone–neck–head graphs, not a stable universal list of layer indices.

## Critical findings

### C1. Current SparK runs are dense fallback runs, not sparse-convolution SparK

**Evidence:** `spconv` is not installed in the active environment and is not declared in `pyproject.toml` or `requirements-ml.txt`. `spark.py` returns a tuple when `spconv` is unavailable; `SparseResNetEncoder` and `SparseCSPDarkNetEncoder` reconstruct a dense zero-filled tensor from that tuple and run ordinary dense layers.

**Impact:** The central SparK claim—avoiding computation on masked inputs via sparse convolutions—is false for current runs. The fallback also changes model semantics: masked regions are numeric zeros propagated through ordinary dense convolutions rather than absent sparse sites. Loss curves and downstream weights from this route should not be described as SparK results.

**Required correction:** Make sparse execution an explicit hard requirement for the SparK command, or rename/split the fallback as a separately implemented dense masked-autoencoder baseline. Pin and test a supported `spconv` build for the target OS/CUDA/PyTorch combination.

### C2. SparK grid-search training fails at the first epoch checkpoint

**Evidence:** `spark_grid_search._run_single_experiment()` calls `save_spark_checkpoint(model, spark_config, str(manager.run_dir), epoch_index)`. The function requires `model, optimizer, scaler, config, path, epoch`.

**Impact:** Each grid experiment reaches its checkpoint stage and raises `TypeError` for missing arguments. The exception is caught by the runner, so the result can be written as an `error` record rather than producing usable pretrained weights. This invalidates all grid configurations as presently executed.

**Required correction:** Use one checkpoint authority in the grid path. Either call `StateManager.save_checkpoint` only and export a dense backbone through a dedicated helper, or call `save_spark_checkpoint` with the optimizer/scaler/config/path/epoch contract. Add an end-to-end one-epoch grid smoke test that asserts checkpoint and exported backbone files exist.

### C3. `yolov26l` is silently trained as YOLO11-nano in SparK

**Evidence:** The CLI accepts `yolov26l`; `configs/spark_grid_search_subset.yaml` also includes it. `SparKNetwork` only recognizes `backbone_type.startswith("yolo11")`; every other non-ResNet value falls through to `SparseCSPDarkNetEncoder(..., backbone_type="yolo11n")`.

**Impact:** A run folder/config/claim labelled `yolov26l` is actually a YOLO11-nano encoder. Any YOLO26 comparison, transfer result, or compute interpretation is invalid.

**Required correction:** Reject unsupported families at construction. Implement a version-pinned YOLO26 adapter only after validating its graph and dense transfer mapping, or remove every YOLO26 option/config/claim.

### C4. SparK “scratch” experiments still load pretrained YOLO classification weights

**Evidence:** `SparseCSPDarkNetEncoder` always calls `YOLO(f"{backbone_type}-cls.pt")`, regardless of `pretrained`. `SparKNetwork` does not pass or honor `pretrained` for a YOLO encoder. `configs/spark_scratch.yaml` claims `pretrained: false` is a random-initialization ablation.

**Impact:** The scratch ablation begins from an ImageNet-pretrained classification checkpoint, not random weights. Comparisons marketed as “SparK trained from scratch” or as proof of domain-only SSL are invalid.

**Required correction:** For scratch runs, instantiate the exact architecture from a YAML definition without `.pt` weights. Record a weight fingerprint/source in metadata and test that scratch and pretrained initial states differ as intended.

### C5. Downstream SparK checkpoint paths and formats are incompatible with what the trainer produces

**Evidence:** `save_spark_checkpoint()` writes `backbone_dense_latest.pt` and the CLI copies `backbone_dense_best.pt`; it never writes `backbone_dense_ep299.pt`. Multiple downstream configs/scripts reference `backbone_dense_ep299.pt`. `configs/downstream_finetune.yaml` instead points at a generic `model_checkpoint.pt`, which contains the sparse model under `model_state_dict`.

`downstream_finetune.finetune.load_spark_backbone_into_yolo()` can load both forms, but its full-checkpoint extraction creates keys such as `backbone_layers.0...` without applying the later dense-export translation to `model.0...`. The fallback only accepts exact matching keys; therefore it can load zero tensors. It never rejects an empty or low-coverage transfer.

**Impact:** Common configured downstream jobs either fail because a referenced file does not exist or proceed with little/no SparK initialization while reporting “SparK weights loaded.” This directly invalidates transfer-learning results.

**Required correction:** Define one versioned export artifact and use it everywhere. Make load coverage mandatory: require mapped tensors for every intended backbone block and fail if coverage is insufficient. Update configs only after the artifact contract is verified on the exact YOLO task/family.

### C6. Package `infer-yolo` drops all ordinary detection and OBB outputs

**Evidence:** `poultry_geo.inference.yolo_result_to_records()` returns immediately when `result.masks is None`. Standard detection and OBB models provide boxes without segmentation masks. The README/CLI present `infer-yolo` as generic YOLO inference, and the repository contains OBB configs.

**Impact:** Running the package inference command on detection or OBB weights produces an empty CSV even when model detections exist. This is a silent data-loss failure in the primary pipeline.

**Required correction:** Implement separate, explicitly tested record paths for boxes, segmentation polygons, and OBB polygons; validate model task at load time and never treat missing masks as no detections.

## High findings

### H1. The sparse YOLO11 `C2PSA` implementation becomes type-incompatible after conversion

**Evidence:** `_patch_yolo_modules()` patches `C2PSA.forward` before `replace_with_sparse_layers()` converts child convolutions. At runtime for a sparse input it calls `dense_x = x.dense()` and then `self.cv1(dense_x)`. After conversion, `self.cv1` is a sparse convolution that expects a `SparseConvTensor`, not a dense tensor.

**Impact:** A real spconv-backed YOLO11 path is likely to fail at C2PSA. If modified to run, its dense attention step would require an explicit and validated sparse-to-dense-to-sparse policy, including treatment of masked positions.

**Required correction:** Do not monkey-patch a changing third-party graph in place. Implement and unit-test a version-pinned sparse adapter for every supported module, with a real spconv forward smoke test.

### H2. “100% architectural and weight parity” is not true for the sparse encoder

**Evidence:** The code replaces every BatchNorm2d with LayerNorm and replaces ResNet’s max pool with a strided sparse convolution. Those operations are not equivalent, and only a subset of ResNet pretrained stem weights is copied. Ultralytics blocks are dynamically monkey-patched and truncated at hard-coded `[:10]` indices.

**Impact:** The sparse model is neither architecture- nor weight-identical to its dense source. A transfer loader that assumes parity has no validated basis, and any comparison needs to treat it as a new architecture.

**Required correction:** Remove parity claims. Validate each dense-to-sparse translation numerically on unmasked inputs, document every intentional architectural substitution, and report actual loaded-parameter coverage.

### H3. The grid-search path does not optimize the band adapter and uses the wrong autocast device on CPU

**Evidence:** Grid-search constructs its optimizer from `model.parameters()` only, excluding `transformer.adapter.parameters()`. Its loss target is `transformer.adapter(batch).detach()`, so a multiband adapter starts random and remains fixed. The direct CLI includes adapter parameters, creating inconsistent objectives between commands.

The grid creates `SparKConfig` without `device_type=resolved_device.type`; its default is `cuda`. A CPU run can enter autocast using a CUDA device type.

**Impact:** Non-RGB grid experiments train against an arbitrary frozen projection; CPU/auto configurations are unreliable. Results are not comparable with direct CLI runs.

**Required correction:** Optimize and checkpoint the transformer/adapter as part of the training state; pass the resolved device type; add RGB and multiband CPU/GPU grid smoke tests.

### H4. Direct SparK resume does not restore the transformer/adapter state or its optimizer semantics

**Evidence:** The direct CLI checkpoint stores `model_state_dict`, optimizer and scaler, but not `transformer.state_dict()`. On resume a newly initialized `BandAdapter` is used, even though it is included in the optimizer and defines the reconstruction target. The resumed optimizer may contain state for prior adapter parameters while the active adapter is new.

**Impact:** Multiband runs change target representation at resume and cannot be considered continuous or reproducible. RGB runs are less affected but still lack a complete artifact contract.

**Required correction:** Save/restore model, transformer, optimizer, scaler, scheduler anchors, random state, data fingerprint, and config signature together. Reject incompatible resume.

### H5. The grid ignores data length and configured epoch sizing, running a hard-coded 4,000 batches per epoch

**Evidence:** `_train_one_epoch()` sets `num_batches = 4000` unconditionally and consumes `_infinite_loader`. It ignores regular dataloader length and has no grid config `epoch_size` control.

**Impact:** “Epoch” does not have the documented dataset meaning. Training budget, LR scheduler progression, sample exposure, and comparisons change when dataset/catalog size changes. This is especially serious for the 300-epoch configs.

**Required correction:** Define epoch semantics explicitly and configure `steps_per_epoch`; persist it in the experiment identity and scheduler state. Otherwise use actual dataloader length.

### H6. Main package inference and standalone NAIP inference disagree on task support and NMS behavior

**Evidence:** `inference.py` requires masks and only supports segmentation polygons. `scripts/run_naip_detection.py` reads only `result.boxes.xyxy`, so it discards segmentation masks and OBB orientation. The standalone script uses a class-agnostic `torchvision.ops.nms`; detections of different classes can suppress one another.

**Impact:** Identical weights/images can yield materially different outputs depending on entry point. Multi-class results may lose valid detections, and OBB/segmentation geometry is discarded in the standalone path.

**Required correction:** Consolidate inference or establish one shared task-aware postprocessor. NMS must be class-aware unless cross-class suppression is an explicit project rule; support OBB and masks where their models are accepted.

### H7. Both GeoTIFF inference paths independently normalize each tile, changing radiometry at tile boundaries

**Evidence:** `inference._normalize_tile_to_pil()` and `scripts/run_naip_detection.array_to_pil()` min–max normalize each raster window independently. The same physical roof pixel can be represented with different RGB values in neighboring windows.

**Impact:** This creates tile-boundary prediction inconsistency and shifts input distribution relative to JPG/PNG training. It can cause duplicate/missing detections and invalidate comparisons across imagery with different dynamic range.

**Required correction:** Use an explicit sensor-aware radiometric transform (for example documented NAIP uint8 band selection or fixed percentile/statistical scaling) consistently in training and inference. Persist it with model metadata.

### H8. Package geospatial output mislabels projected coordinates as latitude/longitude

**Evidence:** `yolo_result_to_records()` applies the tile affine transform and writes the resulting values to `longitude` and `latitude` without checking or transforming the source CRS. Most NAIP GeoTIFFs use projected CRSs, not EPSG:4326.

**Impact:** CSV consumers can interpret meters/feet as degrees and place farms in the wrong location. Bounding-box columns use the same misleading naming.

**Required correction:** Store native `x/y` plus `source_crs`; transform to WGS84 with `pyproj` only when requested, then label outputs longitude/latitude. Add a projected-CRS integration test.

## Medium findings

### M1. SparK input and mask validation is absent

`SparseTransformer` does not validate rank, positive patch size, mask ratio range, or image dimensions divisible by patch size. For non-divisible dimensions, it samples only the floor patch grid then resizes that grid over the entire image, producing irregular border masks and a mismatch between mask semantics and sparse coordinates.

### M2. The direct CLI’s LR-resume anchors are overwritten after restoration

The CLI reads saved `schedule_total_epochs` / `schedule_warmup_epochs`, then later unconditionally sets both from current arguments before the training loop. The comments claim continuous scheduling after changing `--epochs`, but the implementation discards the loaded anchors.

### M3. “Best” SparK checkpoints select noisy training reconstruction loss only

The direct CLI rebuilds a random mask every optimization step and tracks best loss on training batches. It never evaluates the created validation loader. The grid similarly has no validation/model-selection behavior. This is weak evidence for selecting an SSL checkpoint.

### M4. Training identity/configuration is inconsistent across entry points

Direct CLI always resumes if `spark_unified_latest.pt` exists but exposes no `--resume` control; grid state uses `StateManager`. They save different artifact layouts and configuration content. Results cannot be reproduced or compared reliably across commands.

### M5. Test suite tests mocks and stale APIs rather than real SparK contracts

`test_spark_models.py` cannot parse due to a duplicated function/decorator. `test_spark_pipeline.py` imports removed `PatchMasker`, expects 4D sparse indices despite the current 2D spconv representation, and expects an old `generate_mask` return shape. Existing SparK tests therefore do not protect the current code. No test runs real spconv, a full one-epoch grid run, checkpoint resume with adapter, YOLO11/YOLO26 architecture verification, or nonzero transfer coverage.

### M6. Fixed Ultralytics layer slicing is fragile across task, family, and release

The encoder assumes classification backbone modules are `model[:10]` and captures `[2,4,6,9]`. Official Ultralytics documentation describes a YAML-defined graph whose modules and scaling differ across families and versions. `ultralytics>=8.3` is unpinned; active version is 8.4.63.

### M7. YOLO transfer loaders report missing/unexpected keys but do not measure matched coverage

`load_spark_backbone_into_yolo()` and related launchers use `strict=False` and continue after empty/partial mappings. A “loaded” message is not evidence that pretraining reached a downstream model.

### M8. Pipeline memory claim is overstated for detection results

`run_yolo_inference()` streams input tiles but accumulates every detection record in `all_records` before writing CSV. Very large, dense runs can still exhaust RAM due to output accumulation.

### M9. Decoder geometry only matches its intended four-stage stride hierarchy

`HierarchicalDecoder` assumes four features with compatible factor-of-two spatial transitions and final factor-four upsampling. It has no shape checks or interpolation safeguards. Arbitrary input size, future Ultralytics graph, or changed stage selection can fail at skip additions or yield wrong output size.

### M10. LP-FT/LLRD assumes YOLO11l layer numbering and is not integrated with the standard fine-tuning paths

`LPFT_LLRD_OBBTrainer` hard-codes layers 0–10 as backbone and is only useful if a caller explicitly supplies the custom trainer. The reviewed generic `finetune_with_config` and launcher do not do that. Configuration labels such as `use_lpft_llrd: true` in few-shot configs therefore are not evidence the method runs.

## What is sound or directionally useful

- Patch-level Bernoulli-style mask selection and masked-only normalized-patch reconstruction loss are reasonable components of a masked-image objective.
- The 2D spconv coordinate convention in `SparseTransformer`—batch index plus `(y, x)`—matches spconv’s documented `[N, ndim+1]` index requirement.
- The hierarchical ResNet decoder geometry is internally coherent for a 224×224 input with the specific stride hierarchy.
- `StateManager` has useful atomic checkpoint and RNG-restoration primitives, but the SparK grid path currently fails before it can provide an end-to-end guarantee.
- Generic `train_yolo_models` correctly delegates model family loading through `ultralytics_runtime` and has basic unit coverage for RT-DETR routing.

## Recommended repair order

1. Stop treating current SparK/downstream results as valid sparse-pretraining evidence; record whether existing runs used `spconv`.
2. Repair test collection, then add a real-spconv smoke test and make sparse execution mandatory for the SparK command.
3. Fix the grid checkpoint call and unify direct/grid artifact and resume contracts, including transformer state and coverage-checked dense export.
4. Remove unsupported YOLO26/CSP aliases until individually implemented and version-pinned; fix scratch initialization so it is genuinely random.
5. Replace hard-coded Ultralytics module indexing with task/family-specific adapters verified against a pinned Ultralytics model YAML/state dictionary.
6. Make SparK-to-YOLO transfer fail closed: exact architecture identity, nonzero expected-module coverage, and a numerical round-trip test.
7. Consolidate task-aware inference: boxes, segmentation, and OBBs; class-aware NMS; fixed radiometric preprocessing; explicit CRS output.
8. Add deterministic validation/probe selection and persist all schedule/data/RNG metadata before running expensive campaigns.

## Verification status

The audit establishes static contract failures, an absent sparse runtime dependency, and focused test-collection failures. It does not claim to have trained a model or downloaded weights. A repaired workflow should be verified with: real spconv sparse-forward/backward tests; a one-epoch direct and grid checkpoint/resume test; YOLO11 and YOLO26 architecture-adapter tests; a nonzero SparK-to-YOLO transfer coverage test; segmentation/detection/OBB GeoTIFF inference tests; and a projected-CRS coordinate test.
