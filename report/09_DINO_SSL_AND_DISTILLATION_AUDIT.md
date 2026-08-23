# DINO SSL and DINO Distillation Audit

**Date:** 2026-08-13  
**Scope:** Read-only audit of the custom DINO SSL implementation, DINO checkpoint consumers/visualization, the ViT-to-YOLO distillation path, related scripts/configuration, and focused tests. No source code was changed.

## Executive conclusion

The DINO training path has a recognizable DINO-style core (multi-crop teacher/student views, centered/sharpened teacher targets, EMA teacher, and a prototype head), but it is not presently a reliable or reproducible pretraining workflow. Several advertised controls are ignored, no validation/model-selection logic runs, and some important stability/distributed-training details are missing.

The separate **DINO-to-YOLO distillation workflow is currently not trustworthy for experiments or downstream transfer claims**. Its teacher loader does not understand the checkpoint format emitted by the project DINO trainer; the `official` loader silently fails to map most upstream weights into the custom ViT; the input normalization is inconsistent with both teacher choices; and the exported checkpoint cannot be consumed by the downstream-training script as written.

Severity definitions:

- **Critical:** output is random, wrong, unusable, or a claimed experiment is invalid.
- **High:** likely failure, materially invalid training, or silently misleading result.
- **Medium:** stability, reproducibility, resume, or measurement risk.
- **Low:** maintainability or diagnostics issue.

## Evidence and method

Reviewed project code/configuration in:

- `src/poultry_geo/dino.py`, `dino_augment.py`, `dino_loss.py`, `train_dino_ssl.py`, `dino_vis.py`
- `src/poultry_geo/distill_models.py`, `distill_loss.py`, `train_distill.py`
- `src/poultry_geo/cli.py`, `src/downstream_finetune/dino_vit_compare.py`, `dino_vidt_compare.py`
- `configs/dino_pretrain_naip.yaml`, `configs/dino_vit_downstream_compare.yaml`, `configs/vidt_poultry_standalone.yaml`
- `scripts/run_distilled_poultry.py` and DINO wrappers

Focused verification was run with:

```text
.venv\Scripts\python.exe -m pytest tests\test_dino_vit_compare.py -q --tb=short
Result: 1 failed, 4 passed
```

The failure is a real configuration contract mismatch: the test expects three strategies while `configs/dino_vit_downstream_compare.yaml` defines five.

The audit also compared key behavior against the official [facebookresearch/dino training code](https://github.com/facebookresearch/dino/blob/main/main_dino.py) and [DINO projection-head implementation](https://github.com/facebookresearch/dino/blob/main/vision_transformer.py). Those are used only as references for the intended DINO algorithm; this audit does not require an exact upstream reimplementation.

## Critical findings

### C1. Project DINO checkpoints are not loaded by the distillation teacher

**Evidence:** `train_dino_ssl.save_dino_checkpoint()` writes top-level keys named `student_backbone`, `teacher_backbone`, `student_head`, and `teacher_head`. `train_distill.load_teacher()` instead looks for `teacher`; otherwise it passes the *entire checkpoint dictionary* to `RemoteSensingViT.load_state_dict(..., strict=False)`.

The emitted `last.pt` therefore contains no tensor keys matching the standalone `RemoteSensingViT` state dictionary. With `strict=False`, the load can complete without an exception while loading zero useful backbone tensors. The code then prints “Teacher weights loaded successfully.”

**Impact:** Passing a checkpoint produced by `train-dino` to `train-distill --teacher-weights ...` silently trains a student against a randomly initialized frozen teacher. Any resulting distillation loss or downstream result cannot support a DINO-transfer claim.

**Required correction:** Implement one explicit checkpoint schema; extract `teacher_backbone` (or deliberately `student_backbone`), strip the `AdaptedBackbone` prefix only when present, and reject a load unless an expected/high percentage of backbone parameters matched. Persist schema/version/model geometry in every checkpoint.

### C2. The advertised downstream export cannot extract the distilled backbone

**Evidence:** `DistillationStudent` registers its CNN under `encoder` and its projector under `projector`, producing state-dict keys such as `encoder.stem...`. `scripts/run_distilled_poultry.py:extract_backbone()` only selects keys beginning with `backbone.`.

**Impact:** The script extracts an empty state dictionary from every checkpoint emitted by `train-distill`; it reports the count but does not reject zero. The later YOLO load proceeds with no distilled weights. The claimed downstream run is therefore from scratch, not from the distilled backbone.

**Required correction:** Define and test a supported export mapping from the exact `YOLODenseEncoder` architecture to the exact downstream YOLO architecture. Fail on zero matches and record matched/missing tensors. Do not rely on a textual prefix assumption.

### C3. The `official` DINO teacher loader is structurally incompatible and hides the mismatch

**Evidence:** `train_distill.load_teacher()` downloads a legacy Facebook DINO model, removes `cls_token` and the first positional token, then calls `RemoteSensingViT.load_state_dict(state_dict, strict=False)`. The upstream model uses different module/key structure (for example `patch_embed.proj.*` and its own attention implementation), while the local ViT uses custom precision wrapper modules. Unlike the repository’s `RemoteSensingViT.load_pretrained_weights()`, this path does not translate names or filter by shape, and does not inspect missing/unexpected keys.

**Impact:** The success message is not proof that official weights were loaded. In practice many or all intended tensors can remain randomly initialized. Distillation labelled as “official DINO” is unvalidated.

**Required correction:** Either use the downloaded upstream model directly as teacher, including its official preprocessing, or build a verified key/shape translator and enforce a minimum match threshold. Log the exact number of loaded tensors and parameter coverage.

### C4. Distillation input preprocessing disagrees with the selected teacher

**Evidence:** `train-distill` calls `naip_data.get_dataloaders(..., precision='fp32')`; that dataset returns images in `[0, 1]` and applies random crop/flip/color jitter but does not apply ImageNet normalization. `load_teacher('official', ...)` uses ImageNet-trained DINO weights, and custom DINO pretraining applies ImageNet mean/std in `DINOMultiCropDataAugmentation._normalize()` for RGB.

**Impact:** Both teacher modes see a different input distribution during distillation than during their pretraining. The frozen teacher features are shifted, and the student is optimized to reproduce the shifted, not intended, representation.

**Required correction:** Make preprocessing a first-class, checkpointed contract. Apply the same deterministic normalization to teacher and student after augmentation, with explicit handling for NAIP RGB/NIR data. Add a test asserting identical tensor normalization for custom-DINO and distillation when they share a teacher.

## High findings

### H1. The DINO CLI advertises learning-rate warmup and cosine weight decay but implements neither

**Evidence:** `train-dino` accepts/loads `learning_rate`, `weight_decay`, `weight_decay_end`, and `warmup_epochs`. It creates AdamW once with a constant learning rate and one constant weight decay. `warmup_epochs` is only forwarded to the teacher-temperature schedule. There is no LR scheduler, no per-step LR update, and no per-step weight-decay update.

`configs/dino_pretrain_naip.yaml` explicitly describes “AdamW peak learning rate”, “Initial cosine schedule weight decay”, “Final weight decay”, and “Linear warmup epochs”, which do not match execution.

**Impact:** Runs do not use their documented optimization recipe. This is particularly significant for DINO-style training, where optimization and EMA schedules are central to stability and representation quality.

**Required correction:** Implement and checkpoint per-step warmup/cosine schedules for LR and weight decay, or remove the unused configuration and documentation. Log effective LR/weight decay every epoch/step.

### H2. The DINO similarity CLI always throws before loading a model

**Evidence:** `dino_vis.generate_dino_visualizations()` calls `build_dino_model(..., out_dim=65536, ...)`, but the factory parameter is named `dino_out_dim`. Interface inspection confirms no `out_dim` keyword exists.

**Impact:** `poultry-geo visualize-dino-similarity` fails with `TypeError` immediately.

**Required correction:** Use the factory’s actual keyword and add a smoke test that invokes the CLI function with a tiny local image and a non-downloading model stub.

### H3. The visualization checkpoint loader does not support the project’s DINO checkpoints

**Evidence:** The DINO trainer saves `teacher_backbone` and `student_backbone`, whereas `generate_dino_visualizations()` looks only for `teacher` or `model`; otherwise it attempts to load the whole checkpoint into the wrapper with `strict=False`.

**Impact:** Even after H2 is fixed, the documented `last.pt` is silently ignored or mostly ignored, causing plots to represent a fresh/pretrained model rather than the trained checkpoint.

**Required correction:** Reuse the single validated DINO checkpoint-loading helper and report parameter coverage. Visualizations must write model/checkpoint identifiers into their metadata.

### H4. Distillation resume is incompatible with `torch.compile` checkpoint keys

**Evidence:** `train-distill` loads `student` before calling `torch.compile(student)`, then saves `student.state_dict()` after compilation. Compiled modules normally expose keys prefixed with `_orig_mod.`; the next resume constructs an uncompiled `DistillationStudent` and loads those keys directly.

**Impact:** The second run is likely to fail to resume due to state-dict key mismatch. If the behavior varies by PyTorch version, it remains unverified and fragile.

**Required correction:** Save/load the unwrapped module’s state dict through an explicit helper and verify a compile-enabled resume smoke test.

### H5. The default DINO comparison configuration is internally contradictory and cannot run as documented

**Evidence:** `configs/dino_vit_downstream_compare.yaml` and related script text describe a 9-experiment / 3-model comparison, but the config contains five model entries and three datasets (15 combinations). The focused test fails because it expects three models. The default DINO checkpoint path is `runs/dino_pretrain_failed`, which is intentionally/non-generically named and will cause a missing-checkpoint failure for the DINO arm.

The suite only treats `dino` and `imagenet` specially. Its `mae` and `satmae` entries take the `else` branch and train from random initialization; their `pretrained_weights` fields are never consumed.

**Impact:** The comparison labels and reported number of experiments are false, and the MAE/SatMAE baselines are not actually initialized from their stated weights. Results would be invalid baseline comparisons.

**Required correction:** Decide whether the suite is 9 or 15 experiments; align config, tests, wrappers, and documentation. Implement real MAE/SatMAE loaders or remove those strategies. Validate every configured checkpoint before the first run.

### H6. No DINO SSL test exercises the core training contract

**Evidence:** The project test inventory contains tests for `dino_vit_compare`, but no direct tests for `dino.py`, `dino_augment.py`, `dino_loss.py`, `train_dino_ssl.py`, `distill_models.py`, `train_distill.py`, or `dino_vis.py`.

**Impact:** The critical interface errors above reached the repository without detection. The four passing focused downstream tests do not prove DINO training or distillation works.

**Required correction:** Add CPU tiny-model tests for one DINO step, center/EMA/checkpoint round-trip, multi-crop positional interpolation, project-checkpoint teacher loading, official-model adapter coverage, distillation shape/preprocessing, compile-resume, and distilled-to-YOLO export coverage.

## Medium findings

### M1. The DINO implementation claims an iBOT objective but training never supplies patch logits or a mask

**Evidence:** `CompositeDINOLoss` contains `iBOTPatchLoss`, but `train_one_step()` supplies only global DINO logits and optional global student features. It never produces student/teacher patch projection logits or a masked-patch mask.

**Impact:** The active objective is DINO + KoLeo only; iBOT is dead code. Documentation/comments implying a combined DINO+iBOT objective are inaccurate.

**Required correction:** Either implement the masked-token pathway and checkpoint its patch center, or remove/disable the unused objective explicitly.

### M2. DINO center state is not distributed-safe and is downcast after every update

**Evidence:** `DINOLoss.update_center()` averages only the local process’s teacher logits. It does not all-reduce across ranks despite `get_dino_dataloaders()` creating distributed samplers when a process group exists. It also casts `self.center` to float for the calculation, then stores it in the teacher-logit dtype (BF16/FP16 under mixed precision).

The official DINO reference all-reduces the batch center and maintains a global EMA center.

**Impact:** Multi-process runs would train divergent teachers/centers; low-precision center state adds unnecessary numerical error in a collapse-sensitive control signal.

**Required correction:** Either state that distributed training is unsupported and reject it, or wrap/synchronize model state and all-reduce the FP32 center. Keep the center buffer FP32.

### M3. DINO uses a nonstandard/fixed head-stability policy without an exposed control

**Evidence:** The local `DINOHead` creates a weight-normalized last layer but never freezes the magnitude parameter (`weight_g`) and exposes no `norm_last_layer` or early-last-layer-freeze control. The official DINO head conditionally freezes this magnitude; the official training loop also supports early last-layer gradient cancellation. The exact policy may be a valid experiment choice, but it is undocumented and not configurable.

**Impact:** Training stability and comparability with stated DINO defaults are weaker, especially at the configured batch size of 32 rather than the much larger batches typical in the original recipe.

**Required correction:** Expose and log the policy, choose a validated default for each ViT size, and add collapse diagnostics.

### M4. DINO checkpoint/resume is incomplete for the declared composite objective and run provenance

**Evidence:** DINO checkpoints save only `loss_fn.dino_loss.center`; they omit the iBOT patch center. They do not save a global step, scheduler state (none exists), RNG state, dataloader/sampler epoch, or a validated configuration signature.

**Impact:** A resumed run cannot reproduce the interrupted trajectory, and a future iBOT-enabled run resumes with a reset patch center.

**Required correction:** Save all loss buffers, global step/schedule state, RNG/sampler state, data/catalog fingerprint, and immutable config signature. Reject incompatible resumes.

### M5. The DINO CLI creates a validation loader but never evaluates it or selects a best checkpoint

**Evidence:** `train-dino` assigns `train_loader, val_loader = ...`, but only iterates `train_loader`. Every epoch saves `last.pt`; no validation metric, linear probe, collapse metric threshold, or best-checkpoint selection exists.

**Impact:** Training quality cannot be selected or compared consistently. The named validation split adds cost without providing evidence.

**Required correction:** Either remove the unused loader or add a deterministic held-out representation/probe/diagnostic evaluation and a documented best-checkpoint criterion.

### M6. Distillation accepts an unsupported ViT-L architecture

**Evidence:** `DistillConfig` sets ViT-L to embedding 1024 and 16 heads but `load_teacher()` always creates depth 12. Official DINO ViT-L uses 24 transformer blocks (and the current DINOv3 architecture also uses depth 24 for ViT-L).

**Impact:** `--vit-name L --teacher-weights official` cannot faithfully load an official ViT-L teacher; strict-false loading hides the structural mismatch.

**Required correction:** Define a complete architecture table (embed dimension, heads, depth, patch size, implementation/version) and validate it against the checkpoint before loading.

### M7. The YOLO feature extractor assumes one undocumented Ultralytics internal graph

**Evidence:** `YOLODenseEncoder` slices `yolo.model.model` at fixed indices 0:10 and assumes stage 3 is P4/stride 16. Its default accepts arbitrary `backbone_type`, but no architecture/version guard checks that those indices mean the same thing for every YOLO family or release.

**Impact:** Changing `backbone_type`, model family, or Ultralytics release can silently distill from the wrong layer or fail at initialization. This is especially risky because `run_distilled_poultry.py` defaults downstream model config to `yolo11n.yaml`, potentially differing from the distilled backbone (default `yolo11l`).

**Required correction:** Use named modules/feature hooks or an architecture-specific adapter. Persist Ultralytics version and model YAML; require exact source/target architecture compatibility.

### M8. The downstream DINO loaders report counts but do not enforce meaningful coverage

**Evidence:** `dino_vit_compare.load_dino_backbone_into_vit()` and `dino_vidt_compare.load_dino_backbone_into_vit()` use `strict=False` and print missing/unexpected key counts. Neither rejects zero/low coverage, malformed non-square positional token counts, or architecture mismatch.

**Impact:** A “DINO-pretrained” comparison arm may actually be mostly random while still completing and generating results.

**Required correction:** Calculate tensor and parameter coverage, require expected critical tensors (patch embedding, every required block, norm, position embedding), and abort below a threshold.

## Lower-priority observations

- `DINODataset` performs the same random multi-crop augmentation in the nominal validation split. This would make validation nondeterministic if it were used.
- The DINO trainer samples visualization files with Python’s unseeded `random.sample`; there is no run-level seed for model/data/augmentation reproducibility.
- `load_image_tensor()` used by visualization supplies `[0,1]` RGB without the DINO training normalization. Thus after-epoch plots and CLI plots do not inspect the same input distribution used for RGB pretraining.
- The visualizer describes a “salience / attention” map, but it is cosine similarity to the mean patch feature, not transformer attention. This is a labeling/interpretation issue rather than a training defect.
- `train-distill` imports `GradScaler` twice and initializes a BF16 scaler even though scaling is normally unnecessary for BF16. This is not the primary correctness issue, but it complicates numerical behavior and testing.

## What appears conceptually sound

These pieces are directionally appropriate and should be retained behind proper tests:

- Multi-crop layout has two global student/teacher views plus local student-only views, and same-index global pairs are excluded in the DINO cross-view loss.
- The custom ViT supports positional interpolation for local and global crop resolutions divisible by the patch size.
- Teacher parameters are frozen and are EMA-updated after the optimizer step.
- The DINO loss performs teacher softmax and student log-softmax in FP32, which is prudent with a 65,536-way head.
- The student stride-16 P4 grid should geometrically match a 224/16 ViT patch grid (14×14), assuming the fixed Ultralytics graph layout remains valid.

These observations do **not** offset the critical loading/export/preprocessing failures above.

## Recommended repair order

1. Stop treating current distillation or its downstream output as DINO transfer evidence.
2. Establish one DINO checkpoint schema and one strict, coverage-checked loader used by training resume, distillation, visualization, and downstream fine-tuning.
3. Repair or replace the distillation teacher with a verified direct upstream teacher or a tested project-teacher extraction path; enforce identical preprocessing.
4. Repair the student-to-downstream YOLO mapping and prove it with a round-trip test that checks nonzero, correctly mapped tensors.
5. Implement/remove the advertised DINO schedules and iBOT path; add collapse, feature-variance, and checkpoint-selection diagnostics.
6. Align downstream comparison config, test, wrappers, labels, and actual baseline loaders before collecting any comparison metrics.
7. Add small CPU test fixtures for every contract before restarting long GPU runs.

## Verification status

This report establishes static code-contract failures and one focused failing test. It does not claim to have trained models or downloaded external weights. Before a repaired system is used for research results, run: a tiny one-step DINO smoke test; exact checkpoint round-trip/coverage tests; an official-teacher feature equivalence check; a one-epoch distillation smoke test; and an export-to-downstream weight-coverage test.
