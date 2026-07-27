# Manuscript revision and experiment plan

This checklist translates the referee feedback into executable work. The large training jobs are intentionally left for the authors to run. The current repository audit found that the manuscript and code are not yet fully aligned.

## Audit findings that must be resolved first

- The reported SparK configuration uses `n_channels: 3` in `configs/spark_scratch.yaml` and `configs/spark_grid_search.yaml`. The current paper should not claim NIR-aware or multispectral representation learning unless a four-channel experiment is completed.
- The current SparK configuration uses a `0.70` mask ratio, 512-pixel inputs, batch size 32, and 300 pretraining epochs. The manuscript previously stated a 60% mask ratio.
- The current poultry YAML reports 4,824 training images and 702 evaluation images. The manuscript previously reported 1,423 images. Reconcile the number against the exact run that produced the reported metrics.
- `configs/poultry_obb.yaml` points both `val` and `test` to the same directory. This is not an independent validation/test design.
- The current `temp/data_catalog.csv` contains 62,139 rows, while the manuscript reports a 9.87-million-tile, 1.5-TB corpus. Provide the external-corpus manifest or revise the manuscript to the corpus actually used in the reported run.
- The current repository contains 2,101 GRSD labels split into 1,680 training and 421 validation labels, consistent with an 80/20 split. `scripts/prepare_and_train_grsd.py` uses seed 42 and points `test` to the validation directory, so GRSD also lacks an independent test split.

## Programming tasks

### P0 - data and split audit

- [ ] Add `scripts/audit_dataset.py` that reports image counts, label counts, class counts, empty-label files, image dimensions, channel counts, duplicate filenames, and split overlap.
- [ ] Add a manifest writer that saves JSON/CSV metadata for every pretraining and downstream split.
- [ ] Add a geographic grouping key to every NAIP tile and split by geographic unit rather than by adjacent image filenames.
- [ ] Create independent `train`, `validation`, and `test` directories for poultry and GRSD. Never point `val` and `test` to the same path.
- [ ] Add a leakage check that fails if the same source raster, geographic group, or image stem appears in more than one split.
- [ ] Decide and document the final poultry count: 1,423, or 4,824 train plus 702 evaluation images. The paper, YAML, logs, and tables must use the same definition.

### P0 - reproducible experiment runner

- [ ] Update the downstream scripts to accept `--seed`, `--fraction`, `--dataset`, `--init`, `--epochs`, `--run-id`, and `--output-dir`.
- [ ] Save the complete resolved YAML/configuration beside every run.
- [ ] Save the git commit, Python/package versions, GPU name, random seed, and checkpoint path in a run metadata JSON file.
- [ ] Record both the final-epoch metrics and the best-validation-epoch metrics. The paper should use the predeclared checkpoint-selection rule, not whichever row happens to be last.
- [ ] Log the number of matched and skipped tensors during SparK-to-YOLO weight injection and save the mapping to a text or JSON file.
- [ ] Add a model-initialization test that verifies which YOLO layers receive SparK weights and which remain randomly initialized.

### P0 - correct evaluation outputs

- [ ] Add precision, recall, mAP@0.5, mAP@0.5:0.95, F1, and per-class AP to one machine-readable summary file per run.
- [ ] Evaluate the final selected checkpoint once on an untouched test split.
- [ ] Add per-class results for broiler breeder, broiler house, greenhouse, and every DOTA class.
- [ ] Add confusion matrices or class-wise precision/recall where meaningful.
- [ ] Ensure that all reported metrics use the same IoU thresholds, confidence thresholds, NMS settings, and oriented-box evaluator.

### P1 - figures and small statistical analysis

- [ ] Add `scripts/summarize_experiments.py` to aggregate repeated runs and calculate mean, standard deviation, absolute delta, relative delta, and bootstrap confidence intervals.
- [ ] Run the existing `scripts/plot_paper_figures.py` only after the summary format is finalized.
- [ ] Plot mean +/- standard deviation across seeds rather than a single trajectory.
- [ ] Use a zero baseline or visibly mark truncated y-axes in the mAP comparison figures.
- [ ] Rename the current convergence plot to `Downstream validation mAP during fine-tuning`; it is not an MIM reconstruction curve.
- [ ] Add a figure showing original, masked, and reconstructed NAIP tiles.
- [ ] Add a pipeline figure: NAIP tiles -> SparK pretraining -> YOLO-OBB fine-tuning -> tiled inference -> GIS export.

### P1 - channel and pretraining ablations

- [ ] Decide whether the paper is RGB-only or RGB-NIR.
- [ ] If RGB-NIR is the scientific contribution, implement four-channel input support in the pretraining and detector pipelines, including normalization and first-layer initialization.
- [ ] Run an RGB versus RGB-NIR ablation under the same split, seed, and downstream protocol.
- [ ] Run mask-ratio ablations, for example 50%, 70%, and 80%.
- [ ] Run pretraining-data scale ablations, for example 1%, 10%, and 100%, if the full corpus is available.
- [ ] Add a random-initialization YOLO baseline to separate the value of ImageNet initialization from the value of SparK pretraining.

## Large experiments for the author to run

Run these after the P0 code and split work is complete. Use at least three seeds, preferably five for the main comparison.

1. **Poultry full-data baseline:** random initialization, ImageNet initialization, and SparK-NAIP initialization.
2. **GRSD full-data baseline:** the same three initialization conditions.
3. **DOTA full-data baseline:** the same three initialization conditions.
4. **Poultry label-efficiency curve:** 10%, 25%, 50%, and 100% of training data, with fixed validation/test splits.
5. **Repeated 10% poultry experiment:** multiple independently sampled 10% subsets to test whether the 0.9778 ImageNet result is stable.
6. **Geographic generalization:** train and test on held-out geographic regions or counties.
7. **Temporal robustness:** if acquisition years are available, train on one year and test on another.
8. **RGB-NIR experiment:** only if the paper will retain multispectral/NIR claims.
9. **MIM ablation:** compare the selected SparK objective with at least one conventional MIM or contrastive baseline if computationally feasible.

## Manuscript updates after experiments

- [ ] Replace provisional dataset counts with values from the final manifest.
- [ ] Replace all single-run metrics with mean +/- standard deviation and test-set metrics.
- [ ] Add the exact geographic split and leakage-prevention procedure.
- [ ] Report the actual channel configuration and normalization.
- [ ] Report the number of epochs, batch size, learning-rate schedule, augmentation, hardware, runtime, and checkpoint-selection rule.
- [ ] Add the random-initialization baseline.
- [ ] Add class-wise results and uncertainty intervals.
- [ ] Revise the title and abstract to say RGB-only or RGB-NIR accurately.
- [ ] Replace claims of superiority with the narrower claim supported by the evidence: domain-specific MIM can approach ImageNet initialization on selected aerial detection tasks.
- [ ] Complete author information, data-access information, code-access information, and all incomplete bibliography fields.

## Submission gate

The manuscript is ready for journal submission only when the following are true:

- every table value can be regenerated from a saved result file;
- validation and test sets are independent;
- the input channel claim matches the model configuration;
- the pretraining corpus size is supported by a manifest;
- the main comparisons include random, ImageNet, and SparK initialization;
- the main results include repeated-seed uncertainty;
- the figures show the final test-set results without misleading axis scaling.
