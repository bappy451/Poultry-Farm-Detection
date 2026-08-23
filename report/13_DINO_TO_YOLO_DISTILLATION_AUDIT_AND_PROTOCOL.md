# DINO-to-YOLO distillation: audit and experimental protocol

## Verdict

The original implementation was **not a valid test** of whether DINO distillation beats an Ultralytics COCO-initialized YOLO detector. It has now been repaired. The new pipeline uses a randomly initialized YOLO11 *detection* graph as the student, transfers only that exact graph's backbone, and refuses any incomplete or architecture-mismatched downstream load.

This is a sound representation-distillation experiment, but it is not a guarantee that it will beat COCO initialization. COCO pretraining transfers dense supervised detection knowledge. DINO can win when the unlabeled pretraining imagery is large, visually close to the target aerial poultry imagery, and completely excludes held-out evaluation geography.

## Fixed implementation faults

1. **Wrong student architecture and pretraining leakage — critical.** The student was created with `yolo11l-cls.pt`, which downloads a pretrained classification model. The downstream model is a detection YAML. The two graphs are not an experimentally controlled initialization and the classifier's external pretraining confounds the result.
   - Fixed: construct the student from the same random detection YAML (`yolo11l.yaml`) used downstream.

2. **YOLO received DINO-normalized pixels — critical.** The original loop sent ImageNet-normalized images to both teacher and student. Ultralytics YOLO fine-tuning/inference uses RGB tensors in `[0,1]`; the student therefore learned under a different input distribution than the downstream detector.
   - Fixed: only the DINO teacher sees normalized RGB; YOLO sees the raw RGB crop.

3. **Only one feature scale was distilled — major.** Distilling P4 alone does not directly supervise the P3/P5 hierarchy that a YOLO detector uses for small, medium, and large objects.
   - Fixed: P3, P4, and P5 are each projected into DINO token space and supervised against bilinearly resized teacher patch grids.

4. **Partial loads could silently pass — critical.** The downstream script previously loaded whatever keys happened to match and continued even if most of the backbone was random.
   - Fixed: every expected backbone tensor must be present with exactly matching shape. The current YOLO11n smoke test verified 198/198 backbone tensors.

5. **No held-out criterion selected the exported model — major.** The checkpoint was selected from training loss only.
   - Fixed: held-out unlabeled validation distillation loss selects `yolo_backbone_distilled_best.pt`.

6. **Resume could mix incompatible runs — major.** The old resume path did not verify crop size or student architecture.
   - Fixed: resume rejects a different detection YAML or crop size.

7. **The probe forward pass altered fresh BatchNorm statistics — moderate.** Channel discovery was run while the new YOLO backbone was in training mode.
   - Fixed: the probe executes in evaluation mode and restores the previous mode.

8. **Legacy classifier-based checkpoints are blocked — deliberate.** They could contain the invalid old student and cannot be used in the primary result. The downstream script accepts only the new v2 portable artifact.

## Recommended primary experiment

Use the project DINO checkpoint trained on your own unlabeled NAIP/aerial image pool, not `--teacher-weights official`, for the main claim. The official DINO option is useful only as an ablation because it is not domain adapted.

Train and compare three initializations of the exact same YOLO11 architecture:

| Arm | Initialization | Purpose |
|---|---|---|
| Scratch | `yolo11l.yaml` | Measures the target-label-only baseline. |
| COCO | `yolo11l.pt` | The required strong Ultralytics baseline. |
| DINO-distilled | `yolo11l.yaml` + `yolo_backbone_distilled_best.pt` | Tests the proposed method. |

Keep dataset split, image size, augmentation, epochs, optimizer, learning-rate schedule, early-stopping rule, and hyperparameter-search budget identical for all three arms. Run at least three random seeds and report mean ± standard deviation (or bootstrap confidence interval) for mAP50-95, mAP50, precision, recall, and per-class AP.

Create the DINO/distillation image pool from training geography only. Do not use validation/test tiles, overlapping tiles, or nearby tiles from the same farm/site. The existing parent-group split is useful but a spatial/site-level split is preferable.

## Practical training guidance

The command now requires an explicit teacher and writes `yolo_backbone_distilled_best.pt`:

```powershell
poultry-geo train-distill <unlabeled_train_only_rgb_root> runs/dino_yolo_distill --teacher-weights <your_dino_teacher_checkpoint.pt> --vit-name B --student-model yolo11l.yaml --crop-size 640 --batch-size <fits_gpu> --precision bf16
```

Then fine-tune from the portable artifact:

```powershell
python scripts/run_distilled_poultry.py --distill-ckpt runs/dino_yolo_distill/yolo_backbone_distilled_best.pt --data <poultry.yaml> --model-cfg yolo11l.yaml --imgsz 640
```

Use a distillation crop size close to the downstream size. `224` is a valid low-memory smoke-test setting, but `640` (or a multi-resolution curriculum) is methodologically closer to 640-pixel detection and better preserves small poultry-farm structures. It must be divisible by 32; the DINO patch grid is resized at P3/P4/P5 automatically.

## Method assessment

Dense DINO-to-CNN distillation is a reasonable project contribution: DINO provides patch-level semantic structure and YOLO benefits from its native multi-scale CNN hierarchy. The approach is better aligned with detection than global-image embedding matching because it preserves spatial correspondence.

The closest direct evidence is *Self-Supervised YOLO*, which pretrains a YOLO backbone with a self-supervised objective before detection fine-tuning. It supports the broader premise that SSL can improve label efficiency, but it does not prove DINO distillation will outperform COCO for this dataset. *Efficient Teacher* is semi-supervised detection using pseudo-labels, a different method that can be a future extension once labeled data is available.

Primary references: [Ultralytics model-YAML architecture guide](https://docs.ultralytics.com/guides/model-yaml-config/), [DINO](https://arxiv.org/abs/2104.14294), [Self-Supervised YOLO](https://arxiv.org/abs/2508.01966), and [Efficient Teacher](https://arxiv.org/abs/2302.07577).
