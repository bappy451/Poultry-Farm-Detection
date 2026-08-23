# SparK Failure Review and Detector-Aligned Recovery Protocol

## Executive result

The completed YOLO11l SparK run is not a usable detector pretraining result. The raw 10%-label downstream comparison is:

| initialization | raw per-seed mAP50-95 | conclusion |
|---|---:|---|
| Ultralytics COCO-pretrained YOLO11l-OBB | 0.7414 +/- 0.0097 (4 seeds) | reference baseline |
| random YOLO11l-OBB | 0.4131 +/- 0.0062 (4 seeds) | expected low-data control |
| current SparK-to-YOLO path | 0.4054 +/- 0.0033 (3 seeds) | indistinguishable from/randomly worse than scratch |

The 300-epoch `runs/spark-yolo` run ended at SSL loss 0.73996 (best 0.70844 around epoch 71). A lower reconstruction loss is not evidence that the exported weights are compatible with the detector.

## What is wrong

1. **The RGB adapter is trained but absent from the detector export.** `src/poultry_geo/cli.py` constructs a trainable `BandAdapter` for the YOLO path. The SparK target is `transformer.adapter(batch)` (`src/poultry_geo/train_spark_ssl.py`), but downstream inference starts with raw RGB. The detector therefore receives a different input function from the one used during SSL. This is a hard transfer-contract failure.

2. **The SSL target is self-generated from the same learned adapter.** The target is detached, so gradients do not pass through it, but the adapter can still learn a convenient representation that the decoder reconstructs. It is not a fixed RGB target and does not force detector-useful semantics.

3. **Masking is at the wrong grid scale for YOLO11l.** The detector total stride is 32, while the run used `patch_size=16`. The official SparK instructions define the mask patch using the encoder's total downsampling ratio. Masking at an incompatible grid changes the information geometry seen by the backbone.

4. **The recipe is far from the official SparK regime.** This run used batch 8, 300 epochs, AdamW, and an unscaled 2e-4 learning rate. The official implementation uses a much larger effective batch, LAMB, a warm-up, gradient clipping, weight decay 0.04, mask 0.6, and roughly 1600 epochs. Matching only the loss name is not reproducing SparK.

5. **The implementation is not the official sparse implementation.** The official repository uses masked dense convolution; this project uses spconv and densifies sparse features inside SPPF/attention. Densification fills hidden locations with zeros, which downstream nonlinear blocks treat as real activations.

6. **The exported YOLO backbone is incomplete.** The YOLO11 detector backbone is modules 0..10, but older compatibility code uses `BACKBONE_MODULE_COUNT=10` and exports only modules 0..9. The C2PSA module 10 remains randomly initialized or is silently ignored.

7. **Weight translation is not architecture-preserving.** `translate_state_dict()` permutes 4-D tensors generically, including dense Ultralytics modules. The downstream loader then applies heuristic shape repair. A checkpoint that needs heuristic repair is not a valid detector initialization.

8. **The evaluation has mixed contracts.** Some older SparK comparisons use a random neck/head or a checkpoint that was partially initialized from official weights. Only raw per-seed runs with the same detector graph, same labeled split, same augmentations, and the same optimizer schedule should be used for the claim.

## Implemented recovery path

`src/poultry_geo/ema_adapt.py` implements a detector-aligned semi-supervised domain-adaptation trainer:

- student starts from `yolo11l-obb.pt` COCO weights;
- labeled poultry batches always use native Ultralytics OBB loss;
- unlabeled NAIP tiles produce synchronized weak/strong RGB views;
- the frozen EMA copy supplies normalized cosine targets at YOLO11 P3/P4/P5 (modules 4, 6, 10);
- consistency weight ramps from zero to 0.05 over 10 epochs;
- EMA uses a 0.9998 maximum decay with the standard early-update ramp;
- checkpoints remain ordinary Ultralytics `best.pt`/`last.pt` files and can be resumed or fine-tuned normally;
- unlabeled options are removed before Ultralytics argument validation, so custom options cannot leak into its config parser.

`scripts/train_ema_adapt.py` runs adaptation and then supervised fine-tuning. `scripts/train_ema_adapt.ps1` is the Windows/Conda entry point.

## Recommended experiment protocol

Run three independent seeds for each row:

1. COCO-pretrained YOLO11l -> supervised fine-tuning (baseline).
2. COCO-pretrained YOLO11l -> EMA adaptation -> supervised fine-tuning (new method).
3. The same new method with `--consistency-weight 0` (pipeline/control).

Select hyperparameters only using the validation split. Report mean +/- standard deviation on the held-out test split. Do not use the test split to choose the consistency weight, epoch, confidence threshold, or seed.

Start with 60 adaptation epochs and 100 supervised epochs. If the adapted validation mAP is below the control after 15--20 epochs, stop that run and reduce `--consistency-weight` to 0.01 or 0.02; unlabeled feature consistency is a regularizer, not a guarantee of improvement.

## Literature / implementation references

- SparK: https://arxiv.org/abs/2301.03580 and official code https://github.com/keyu-tian/SparK
- DenseCL detector-aligned dense pretraining: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Dense_Contrastive_Learning_for_Self-Supervised_Visual_Pre-Training_CVPR_2021_paper.html
- DetCo detector-oriented contrastive learning: https://openaccess.thecvf.com/content/ICCV2021/html/Xie_DetCo_Unsupervised_Contrastive_Learning_for_Object_Detection_ICCV_2021_paper.html
- EMA teacher/pseudo-label adaptation (Soft Teacher): https://arxiv.org/abs/2106.09018
- Ultralytics YOLO11 model/training reference: https://docs.ultralytics.com/models/yolo11

These references support detector-aligned feature learning and EMA teacher training, but they do not guarantee that any particular Poultry/NAIP run will beat the COCO baseline. The only valid claim is the pre-registered multi-seed comparison.
