"""Fair YOLO11-OBB comparison: Ultralytics weights, random initialization, and DINO-distilled backbone."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DISTILLED_BACKBONE_PT = ROOT / "runs" / "spark-yolo" / "yolo11l_spark_latest.pt"
DEFAULT_DATA_YAML = ROOT / "configs" / "poultry_80_10_10.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "downstream_dino_distill_compare"
DEFAULT_MODEL_CFG = "yolo11l-obb.yaml"
DEFAULT_PRETRAINED_MODEL = "yolo11l-obb.pt"
DEFAULT_TRAIN_ARGS: dict[str, Any] = {
    "epochs": 50, "imgsz": 640, "batch": 16, "device": 0,
    "optimizer": "AdamW", "lr0": 1e-3, "lrf": 0.01, "warmup_epochs": 3,
    "cos_lr": True, "weight_decay": 5e-4, "momentum": 0.937,
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 15.0,
    "flipud": 0.5, "fliplr": 0.5, "mosaic": 1.0, "close_mosaic": 10,
    "amp": True, "workers": 8, "verbose": True, "exist_ok": True, "plots": True,
}
METRIC_KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")


def load_distilled_backbone(model: Any, artifact_path: Path) -> dict[str, Any]:
    """Load only an exact-match distilled YOLO backbone into a fresh OBB detector."""
    from poultry_geo.distill_models import DISTILLATION_ARTIFACT_FORMAT, validate_backbone_compatibility

    artifact_path = Path(artifact_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    
    if isinstance(artifact, dict) and artifact.get("format") == DISTILLATION_ARTIFACT_FORMAT:
        state = artifact.get("backbone_state_dict")
    elif isinstance(artifact, dict) and artifact.get("format_version") == 2 and "state_dict" in artifact:
        # SparK exported keys as '0.conv.weight', YOLO detector expects 'model.0.conv.weight'
        state = {f"model.{k}" if not k.startswith("model.") else k: v for k, v in artifact.get("state_dict").items()}
        # Fix the double bn suffix that SparK injects in SparseConv wrappers
        state = {k.replace(".bn.bn.", ".bn."): v for k, v in state.items()}
        # Un-scramble incorrectly permuted 4D dense weights
        target_state = model.model.state_dict()
        for k, v in state.items():
            if k in target_state and v.ndim == 4:
                expected_shape = tuple(target_state[k].shape)
                if tuple(v.shape) != expected_shape and tuple(v.permute(0, 2, 3, 1).shape) == expected_shape:
                    state[k] = v.permute(0, 2, 3, 1).contiguous()
    else:
        raise ValueError(f"'{artifact_path}' is not a supported portable DINO-to-YOLO or SparK artifact.")
        
    if not isinstance(state, dict):
        raise ValueError(f"'{artifact_path}' does not contain a valid state_dict.")
    compatible = validate_backbone_compatibility(model.model.state_dict(), state)
    missing, unexpected = model.model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Validated distilled load still reported unexpected keys: {unexpected}")
    print(
        f"Loaded {len(compatible)} exact distilled backbone tensors from {artifact_path.name}. "
        f"Neck and OBB head remain random; missing non-backbone tensors: {len(missing)}."
    )
    return dict(artifact.get("metadata", {}))


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Return a device-independent digest for an initialization tensor."""
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def preserve_initialized_model(model: Any, *, initialization: str) -> None:
    """Preserve in-memory weights when Ultralytics starts a YAML-based model.

    YOLO("model.yaml") has no checkpoint marker. Ultralytics consequently rebuilds
    its YAML inside Model.train(), discarding an already loaded DINO backbone. The
    marker below selects the model weight-transfer path. The callback verifies the
    state immediately before epoch 0 and fails loudly if this changes upstream.
    """
    state = model.model.state_dict()
    probe_keys = ("model.0.conv.weight", "model.4.cv1.conv.weight", "model.10.cv2.conv.weight")
    expected = {key: _tensor_digest(state[key]) for key in probe_keys if key in state}
    if len(expected) != len(probe_keys):
        raise RuntimeError(f"Could not find all YOLO backbone verification tensors: {probe_keys}")
    if not model.ckpt:
        model.ckpt = {"model": model.model}

    def verify_initialization(trainer: Any) -> None:
        actual = trainer.model.state_dict()
        changed = [key for key, digest in expected.items() if _tensor_digest(actual[key]) != digest]
        if changed:
            raise RuntimeError(
                f"Ultralytics replaced the {initialization} initialization before training. "
                f"Mismatched backbone tensors: {changed}."
            )
        print(f"Verified {initialization} initialization was preserved in the trainer.")

    model.add_callback("on_pretrain_routine_end", verify_initialization)

def extract_metrics(run_dir: Path) -> dict[str, str]:
    path = run_dir / "results.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {key.strip(): value.strip() for key, value in rows[-1].items()} if rows else {}


def is_completed(run_dir: Path, epochs: int) -> bool:
    metrics = extract_metrics(run_dir)
    try:
        return (run_dir / "weights" / "best.pt").is_file() and int(float(metrics.get("epoch", "-1"))) >= epochs - 1
    except ValueError:
        return False


def build_model(initialization: str, *, model_cfg: str, pretrained_model: str, artifact_path: Path) -> tuple[Any, dict[str, Any]]:
    from ultralytics import YOLO

    if initialization == "ultralytics_pretrained":
        return YOLO(pretrained_model), {}
    model = YOLO(model_cfg)  # Random detector: no pretrained neck/head leakage.
    if initialization == "random_init":
        return model, {}
    if initialization in ("dino_distilled", "spark_yolo"):
        return model, load_distilled_backbone(model, artifact_path)
    raise ValueError(f"Unknown initialization '{initialization}'.")


def run_experiment(
    *, name: str, initialization: str, train_args: dict[str, Any], artifact_path: Path,
    model_cfg: str, pretrained_model: str, output_dir: Path,
) -> dict[str, str]:
    save_dir = output_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\nStarting {name}: {initialization}\n{'=' * 72}")
    if not is_completed(save_dir, int(train_args["epochs"])):
        model, metadata = build_model(initialization, model_cfg=model_cfg, pretrained_model=pretrained_model, artifact_path=artifact_path)
        preserve_initialized_model(model, initialization=initialization)
        kwargs = dict(train_args, project=str(output_dir), name=name)
        for attempt in range(3):
            try:
                if attempt:
                    kwargs["resume"] = True
                model.train(**kwargs)
                break
            except PermissionError as exc:
                if attempt == 2:
                    raise
                print(f"Windows file lock: {exc}; retrying in 10 seconds.")
                time.sleep(10)
    else:
        metadata = {}
        print(f"[Skip] Completed run found: {save_dir}")
    metrics = extract_metrics(save_dir)
    metrics.update({"experiment": name, "initialization": initialization})
    if metadata:
        metrics["distillation_teacher"] = str(metadata.get("teacher_weights", ""))
    return metrics


def resolve_path(value: str | Path, default: Path) -> Path:
    path = Path(value) if value else default
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair DINO-distilled YOLO11-OBB comparison.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "downstream_scratch_compare.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    paths = config.get("paths", {})
    artifact_path = resolve_path(paths.get("distilled_backbone_pt", ""), DEFAULT_DISTILLED_BACKBONE_PT)
    data_yaml = resolve_path(paths.get("data_yaml", ""), DEFAULT_DATA_YAML)
    output_dir = resolve_path(paths.get("output_dir", ""), DEFAULT_OUTPUT_DIR)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Distilled YOLO backbone artifact not found: {artifact_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    training = dict(DEFAULT_TRAIN_ARGS)
    training.update(config.get("training", {}))
    training["data"] = str(data_yaml)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = str(paths.get("model_cfg", DEFAULT_MODEL_CFG))
    pretrained_model = str(paths.get("pretrained_model", DEFAULT_PRETRAINED_MODEL))
    repeats = int(config.get("repeats", 1))
    if repeats < 1 or repeats > 5:
        raise ValueError("repeats must be between 1 and 5.")
    seeds = (0, 7, 42, 1337, 2024)[:repeats]
    experiments = config.get("experiments") or [
        {"name": "spark_yolo", "initialization": "spark_yolo"},
        {"name": "ultralytics_pretrained", "initialization": "ultralytics_pretrained"},
        {"name": "random_init", "initialization": "random_init"},
    ]

    rows: list[dict[str, str]] = []
    summaries: dict[str, dict[str, float]] = {}
    
    experiment_runs = {str(exp["name"]): [] for exp in experiments}
    
    for seed in seeds:
        for experiment in experiments:
            name, initialization = str(experiment["name"]), str(experiment["initialization"])
            seeded_training = dict(training, seed=seed)
            run_result = run_experiment(name=f"{name}_seed{seed}", initialization=initialization, train_args=seeded_training,
                                       artifact_path=artifact_path, model_cfg=model_cfg, pretrained_model=pretrained_model,
                                       output_dir=output_dir)
            experiment_runs[name].append(run_result)
            
    for experiment in experiments:
        name, initialization = str(experiment["name"]), str(experiment["initialization"])
        runs = experiment_runs[name]
        
        if repeats == 1:
            row = runs[0]
        else:
            row = {"experiment": name, "initialization": initialization}
            for key in METRIC_KEYS:
                values = [float(run[key]) for run in runs if run.get(key)]
                if values:
                    row[key] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                    row[f"{key}_mean"] = f"{np.mean(values):.6f}"
            summaries[name] = {key: float(row.get(f"{key}_mean", "nan")) for key in METRIC_KEYS}
        rows.append(row)

    out_path = output_dir / "comparison_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")
    if repeats > 1 and "spark_yolo" in summaries and "ultralytics_pretrained" in summaries:
        delta = summaries["spark_yolo"]["metrics/mAP50-95(B)"] - summaries["ultralytics_pretrained"]["metrics/mAP50-95(B)"]
        print(f"SparK-YOLO minus Ultralytics-pretrained mAP50-95: {delta:+.4f}")


if __name__ == "__main__":
    main()
