"""Pretrain a baseline-compatible YOLO11 OBB backbone on unlabeled MS tiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for location in (ROOT, SRC):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from poultry_geo.naip_data import NAIPDataset
from scripts.ssl_anchored_dino_yolo_obb import (
    AnchoredYOLOOBBStudent,
    TimmDinoV2Teacher,
    anchored_distillation_loss,
    capture_parameter_anchors,
    make_artifact,
    relative_anchor_loss,
)


VALIDATION_SAMPLING_CONTRACT = "source_group_round_robin_v1"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _hash_config(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    if value.isdigit():
        value = f"cuda:{value}"
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return result


def _validate(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "unlabeled_root", "init_model", "output_dir", "sample_size", "val_fraction",
        "subset_seed", "crop_size", "batch", "workers", "epochs", "epoch_size",
        "val_batches", "precision", "backbone_lr", "projector_lr", "weight_decay",
        "anchor_weight", "warmup_epochs", "teacher_model", "device",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"SSL pretraining config is missing: {sorted(missing)}")
    if int(config["crop_size"]) > 512 or int(config["crop_size"]) % 32 or int(config["crop_size"]) % 14:
        raise ValueError("crop_size must be <=512 and divisible by both 32 and 14 (use 448).")
    if not 0 < float(config["val_fraction"]) < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5).")
    for name in ("sample_size", "batch", "epochs", "epoch_size", "val_batches"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(config["workers"]) < 0:
        raise ValueError("workers cannot be negative.")
    if str(config["precision"]) not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16.")
    if float(config["backbone_lr"]) <= 0 or float(config["projector_lr"]) <= 0:
        raise ValueError("learning rates must be positive.")
    if float(config["weight_decay"]) < 0:
        raise ValueError("weight_decay cannot be negative.")
    if float(config["anchor_weight"]) < 0:
        raise ValueError("anchor_weight cannot be negative.")
    if int(config["warmup_epochs"]) < 0 or int(config["warmup_epochs"]) > int(config["epochs"]):
        raise ValueError("warmup_epochs must be between zero and epochs.")
    if float(config.get("clip_grad_norm", 1.0)) <= 0:
        raise ValueError("clip_grad_norm must be positive.")
    scale_weights = config.get("scale_weights", {"p3": 1.0, "p4": 1.0, "p5": 1.0})
    if not isinstance(scale_weights, dict) or set(scale_weights) != {"p3", "p4", "p5"}:
        raise ValueError("scale_weights must define exactly p3, p4, and p5.")
    if any(float(value) <= 0 for value in scale_weights.values()):
        raise ValueError("Every scale weight must be positive.")
    if not str(config["teacher_model"]).strip():
        raise ValueError("teacher_model cannot be empty.")
    return config


def _group_split(files: list[Path], *, root: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Sample path falls outside unlabeled_root: {path}") from exc
        group = relative.parts[0] if len(relative.parts) > 1 else path.stem
        grouped[group].append(path)
    if len(grouped) < 2:
        raise ValueError("Source-disjoint SSL validation needs at least two MS source groups.")
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    target = max(1, round(len(files) * val_fraction))
    val_groups: set[str] = set()
    count = 0
    for group in groups:
        if count >= target and val_groups:
            break
        val_groups.add(group)
        count += len(grouped[group])
    if len(val_groups) == len(groups):
        val_groups.remove(groups[-1])
    train = sorted(path for group, paths in grouped.items() if group not in val_groups for path in paths)
    val = sorted(path for group, paths in grouped.items() if group in val_groups for path in paths)
    if not train or not val:
        raise RuntimeError("MS source-group split produced an empty subset.")
    return train, val


def _source_group_round_robin(files: list[Path], *, root: Path, seed: int) -> list[Path]:
    """Interleave source groups so a truncated validation pass is representative."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Validation sample falls outside unlabeled_root: {path}") from exc
        group = relative.parts[0] if len(relative.parts) > 1 else path.stem
        grouped[group].append(path)
    if not grouped:
        raise ValueError("Cannot order an empty SSL validation set.")
    rng = random.Random(seed)
    group_names = sorted(grouped)
    rng.shuffle(group_names)
    for name in group_names:
        grouped[name].sort()
        rng.shuffle(grouped[name])
    ordered: list[Path] = []
    index = 0
    while len(ordered) < len(files):
        added = False
        for name in group_names:
            if index < len(grouped[name]):
                ordered.append(grouped[name][index])
                added = True
        if not added:
            break
        index += 1
    if len(ordered) != len(files) or len(set(ordered)) != len(files):
        raise RuntimeError("Source-group validation ordering lost or duplicated samples.")
    return ordered


def _load_or_create_manifest(config: dict[str, Any], output_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest = output_dir / "ms_ssl_manifest.csv"
    root = _resolve(config["unlabeled_root"])
    if manifest.is_file():
        train, val = [], []
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"split", "source_group", "path"}
            if not required_fields.issubset(reader.fieldnames or []):
                raise ValueError(f"Existing SSL manifest lacks fields {sorted(required_fields)}.")
            for row in reader:
                if row["split"] not in {"train", "val"}:
                    raise ValueError(f"Invalid SSL manifest split: {row['split']!r}")
                path = Path(row["path"])
                if not path.is_absolute():
                    raise ValueError(f"SSL manifest paths must be absolute: {path}")
                try:
                    relative = path.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"SSL manifest path falls outside unlabeled_root: {path}") from exc
                expected_group = relative.parts[0] if len(relative.parts) > 1 else path.stem
                if row["source_group"] != expected_group:
                    raise ValueError(f"SSL manifest source_group mismatch for {path}")
                (val if row["split"] == "val" else train).append(path)
        if len(train) + len(val) != int(config["sample_size"]):
            raise ValueError("Existing SSL manifest does not match configured sample_size.")
        if len(set(train)) + len(set(val)) != len(train) + len(val) or set(train) & set(val):
            raise ValueError("Existing SSL manifest contains duplicate or cross-split paths.")
        train_groups = {
            (relative.parts[0] if len(relative.parts) > 1 else path.stem)
            for path in train for relative in (path.relative_to(root),)
        }
        val_groups = {
            (relative.parts[0] if len(relative.parts) > 1 else path.stem)
            for path in val for relative in (path.relative_to(root),)
        }
        if train_groups & val_groups:
            raise ValueError("Existing SSL manifest is not source-group disjoint.")
        missing = next((path for path in [*train, *val] if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"Existing SSL manifest contains missing sampled file: {missing}")
        return train, val

    sampled = NAIPDataset(
        root, crop_size=int(config["crop_size"]), channels=3, precision="fp32",
        augment=False, max_files=int(config["sample_size"]), subset_seed=int(config["subset_seed"]),
    )
    files = list(sampled.files)
    if len(files) != int(config["sample_size"]):
        raise RuntimeError(f"Requested {config['sample_size']} MS samples, found {len(files)}.")
    train, val = _group_split(
        files, root=root, val_fraction=float(config["val_fraction"]), seed=int(config["subset_seed"])
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source_group", "path"])
        writer.writeheader()
        for split, paths in (("train", train), ("val", val)):
            for path in paths:
                writer.writerow({"split": split, "source_group": path.relative_to(root).parts[0], "path": str(path)})
    return train, val


def _lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _autocast_dtype(precision: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def _build_optimizer(
    student: AnchoredYOLOOBBStudent, config: dict[str, Any], *, legacy_layout: bool = False,
) -> torch.optim.AdamW:
    backbone_lr, projector_lr = float(config["backbone_lr"]), float(config["projector_lr"])
    weight_decay = float(config["weight_decay"])
    if legacy_layout:
        return torch.optim.AdamW(
            [
                {"params": student.backbone.parameters(), "lr": backbone_lr},
                {"params": student.projectors.parameters(), "lr": projector_lr},
            ],
            weight_decay=weight_decay,
        )
    groups: list[dict[str, Any]] = []
    for role, module, learning_rate in (
        ("backbone", student.backbone, backbone_lr),
        ("projector", student.projectors, projector_lr),
    ):
        decayed, no_decay = [], []
        for _, parameter in module.named_parameters():
            (decayed if parameter.ndim >= 2 else no_decay).append(parameter)
        if not decayed or not no_decay:
            raise RuntimeError(f"Expected both decay and no-decay {role} parameters.")
        groups.extend(
            [
                {
                    "params": decayed, "lr": learning_rate, "weight_decay": weight_decay,
                    "role": role, "decay_contract": "kernel_or_matrix",
                },
                {
                    "params": no_decay, "lr": learning_rate, "weight_decay": 0.0,
                    "role": role, "decay_contract": "bias_or_norm",
                },
            ]
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = list(student.parameters())
    if len(optimized) != len(expected) or {id(value) for value in optimized} != {id(value) for value in expected}:
        raise RuntimeError("AdamW parameter grouping lost or duplicated student parameters.")
    return optimizer


def _atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _last_metric_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return int(rows[-1]["epoch"]) if rows else None


def _save_training_state(
    path: Path, *, student, optimizer, scaler, generator: torch.Generator,
    epoch: int, best: float, config_hash: str,
) -> None:
    _atomic_torch_save(
        {
            "student": student.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch, "best_val_loss": best, "config_sha256": config_hash,
            "data_generator_state": generator.get_state(),
            "validation_sampling_contract": VALIDATION_SAMPLING_CONTRACT,
            "optimizer_contract": "adamw_kernel_matrix_decay_bias_norm_no_decay_v1",
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = _validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    output_dir = _resolve(config["output_dir"])
    init_model = _resolve(config["init_model"])
    unlabeled_root = _resolve(config["unlabeled_root"])
    if not init_model.is_file() or not unlabeled_root.is_dir():
        raise FileNotFoundError(f"Missing init model or unlabeled root: {init_model}, {unlabeled_root}")
    if args.preflight_only:
        print("Preflight passed: anchored SSL uses MS only; no AL labels or test data are configured.")
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    config_text = config_path.read_text(encoding="utf-8")
    config_snapshot = output_dir / "pretrain_config.yaml"
    if config_snapshot.is_file() and config_snapshot.read_text(encoding="utf-8") != config_text:
        raise ValueError("SSL output directory belongs to a different pretraining config snapshot.")
    if not config_snapshot.is_file():
        _atomic_text(config_snapshot, config_text)
    config_hash = _hash_config(config)
    completion_path = output_dir / "pretraining_complete.json"
    if completion_path.is_file():
        raise FileExistsError(
            f"SSL pretraining is already marked complete: {completion_path}. "
            "Let the protocol runner reuse it or select a new output directory."
        )
    train_files, val_files = _load_or_create_manifest(config, output_dir)
    manifest_sha256 = _file_sha256(output_dir / "ms_ssl_manifest.csv")
    val_files = _source_group_round_robin(
        val_files, root=unlabeled_root, seed=int(config["subset_seed"]) + 1
    )
    crop_size, batch = int(config["crop_size"]), int(config["batch"])
    train_data = NAIPDataset(unlabeled_root, files=train_files, crop_size=crop_size, channels=3, precision="fp32", augment=True)
    val_data = NAIPDataset(unlabeled_root, files=val_files, crop_size=crop_size, channels=3, precision="fp32", augment=False)
    generator = torch.Generator().manual_seed(int(config["subset_seed"]))
    loader_options = {
        "batch_size": batch, "num_workers": int(config["workers"]), "pin_memory": True,
        # Recreate workers each epoch so their augmentation RNG seeds are a
        # deterministic function of the epoch and survive process restarts.
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, drop_last=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, drop_last=False, **loader_options)
    if not len(train_loader) or not len(val_loader):
        raise RuntimeError("SSL train/validation loader is empty.")

    device = _device(str(config["device"]))
    teacher = TimmDinoV2Teacher(
        str(config["teacher_model"]), image_size=crop_size, pretrained=True
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = AnchoredYOLOOBBStudent(init_model, teacher_dim=teacher.embed_dim).to(device)
    anchors = capture_parameter_anchors(student)
    trainable_backbone_parameters = sum(
        parameter.numel() for parameter in student.backbone.parameters() if parameter.requires_grad
    )
    total_backbone_parameters = sum(parameter.numel() for parameter in student.backbone.parameters())
    if trainable_backbone_parameters != total_backbone_parameters or total_backbone_parameters == 0:
        raise RuntimeError(
            "Backbone trainability invariant failed before optimizer construction: "
            f"{trainable_backbone_parameters}/{total_backbone_parameters}."
        )
    optimizer = _build_optimizer(student, config)
    precision = str(config["precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16" and device.type == "cuda")
    epochs, epoch_size = int(config["epochs"]), min(int(config["epoch_size"]), len(train_loader))
    val_batches = min(int(config["val_batches"]), len(val_loader))
    total_steps = epochs * epoch_size
    warmup_steps = int(config["warmup_epochs"]) * epoch_size
    start_epoch, best_val = 0, float("inf")
    state_path = output_dir / "training_last.pt"
    metrics_path = output_dir / "metrics.csv"
    if args.resume:
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but missing {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("config_sha256") != config_hash:
            raise ValueError("Resume config differs from the saved SSL run.")
        student.load_state_dict(state["student"])
        saved_optimizer = state["optimizer"]
        if len(saved_optimizer.get("param_groups", [])) == len(optimizer.param_groups):
            optimizer.load_state_dict(saved_optimizer)
        elif len(saved_optimizer.get("param_groups", [])) == 2:
            legacy_optimizer = _build_optimizer(student, config, legacy_layout=True)
            legacy_optimizer.load_state_dict(saved_optimizer)
            for parameter, parameter_state in legacy_optimizer.state.items():
                optimizer.state[parameter] = parameter_state
            print(
                "Migrated legacy two-group AdamW state to bias/norm-excluded weight decay groups.",
                flush=True,
            )
        else:
            raise ValueError("Saved SSL optimizer has an unsupported parameter-group layout.")
        if scaler.is_enabled() and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        if state.get("data_generator_state") is not None:
            generator.set_state(state["data_generator_state"])
        start_epoch, best_val = int(state["epoch"]) + 1, float(state["best_val_loss"])
        saved_validation_contract = state.get("validation_sampling_contract")
        if saved_validation_contract is None:
            # Epochs from the legacy sorted validation sampler remain valid
            # training epochs, but their biased scores must not select "best".
            best_val = float("inf")
            print(
                "Legacy checkpoint detected: resetting best SSL validation score for the "
                f"corrected {VALIDATION_SAMPLING_CONTRACT} sampler.",
                flush=True,
            )
        elif saved_validation_contract != VALIDATION_SAMPLING_CONTRACT:
            raise ValueError(
                "Resume validation sampling contract differs from the current implementation: "
                f"{saved_validation_contract!r} != {VALIDATION_SAMPLING_CONTRACT!r}."
            )
        metric_epoch = _last_metric_epoch(metrics_path)
        if metric_epoch != int(state["epoch"]):
            raise ValueError(
                "Resume metrics/checkpoint transaction is inconsistent: "
                f"metrics epoch={metric_epoch}, checkpoint epoch={state['epoch']}."
            )
        resumed_drift = float(relative_anchor_loss(student, anchors).detach())
        if not math.isfinite(resumed_drift) or resumed_drift <= 0.0:
            raise ValueError(
                "Saved SSL state has no finite positive backbone drift from the initialization; "
                "refusing to resume an invalid frozen-backbone run."
            )
        if start_epoch > epochs:
            raise ValueError(f"Resume checkpoint epoch exceeds configured epochs ({start_epoch}>{epochs}).")

    needs_header = not metrics_path.is_file()
    metrics_handle = metrics_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(metrics_handle, fieldnames=["epoch", "train_total", "train_feature", "train_anchor", "val_total", "val_feature", "val_anchor", "backbone_lr", "projector_lr"])
    if needs_header:
        writer.writeheader()
    try:
        for epoch in range(start_epoch, epochs):
            student.train()
            sums = [0.0, 0.0, 0.0]
            # Epoch-specific sampling and worker seeds make epoch N identical
            # whether reached continuously or after --resume.
            generator.manual_seed(int(config["subset_seed"]) + epoch)
            train_iterator = iter(train_loader)
            for index in range(epoch_size):
                batch_data = next(train_iterator)
                images = batch_data["image"].to(device, dtype=torch.float32, non_blocking=True)
                global_step = epoch * epoch_size + index
                factor = _lr_factor(global_step, total_steps, warmup_steps)
                for group in optimizer.param_groups:
                    base_lr = (
                        float(config["backbone_lr"])
                        if group.get("role") == "backbone"
                        else float(config["projector_lr"])
                    )
                    group["lr"] = base_lr * factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=precision != "fp32", dtype=_autocast_dtype(precision)):
                    with torch.no_grad():
                        teacher_tokens = teacher.forward_features(images)
                    student_tokens = student(images)
                    total, feature, anchor = anchored_distillation_loss(
                        student_tokens, teacher_tokens, student, anchors,
                        anchor_weight=float(config["anchor_weight"]),
                        scale_weights=config.get("scale_weights"),
                    )
                if scaler.is_enabled():
                    scaler.scale(total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.get("clip_grad_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), float(config.get("clip_grad_norm", 1.0)))
                    optimizer.step()
                for target, value in zip(range(3), (total, feature, anchor)):
                    sums[target] += float(value.detach())
                if (index + 1) % 100 == 0 or index + 1 == epoch_size:
                    print(f"epoch {epoch + 1}/{epochs} batch {index + 1}/{epoch_size} total={sums[0] / (index + 1):.5f}", flush=True)

            student.eval()
            val_sums = [0.0, 0.0, 0.0]
            with torch.inference_mode():
                for index, batch_data in enumerate(val_loader):
                    if index >= val_batches:
                        break
                    images = batch_data["image"].to(device, dtype=torch.float32, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=precision != "fp32", dtype=_autocast_dtype(precision)):
                        teacher_tokens = teacher.forward_features(images)
                        values = anchored_distillation_loss(
                            student(images), teacher_tokens, student, anchors,
                            anchor_weight=float(config["anchor_weight"]),
                            scale_weights=config.get("scale_weights"),
                        )
                    for target, value in enumerate(values):
                        val_sums[target] += float(value)
            train_means = [value / epoch_size for value in sums]
            val_means = [value / val_batches for value in val_sums]
            row = {
                "epoch": epoch, "train_total": train_means[0], "train_feature": train_means[1], "train_anchor": train_means[2],
                "val_total": val_means[0], "val_feature": val_means[1], "val_anchor": val_means[2],
                "backbone_lr": next(group["lr"] for group in optimizer.param_groups if group.get("role") == "backbone"),
                "projector_lr": next(group["lr"] for group in optimizer.param_groups if group.get("role") == "projector"),
            }
            writer.writerow(row)
            metrics_handle.flush()
            metadata = {
                "created_utc": datetime.now(timezone.utc).isoformat(), "epoch": epoch,
                "config_sha256": config_hash, "config": config, "validation_only": True,
                "final_test_used": False, "train_images": len(train_data), "val_images": len(val_data),
                "teacher_model": str(config["teacher_model"]), "init_model": str(init_model),
                "trainable_backbone_parameters": trainable_backbone_parameters,
                "total_backbone_parameters": total_backbone_parameters,
                "backbone_relative_anchor_loss": val_means[2],
                "validation_sampling_contract": VALIDATION_SAMPLING_CONTRACT,
                "ms_ssl_manifest_sha256": manifest_sha256,
                "runtime_versions": {
                    "torch": torch.__version__,
                    "timm": __import__("timm").__version__,
                    "ultralytics": __import__("ultralytics").__version__,
                },
                "optimizer_contract": "adamw_kernel_matrix_decay_bias_norm_no_decay_v1",
            }
            _atomic_torch_save(
                make_artifact(student, init_model=init_model, metadata=metadata),
                output_dir / "anchored_ssl_backbone_last.pt",
            )
            if val_means[0] < best_val:
                best_val = val_means[0]
                _atomic_torch_save(
                    make_artifact(student, init_model=init_model, metadata={**metadata, "best_val_loss": best_val}),
                    output_dir / "anchored_ssl_backbone_best.pt",
                )
            _save_training_state(
                state_path, student=student, optimizer=optimizer, scaler=scaler, generator=generator,
                epoch=epoch, best=best_val, config_hash=config_hash,
            )
            print(f"epoch {epoch + 1}: val_total={val_means[0]:.5f}, best={best_val:.5f}", flush=True)
    finally:
        metrics_handle.close()
    best_artifact = output_dir / "anchored_ssl_backbone_best.pt"
    if not best_artifact.is_file() or not state_path.is_file():
        raise RuntimeError("SSL loop ended without the required best artifact and training state.")
    final_state = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(final_state.get("epoch", -1)) != epochs - 1:
        raise RuntimeError(
            f"SSL loop ended at epoch {final_state.get('epoch')}; expected {epochs - 1}. "
            "No completion marker was written."
        )
    _atomic_json(
        completion_path,
        {
            "complete": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "epochs_completed": epochs,
            "best_val_loss": best_val,
            "best_artifact": str(best_artifact.resolve()),
            "best_artifact_sha256": _file_sha256(best_artifact),
            "final_test_used": False,
            "validation_sampling_contract": VALIDATION_SAMPLING_CONTRACT,
            "ms_ssl_manifest_sha256": manifest_sha256,
            "optimizer_contract": "adamw_kernel_matrix_decay_bias_norm_no_decay_v1",
        },
    )
    print(f"Anchored SSL artifact: {best_artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
