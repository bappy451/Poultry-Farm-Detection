"""Train the frozen YOLO26x-OBB MuSGD validation challenger.

This entry point intentionally exposes only one training recipe.  The
publication protocol owns run identity, reuse, and resume decisions; this
module independently rechecks the model/data hashes and the saved Ultralytics
arguments so an accidental CLI override cannot silently change the method.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load

from scripts.checkpoint_retry import RetryingOBBTrainer


METRIC = "metrics/mAP50-95(B)"
PIPELINE = "official_yolo26x_obb_musgd_full_al_validation_v1"
OPTIMIZER_CONTRACT = "ultralytics_8.4.63_musgd_lr001_m09_wd0005_nbs64_v1"
RECIPE: dict[str, Any] = {
    "epochs": 100,
    "batch": 8,
    "imgsz": 640,
    "workers": 8,
    "device": "0",
    "optimizer": "MuSGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "nbs": 64,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.0,
    "cos_lr": True,
    "close_mosaic": 10,
    "patience": 50,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "angle": 1.0,
}


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--init-model-sha256", required=True)
    parser.add_argument("--protocol-config-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    parser.add_argument("--batch", type=int, default=RECIPE["batch"])
    parser.add_argument("--imgsz", type=int, default=RECIPE["imgsz"])
    parser.add_argument("--workers", type=int, default=RECIPE["workers"])
    parser.add_argument("--device", default=RECIPE["device"])
    parser.add_argument("--lr0", type=float, default=RECIPE["lr0"])
    parser.add_argument("--momentum", type=float, default=RECIPE["momentum"])
    parser.add_argument("--weight-decay", type=float, default=RECIPE["weight_decay"])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _assert_frozen_cli(args: argparse.Namespace) -> None:
    exact = {
        "epochs": RECIPE["epochs"], "batch": RECIPE["batch"],
        "imgsz": RECIPE["imgsz"], "workers": RECIPE["workers"],
        "device": RECIPE["device"],
    }
    for name, expected in exact.items():
        observed = getattr(args, name)
        if str(observed) != str(expected):
            raise ValueError(f"Frozen recipe requires {name}={expected!r}; received {observed!r}.")
    for name in ("lr0", "momentum", "weight_decay"):
        observed = getattr(args, name)
        if not _same_number(observed, RECIPE[name]):
            raise ValueError(f"Frozen recipe requires {name}={RECIPE[name]}; received {observed!r}.")
    if args.seed not in {7, 17, 42}:
        raise ValueError("Frozen challenger seed must be one of 7, 17, or 42.")
    for name in ("data_sha256", "init_model_sha256", "protocol_config_sha256"):
        value = str(getattr(args, name)).lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def expected_train_args(
    *, data: Path, init_model: Path, seed: int, output_dir: Path,
) -> dict[str, Any]:
    """Return the immutable Ultralytics arguments used for training/auditing."""

    return {
        "task": "obb", "data": str(data), "model": str(init_model),
        "epochs": RECIPE["epochs"], "batch": RECIPE["batch"],
        "imgsz": RECIPE["imgsz"], "workers": RECIPE["workers"],
        "device": RECIPE["device"], "seed": seed, "deterministic": True,
        "optimizer": RECIPE["optimizer"], "lr0": RECIPE["lr0"],
        "lrf": RECIPE["lrf"], "momentum": RECIPE["momentum"],
        "weight_decay": RECIPE["weight_decay"], "nbs": RECIPE["nbs"],
        "warmup_epochs": RECIPE["warmup_epochs"],
        "warmup_momentum": RECIPE["warmup_momentum"],
        "warmup_bias_lr": RECIPE["warmup_bias_lr"], "cos_lr": True,
        "close_mosaic": RECIPE["close_mosaic"], "patience": RECIPE["patience"],
        "hsv_h": RECIPE["hsv_h"], "hsv_s": RECIPE["hsv_s"],
        "hsv_v": RECIPE["hsv_v"], "degrees": RECIPE["degrees"],
        "translate": RECIPE["translate"], "scale": RECIPE["scale"],
        "flipud": RECIPE["flipud"], "fliplr": RECIPE["fliplr"],
        "mosaic": RECIPE["mosaic"], "box": RECIPE["box"],
        "cls": RECIPE["cls"], "dfl": RECIPE["dfl"], "angle": RECIPE["angle"],
        "amp": True, "pretrained": str(init_model), "val": True,
        "fraction": 1.0, "cache": False, "rect": False, "multi_scale": 0.0,
        "compile": False, "project": str(output_dir.parent), "name": output_dir.name,
        "exist_ok": True, "plots": True,
    }


def audit_train_args(
    actual: dict[str, Any], expected: dict[str, Any], *, run_dir: Path,
) -> None:
    """Require an exact recipe, allowing only Ultralytics' resume model rewrite."""

    path_fields = {"data", "pretrained", "project"}
    bool_fields = {
        "deterministic", "cos_lr", "amp", "val", "cache", "rect",
        "compile", "exist_ok", "plots",
    }
    numeric_fields = {
        "epochs", "batch", "imgsz", "workers", "seed", "lr0", "lrf",
        "momentum", "weight_decay", "nbs", "warmup_epochs", "warmup_momentum",
        "warmup_bias_lr", "close_mosaic", "patience", "hsv_h", "hsv_s",
        "hsv_v", "degrees", "translate", "scale", "flipud", "fliplr",
        "mosaic", "box", "cls", "dfl", "angle", "fraction", "multi_scale",
    }
    for name, expected_value in expected.items():
        observed = actual.get(name)
        if name == "model":
            allowed = {
                Path(str(expected_value)).resolve(),
                (run_dir / "weights" / "last.pt").resolve(),
            }
            matches = Path(str(observed)).resolve() in allowed
        elif name in path_fields:
            matches = Path(str(observed)).resolve() == Path(str(expected_value)).resolve()
        elif name in bool_fields:
            matches = observed is expected_value
        elif name in numeric_fields:
            matches = _same_number(observed, expected_value)
        else:
            matches = str(observed) == str(expected_value)
        if not matches:
            raise ValueError(
                f"Saved train arg {name!r} is {observed!r}; expected {expected_value!r}."
            )


def _checkpoint_metadata(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    checkpoint, _ = torch_safe_load(str(path))
    try:
        args = checkpoint.get("train_args")
        metrics = checkpoint.get("train_metrics")
        if not isinstance(args, dict) or not isinstance(metrics, dict):
            raise ValueError(f"Checkpoint lacks train_args/train_metrics: {path}")
        return str(checkpoint.get("version", "")), dict(args), dict(metrics)
    finally:
        del checkpoint
        gc.collect()


def _best_validation(run_dir: Path) -> tuple[int, float]:
    result_path = run_dir / "results.csv"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[tuple[int, float]] = []
    for row in rows:
        raw = row.get(METRIC)
        if raw in (None, ""):
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {METRIC} in {result_path}.")
        candidates.append((int(float(row["epoch"])), value))
    if not candidates:
        raise ValueError(f"No finite {METRIC} values in {result_path}.")
    return max(candidates, key=lambda item: item[1])


def _assert_yolo26x_obb(model: YOLO, source: Path) -> dict[str, Any]:
    core = model.model
    yaml_config = getattr(core, "yaml", {})
    if model.task != "obb" or core.__class__.__name__ != "OBBModel":
        raise ValueError(f"{source} is not an Ultralytics OBB checkpoint.")
    if not isinstance(yaml_config, dict):
        raise ValueError(f"{source} has no model YAML metadata.")
    if str(yaml_config.get("scale")) != "x" or yaml_config.get("end2end") is not True:
        raise ValueError(f"{source} is not the end-to-end YOLO26x-OBB architecture.")
    if int(yaml_config.get("reg_max", -1)) != 1:
        raise ValueError(f"{source} does not have the YOLO26 DFL-free regression head.")
    return {
        "task": model.task, "model_class": core.__class__.__name__,
        "scale": str(yaml_config["scale"]), "end2end": bool(yaml_config["end2end"]),
        "reg_max": int(yaml_config["reg_max"]),
        "parameters": sum(parameter.numel() for parameter in core.parameters()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _assert_frozen_cli(args)
    data = resolve(args.data)
    init_model = resolve(args.init_model)
    output_dir = resolve(args.output_dir)
    for path in (data, init_model):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(data) != args.data_sha256:
        raise ValueError(f"Training YAML hash mismatch: {data}")
    if sha256(init_model) != args.init_model_sha256:
        raise ValueError(f"Official initialization checkpoint hash mismatch: {init_model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / "yolo26x_challenger.json"
    if marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite completion marker: {marker_path}")
    last = output_dir / "weights" / "last.pt"
    expected = expected_train_args(
        data=data, init_model=init_model, seed=args.seed, output_dir=output_dir,
    )
    if args.resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume requested but missing {last}")
        version, saved_args, _ = _checkpoint_metadata(last)
        import ultralytics

        if version != str(ultralytics.__version__):
            raise ValueError(
                f"Resume checkpoint uses Ultralytics {version}; runtime is {ultralytics.__version__}."
            )
        audit_train_args(saved_args, expected, run_dir=output_dir)
        model = YOLO(str(last))
        _assert_yolo26x_obb(model, last)
    else:
        unexpected = [output_dir / "results.csv", output_dir / "weights" / "best.pt", last]
        if any(path.exists() for path in unexpected):
            raise FileExistsError(f"Existing partial training artifacts require a verified resume: {output_dir}")
        model = YOLO(str(init_model))
        _assert_yolo26x_obb(model, init_model)

    train_kwargs = dict(expected)
    train_kwargs.pop("task")
    train_kwargs.pop("model")
    train_kwargs["trainer"] = RetryingOBBTrainer
    train_kwargs["resume"] = str(last) if args.resume else False
    model.train(**train_kwargs)

    best = output_dir / "weights" / "best.pt"
    args_yaml = output_dir / "args.yaml"
    for path in (best, last, args_yaml, output_dir / "results.csv"):
        if not path.is_file():
            raise FileNotFoundError(f"Training completed without required artifact: {path}")
    saved_yaml = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
    if not isinstance(saved_yaml, dict):
        raise ValueError(f"Invalid Ultralytics args YAML: {args_yaml}")
    audit_train_args(saved_yaml, expected, run_dir=output_dir)
    version, checkpoint_args, checkpoint_metrics = _checkpoint_metadata(best)
    import torch
    import ultralytics

    if version != str(ultralytics.__version__):
        raise ValueError(f"Best checkpoint runtime mismatch: {version}")
    audit_train_args(checkpoint_args, expected, run_dir=output_dir)
    best_epoch, best_metric = _best_validation(output_dir)
    stored_metric = checkpoint_metrics.get(METRIC)
    if not _same_number(stored_metric, best_metric, tolerance=5e-6):
        raise ValueError(
            f"Best checkpoint stores {stored_metric!r}, but results.csv has {best_metric!r}."
        )
    payload = {
        "version": 1, "pipeline": PIPELINE,
        "optimizer_contract": OPTIMIZER_CONTRACT,
        "protocol_config_sha256": args.protocol_config_sha256,
        "data": str(data), "data_sha256": args.data_sha256,
        "init_model": str(init_model), "init_model_sha256": args.init_model_sha256,
        "best_checkpoint": str(best.resolve()), "best_checkpoint_sha256": sha256(best),
        "best_epoch": best_epoch, "best_val_map50_95": best_metric,
        "epochs": RECIPE["epochs"], "batch": RECIPE["batch"],
        "imgsz": RECIPE["imgsz"], "seed": args.seed,
        "effective_nominal_batch": RECIPE["nbs"],
        "resumed": bool(args.resume), "validation_only": True,
        "test_used_for_selection": False,
        "runtime": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "ultralytics": str(ultralytics.__version__),
        },
    }
    temporary = marker_path.with_name(marker_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(marker_path)
    print(f"Verified YOLO26x-OBB challenger checkpoint: {best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
