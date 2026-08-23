"""Matched multi-seed benchmark for COCO, EMA-adapted, and random YOLO11-OBB.

The runner deliberately trains every method before touching the independent final test
split. It writes per-seed metrics, aggregate mean/std, and paired EMA-minus-
COCO deltas. It does not call the legacy DINO/SparK evaluation scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poultry_geo.progress import CommandProgress
from ultralytics import YOLO

METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched multi-seed COCO-vs-EMA-vs-random YOLO11-OBB benchmark."
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "poultry_80_10_10.yaml"))
    parser.add_argument(
        "--final-test-data",
        default=str(PROJECT_ROOT / "temp" / "poultry_houses_test" / "data.yaml"),
        help="Independent labeled dataset YAML used only for final reporting.",
    )
    parser.add_argument("--final-test-split", default="test")
    parser.add_argument("--unlabeled-root", default=r"E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg")
    parser.add_argument("--unlabeled-catalog", default=None)
    parser.add_argument("--model", default="yolo11l-obb.pt")
    parser.add_argument("--random-model", default="yolo11l-obb.yaml")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs" / "publication_ema_10pct_fixed"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 7, 42])
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--adapt-epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--unlabeled-batch", type=int, default=16)
    parser.add_argument("--test-batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--unlabeled-imgsz", type=int, default=448)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--unlabeled-workers", type=int, default=2)
    parser.add_argument("--unlabeled-limit", type=int, default=250000, help="Maximum deterministic NAIP tiles kept in memory (0 disables the limit).")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--include-random", action="store_true", help="Also train a random-initialized control.")
    parser.add_argument("--resume-ema", action="store_true", help="Resume incomplete EMA stage-1 runs when possible.")
    parser.add_argument("--skip-training", action="store_true", help="Only evaluate already-existing checkpoints.")
    parser.add_argument("--skip-test", action="store_true", help="Train and aggregate only; do not evaluate test split.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing best.pt instead of retraining that method/seed.")
    parser.add_argument("--no-plots", action="store_true", help="Disable final-test confusion/PR plots.")
    parser.add_argument(
        "--allow-test-overlap",
        action="store_true",
        help="Allow exact image overlap with training data (exploratory only; not publication-valid).",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the publication-level Rich progress bar.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("At least one seed is required.")
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if args.imgsz % 32 or args.unlabeled_imgsz % 32:
        raise ValueError("imgsz and unlabeled-imgsz must be divisible by 32.")
    if args.batch <= 0 or args.unlabeled_batch <= 0 or args.test_batch <= 0:
        raise ValueError("batch sizes must be positive.")
    if args.unlabeled_workers < 0 or args.unlabeled_limit < 0 or args.prefetch_factor <= 0:
        raise ValueError("unlabeled-workers and unlabeled-limit cannot be negative; prefetch-factor must be positive.")
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")
    if not Path(args.final_test_data).exists():
        raise FileNotFoundError(f"Final test dataset YAML not found: {args.final_test_data}")
    if not args.skip_training and not Path(args.unlabeled_root).exists():
        raise FileNotFoundError(f"Unlabeled NAIP root not found: {args.unlabeled_root}")


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".bmp", ".webp"}


def _dataset_image_paths(data_yaml: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> list[Path]:
    """Resolve all train/val/test image paths referenced by a YOLO YAML."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - ultralytics normally brings PyYAML
        raise RuntimeError("PyYAML is required for the dataset-overlap audit.") from exc

    data_yaml = data_yaml.expanduser().resolve()
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root_value = payload.get("path")
    root = Path(root_value) if root_value else data_yaml.parent
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    paths: list[Path] = []
    for split in splits:
        values = payload.get(split)
        if values is None:
            continue
        if isinstance(values, (str, Path)):
            values = [values]
        for value in values:
            candidate = Path(os.path.expandvars(str(value)))
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(candidate)
            elif candidate.is_file() and candidate.suffix.lower() in {".txt", ".list"}:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    item = line.strip()
                    if not item or item.startswith("#"):
                        continue
                    manifest_item = Path(os.path.expandvars(item))
                    if not manifest_item.is_absolute():
                        manifest_item = candidate.parent / manifest_item
                    manifest_item = manifest_item.resolve()
                    if manifest_item.is_file() and manifest_item.suffix.lower() in IMAGE_SUFFIXES:
                        paths.append(manifest_item)
            elif candidate.is_dir():
                paths.extend(
                    path for path in candidate.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
    return sorted(set(paths))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_test_overlap(
    training_yaml: Path,
    final_test_yaml: Path,
    *,
    final_test_split: str = "test",
) -> dict[str, Any]:
    """Compare final-test images only with the splits used for training.

    A YOLO YAML may include a test entry for convenience. That entry is
    not part of the training population and must not be counted as leakage
    when it points to the same final-test directory.
    """
    train_files = _dataset_image_paths(training_yaml, ("train", "val"))
    final_files = _dataset_image_paths(final_test_yaml, (final_test_split,))
    by_name: dict[str, list[Path]] = {}
    for path in train_files:
        by_name.setdefault(path.name.lower(), []).append(path)

    filename_matches = 0
    exact_matches: list[dict[str, str]] = []
    for final_path in final_files:
        candidates = by_name.get(final_path.name.lower(), [])
        filename_matches += len(candidates)
        for train_path in candidates:
            if final_path.stat().st_size == train_path.stat().st_size and _sha256(final_path) == _sha256(train_path):
                exact_matches.append({"final": str(final_path), "training": str(train_path)})
    return {
        "training_splits_audited": ["train", "val"],
        "final_test_split_audited": final_test_split,
        "training_image_count": len(train_files),
        "final_test_image_count": len(final_files),
        "same_filename_matches": filename_matches,
        "exact_sha256_matches": len(exact_matches),
        "exact_matches": exact_matches,
    }

def prepare_isolated_training_yaml(
    source_yaml: Path,
    final_test_yaml: Path,
    overlap: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create reproducible train/val manifests with final-test duplicates removed."""
    import yaml

    blocked = {Path(item["training"]).resolve() for item in overlap["exact_matches"]}
    train_files = _dataset_image_paths(source_yaml, ("train",))
    val_files = _dataset_image_paths(source_yaml, ("val",))
    safe_train = [path for path in train_files if path not in blocked]
    safe_val = [path for path in val_files if path not in blocked]
    if not safe_train or not safe_val:
        raise RuntimeError("Removing final-test duplicates left an empty train or validation split.")

    train_manifest = output_dir / "isolated_train_images.txt"
    val_manifest = output_dir / "isolated_val_images.txt"
    train_manifest.write_text(
        chr(10).join(path.as_posix() for path in safe_train) + chr(10),
        encoding="utf-8",
    )
    val_manifest.write_text(
        chr(10).join(path.as_posix() for path in safe_val) + chr(10),
        encoding="utf-8",
    )
    source_payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    isolated_payload = {
        "train": train_manifest.as_posix(),
        "val": val_manifest.as_posix(),
        "nc": source_payload.get("nc", len(source_payload.get("names", []))),
        "names": source_payload.get("names", {}),
    }
    isolated_yaml = output_dir / "isolated_training_data.yaml"
    isolated_yaml.write_text(
        yaml.safe_dump(isolated_payload, sort_keys=False),
        encoding="utf-8",
    )
    stats = {
        "source_data": str(source_yaml),
        "final_test_data": str(final_test_yaml),
        "blocked_exact_images": len(blocked),
        "source_train_images": len(train_files),
        "isolated_train_images": len(safe_train),
        "source_val_images": len(val_files),
        "isolated_val_images": len(safe_val),
        "isolated_data": str(isolated_yaml),
    }
    return isolated_yaml, stats

def best_checkpoint(run_dir: Path) -> Path:
    return run_dir / "weights" / "best.pt"


def _assert_fresh_training_dir(run_dir: Path, *, allow_existing: bool) -> None:
    """Prevent silent mixing of checkpoints/results from different experiments."""
    if allow_existing or not run_dir.exists():
        return
    artifacts = [
        run_dir / "weights",
        run_dir / "results.csv",
        run_dir / "args.yaml",
        run_dir / "ema_handoff.json",
    ]
    if any(path.exists() for path in artifacts):
        raise RuntimeError(
            f"Training output already contains artifacts: {run_dir}. "
            "Choose a new --output-dir, or explicitly use --reuse-existing/--resume-ema."
        )


def train_supervised(args: argparse.Namespace, *, seed: int, method: str, model_path: str, pretrained: bool, run_dir: Path) -> Path:
    checkpoint = best_checkpoint(run_dir)
    if checkpoint.exists() and args.reuse_existing:
        print(f"[{method} seed={seed}] reusing {checkpoint}")
        return checkpoint
    if args.skip_training:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for --skip-training: {checkpoint}")
        return checkpoint

    _assert_fresh_training_dir(run_dir, allow_existing=args.reuse_existing)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{method} seed={seed}] training supervised model")
    model = YOLO(model_path)
    model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.finetune_epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3.0,
        weight_decay=5e-4,
        cos_lr=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        amp=True,
        pretrained=pretrained,
        seed=seed,
        patience=max(30, min(100, args.finetune_epochs // 2)),
        val=True,
        plots=False,
        project=str(run_dir.parent.resolve()),
        name=run_dir.name,
        exist_ok=True,
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training finished without {checkpoint}")
    return checkpoint


def train_ema(args: argparse.Namespace, *, seed: int, run_dir: Path) -> Path:
    stage2_dir = run_dir.with_name(run_dir.name + "_finetune")
    checkpoint = best_checkpoint(stage2_dir)
    handoff_marker = stage2_dir / "ema_handoff.json"
    if checkpoint.exists() and args.reuse_existing:
        if not handoff_marker.exists():
            raise RuntimeError(
                f"Cannot reuse {checkpoint}: missing verified EMA handoff marker. "
                "Use a new --output-dir for a publication-valid rerun."
            )
        else:
            try:
                marker = json.loads(handoff_marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Invalid EMA handoff marker: {handoff_marker}") from exc
            if marker.get("version") != 1 or marker.get("weights_preserved") is not True:
                raise RuntimeError(f"EMA handoff marker is not valid: {handoff_marker}")
            print(f"[ema seed={seed}] reusing verified checkpoint {checkpoint}")
            return checkpoint
    if args.skip_training:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for --skip-training: {checkpoint}")
        return checkpoint

    _assert_fresh_training_dir(run_dir, allow_existing=args.resume_ema)
    if not args.resume_ema:
        _assert_fresh_training_dir(stage2_dir, allow_existing=False)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_ema_adapt.py"),
        "--unlabeled-root", str(Path(args.unlabeled_root).resolve()),
        "--data", str(Path(args.data).resolve()),
        "--output-dir", str(run_dir.resolve()),
        "--model", args.model,
        "--adapt-epochs", str(args.adapt_epochs),
        "--finetune-epochs", str(args.finetune_epochs),
        "--fraction", str(args.fraction),
        "--batch", str(args.batch),
        "--unlabeled-batch", str(args.unlabeled_batch),
        "--imgsz", str(args.imgsz),
        "--unlabeled-imgsz", str(args.unlabeled_imgsz),
        "--workers", str(args.workers),
        "--unlabeled-workers", str(args.unlabeled_workers),
        "--unlabeled-limit", str(args.unlabeled_limit),
        "--prefetch-factor", str(args.prefetch_factor),
        "--device", str(args.device),
        "--seed", str(seed),
        "--exist-ok",
    ]
    if args.unlabeled_catalog:
        cmd.extend(["--unlabeled-catalog", str(Path(args.unlabeled_catalog).resolve())])
    if args.resume_ema:
        cmd.append("--resume")
    print(f"[ema seed={seed}] training adaptation and supervised fine-tuning")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    if not checkpoint.exists():
        raise FileNotFoundError(f"EMA training finished without {checkpoint}")
    if not handoff_marker.exists():
        raise RuntimeError(f"EMA training finished without a verified handoff marker: {handoff_marker}")
    return checkpoint


def evaluate_checkpoint(
    args: argparse.Namespace,
    *,
    method: str,
    seed: int,
    checkpoint: Path,
    eval_root: Path,
) -> dict[str, Any]:
    print(f"[test] {method} seed={seed}: {checkpoint}")
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(Path(args.final_test_data).resolve()),
        split=args.final_test_split,
        imgsz=args.imgsz,
        batch=args.test_batch,
        device=args.device,
        workers=args.workers,
        plots=not args.no_plots,
        project=str(eval_root.resolve()),
        name=f"{method}_seed{seed}",
        exist_ok=True,
    )
    values = getattr(result, "results_dict", {})
    row: dict[str, Any] = {
        "method": method,
        "seed": int(seed),
        "checkpoint": str(checkpoint.resolve()),
        "test_data": str(Path(args.final_test_data).resolve()),
        "test_split": args.final_test_split,
    }
    for key in METRIC_KEYS:
        value = values.get(key)
        row[key] = float(value) if value is not None else None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {"method": method, "n_seeds": len(subset)}
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in subset if row.get(key) is not None]
            item[f"{key}/mean"] = mean(vals) if vals else None
            item[f"{key}/std"] = stdev(vals) if len(vals) > 1 else 0.0 if vals else None
            item[f"{key}/min"] = min(vals) if vals else None
            item[f"{key}/max"] = max(vals) if vals else None
        output.append(item)
    return output


def paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {int(row["seed"]): row for row in rows if row["method"] == "ultralytics_pretrained"}
    adapted = {int(row["seed"]): row for row in rows if row["method"] == "ema_adapted"}
    output = []
    for seed in sorted(set(baseline) & set(adapted)):
        item = {"seed": seed}
        for key in METRIC_KEYS:
            item[key] = float(adapted[seed][key]) - float(baseline[seed][key])
        output.append(item)
    if output:
        mean_row = {"seed": "mean"}
        std_row = {"seed": "std"}
        for key in METRIC_KEYS:
            vals = [float(row[key]) for row in output]
            mean_row[key] = mean(vals)
            std_row[key] = stdev(vals) if len(vals) > 1 else 0.0
        output.extend([mean_row, std_row])
    return output


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
    }
    try:
        import torch
        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        metadata["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - diagnostic only
        metadata["torch_error"] = str(exc)
    try:
        import ultralytics
        metadata["ultralytics"] = getattr(ultralytics, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        metadata["ultralytics_error"] = str(exc)
    try:
        metadata["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        metadata["git_commit"] = None
    return metadata

def run_publication_phase(
    progress: Any,
    completed: int,
    total: int,
    label: str,
    action: Callable[[], Any],
) -> tuple[Any, int]:
    if progress is not None:
        progress.update(completed, total, description=label, detail="[bold yellow]Running")
    try:
        result = action()
    except BaseException:
        if progress is not None:
            progress.finish(success=False)
        raise
    completed += 1
    if progress is not None:
        progress.update(completed, total, description=label, detail="[bold green]Completed")
    return result, completed

def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_data = Path(args.data).resolve()
    final_test_data = Path(args.final_test_data).resolve()
    overlap = audit_test_overlap(source_data, final_test_data, final_test_split=args.final_test_split)
    print(f"[preflight] training images={overlap['training_image_count']} final-test images={overlap['final_test_image_count']} exact SHA-256 overlaps={overlap['exact_sha256_matches']}")
    if overlap["training_image_count"] <= 0 or overlap["final_test_image_count"] <= 0:
        raise RuntimeError("Preflight resolved an empty training/validation or final-test split; refusing to train/evaluate.")
    if overlap["exact_sha256_matches"] and args.allow_test_overlap:
        isolation = {
            "mode": "allow_test_overlap",
            "warning": "Exploratory only; final metrics are not publication-valid.",
            "source_data": str(source_data),
        }
    else:
        isolated_yaml, isolation = prepare_isolated_training_yaml(source_data, final_test_data, overlap, output_dir)
        args.data = str(isolated_yaml)
        isolation["mode"] = "isolated_manifests"
        if overlap["exact_sha256_matches"]:
            print(f"[preflight] removed {overlap['exact_sha256_matches']} exact final-test duplicates from train/val manifests")
    overlap["isolation"] = isolation
    (output_dir / "dataset_overlap_audit.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    method_count = 2 + (1 if args.include_random else 0)
    train_phase_count = len(args.seeds) * method_count
    eval_phase_count = 0 if args.skip_test else train_phase_count
    total_phases = train_phase_count + eval_phase_count
    progress = None if args.no_progress else CommandProgress("publication-experiment", verbose=True)
    if progress is not None:
        progress.start()
        progress.update(0, total_phases, description="Publication • Preflight", detail="[bold cyan]Ready")
    completed = 0
    methods: list[tuple[str, int, Path]] = []

    # Train all methods for all seeds first. Final-test evaluation happens only after
    # this loop, preventing accidental test-driven model selection.
    for seed in args.seeds:
        baseline_dir = output_dir / f"ultralytics_pretrained_seed{seed}"
        _, completed = run_publication_phase(
            progress,
            completed,
            total_phases,
            f"Seed {seed} • COCO baseline",
            lambda baseline_dir=baseline_dir, seed=seed: train_supervised(
                args,
                seed=seed,
                method="ultralytics_pretrained",
                model_path=args.model,
                pretrained=True,
                run_dir=baseline_dir,
            ),
        )
        methods.append(("ultralytics_pretrained", seed, best_checkpoint(baseline_dir)))

        ema_dir = output_dir / f"ema_adapted_seed{seed}"
        _, completed = run_publication_phase(
            progress,
            completed,
            total_phases,
            f"Seed {seed} • EMA adaptation + fine-tune",
            lambda ema_dir=ema_dir, seed=seed: train_ema(args, seed=seed, run_dir=ema_dir),
        )
        methods.append(("ema_adapted", seed, best_checkpoint(ema_dir.with_name(ema_dir.name + "_finetune"))))

        if args.include_random:
            random_dir = output_dir / f"random_init_seed{seed}"
            _, completed = run_publication_phase(
                progress,
                completed,
                total_phases,
                f"Seed {seed} • Random initialization",
                lambda random_dir=random_dir, seed=seed: train_supervised(
                    args,
                    seed=seed,
                    method="random_init",
                    model_path=args.random_model,
                    pretrained=False,
                    run_dir=random_dir,
                ),
            )
            methods.append(("random_init", seed, best_checkpoint(random_dir)))

    if args.skip_test:
        if progress is not None:
            progress.finish(success=True)
        print("Training complete; test evaluation skipped by request.")
        return 0

    rows = []
    eval_root = output_dir / "test_evaluations"
    for method, seed, checkpoint in methods:
        row, completed = run_publication_phase(
            progress,
            completed,
            total_phases,
            f"Seed {seed} • Final test ({method})",
            lambda method=method, seed=seed, checkpoint=checkpoint: evaluate_checkpoint(
                args,
                method=method,
                seed=seed,
                checkpoint=checkpoint,
                eval_root=eval_root,
            ),
        )
        rows.append(row)
    aggregates = aggregate(rows)
    deltas = paired_deltas(rows)
    write_csv(output_dir / "per_seed_test_metrics.csv", rows)
    write_csv(output_dir / "aggregate_test_metrics.csv", aggregates)
    write_csv(output_dir / "paired_ema_minus_coco.csv", deltas)
    payload = {
        "environment": environment_metadata(),
        "protocol": {
            "source_data": str(source_data),
            "data": str(Path(args.data).resolve()),
            "final_test_data": str(Path(args.final_test_data).resolve()),
            "final_test_split": args.final_test_split,
            "fraction": args.fraction,
            "seeds": args.seeds,
            "adapt_epochs": args.adapt_epochs,
            "finetune_epochs": args.finetune_epochs,
            "batch": args.batch,
            "unlabeled_batch": args.unlabeled_batch,
            "unlabeled_limit": args.unlabeled_limit,
            "workers": args.workers,
            "unlabeled_workers": args.unlabeled_workers,
            "prefetch_factor": args.prefetch_factor,
            "test_used_for_selection": False,
            "final_test_used_only_after_training": True,
        },
        "per_seed": rows,
        "aggregate": aggregates,
        "paired_ema_minus_coco": deltas,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if progress is not None:
        progress.finish(success=True)

    print("\nFinal independent final test summary")
    for row in aggregates:
        key = "metrics/mAP50-95(B)"
        print(f"{row['method']:<24} mean={row[f'{key}/mean']:.5f} std={row[f'{key}/std']:.5f} n={row['n_seeds']}")
    if deltas:
        print(f"EMA - COCO mAP50-95: mean={deltas[-2]['metrics/mAP50-95(B)']:+.5f} std={deltas[-1]['metrics/mAP50-95(B)']:.5f}")
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


