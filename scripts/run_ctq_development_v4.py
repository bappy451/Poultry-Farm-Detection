"""Run CTQ development-v4 audit, validation diagnostics, or test-v2 creation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def command(parts: list[str]) -> None:
    print("$ " + " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_development_v4.yaml")
    parser.add_argument("--phase", choices=("leakage", "diagnostics", "create-test-v2"), required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = {"training_data", "locked_test_data", "locked_test_split", "unlabeled_root", "publication_protocol", "teacher_method", "output_dir", "diagnostics", "test_v2"}
    if required - config.keys():
        raise ValueError(f"Missing config keys: {sorted(required - config.keys())}")
    training, locked_test, unlabeled, protocol, output = (resolve(config[key]) for key in ("training_data", "locked_test_data", "unlabeled_root", "publication_protocol", "output_dir"))
    for path in (training, locked_test, unlabeled, protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "leakage":
        test_manifest = config.get("locked_test_source_manifest")
        output_path = output / "locked_test_unlabeled_audit.json"
        if output_path.exists():
            raise FileExistsError(f"Audit output exists: {output_path}")
        parts = [sys.executable, "-m", "scripts.audit_ctq_unlabeled_leakage", "--training-data", str(training), "--final-test-data", str(locked_test), "--final-test-split", str(config["locked_test_split"]), "--unlabeled-root", str(unlabeled), "--output", str(output_path)]
        if test_manifest:
            parts.extend(["--test-source-manifest", str(resolve(test_manifest))])
        command(parts)
    elif args.phase == "diagnostics":
        settings = config["diagnostics"]
        directory = output / "validation_pseudo_diagnostics"
        if directory.exists():
            raise FileExistsError(f"Diagnostics output exists: {directory}")
        command([sys.executable, "-m", "scripts.diagnose_ctq_pseudo_labels", "--protocol-dir", str(protocol), "--data", str(training), "--split", str(settings["split"]), "--method", str(config["teacher_method"]), "--output-dir", str(directory), "--thresholds", str(settings["thresholds"]), "--precision-target", str(settings["precision_target"]), "--min-predictions", str(settings["min_predictions"]), "--fallback", str(settings["fallback"]), "--iou-match", str(settings["iou_match"]), "--imgsz", str(settings["imgsz"]), "--batch", str(settings["batch"]), "--workers", str(settings["workers"]), "--device", str(settings["device"])])
    else:
        labeled_manifest = config.get("labeled_source_manifest")
        if not labeled_manifest:
            raise ValueError("create-test-v2 requires labeled_source_manifest in the config.")
        split = config["test_v2"]
        command([sys.executable, "-m", "scripts.create_ctq_source_disjoint_split", "--data", str(training), "--source-manifest", str(resolve(labeled_manifest)), "--output-root", str(resolve(split["output_root"])), "--output-data", str(resolve(split["output_data"])), "--test-fraction", str(split["test_fraction"]), "--val-fraction", str(split["val_fraction"]), "--seed", str(split["split_seed"]), "--unlabeled-root", str(unlabeled), "--unlabeled-catalog", str(resolve(split["unlabeled_catalog"]))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
