"""Run the hardened CTQ development phases without accessing test-v2 in training."""

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


def launch(parts: list[str]) -> None:
    print("$ " + " ".join(parts), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_development_v5.yaml")
    parser.add_argument("--phase", required=True, choices=("leakage", "diagnostics", "create-test-v2"))
    args = parser.parse_args()
    config: dict[str, Any] = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8")) or {}
    required = {"training_data", "locked_test_data", "locked_test_split", "unlabeled_root", "publication_protocol", "teacher_method", "output_dir", "diagnostics", "test_v2"}
    if required - config.keys():
        raise ValueError(f"Missing config keys: {sorted(required - config.keys())}")
    training, locked_test, unlabeled, protocol, output = (resolve(config[key]) for key in ("training_data", "locked_test_data", "unlabeled_root", "publication_protocol", "output_dir"))
    for path in (training, locked_test, unlabeled, protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "leakage":
        manifest = config.get("locked_test_source_manifest")
        if not manifest:
            raise ValueError("leakage requires locked_test_source_manifest from original NAIP provenance.")
        report = output / "locked_test_unlabeled_audit.json"
        if report.exists():
            raise FileExistsError(f"Audit output exists: {report}")
        launch([sys.executable, "-m", "scripts.audit_ctq_unlabeled_leakage_v2", "--training-data", str(training), "--final-test-data", str(locked_test), "--final-test-split", str(config["locked_test_split"]), "--unlabeled-root", str(unlabeled), "--test-source-manifest", str(resolve(manifest)), "--write-catalog", str(output / "unlabeled_without_locked_test_sources.csv"), "--output", str(report)])
    elif args.phase == "diagnostics":
        settings = config["diagnostics"]
        directory = output / "validation_pseudo_diagnostics"
        if directory.exists():
            raise FileExistsError(f"Diagnostics output exists: {directory}")
        launch([sys.executable, "-m", "scripts.diagnose_ctq_pseudo_labels", "--protocol-dir", str(protocol), "--data", str(training), "--split", str(settings["split"]), "--method", str(config["teacher_method"]), "--output-dir", str(directory), "--thresholds", str(settings["thresholds"]), "--precision-target", str(settings["precision_target"]), "--min-predictions", str(settings["min_predictions"]), "--fallback", str(settings["fallback"]), "--iou-match", str(settings["iou_match"]), "--imgsz", str(settings["imgsz"]), "--batch", str(settings["batch"]), "--workers", str(settings["workers"]), "--device", str(settings["device"])])
    else:
        manifest = config.get("labeled_source_manifest")
        if not manifest:
            raise ValueError("create-test-v2 requires labeled_source_manifest from original NAIP provenance.")
        split = config["test_v2"]
        launch([sys.executable, "-m", "scripts.create_ctq_source_disjoint_split_v2", "--data", str(training), "--source-manifest", str(resolve(manifest)), "--output-root", str(resolve(split["output_root"])), "--output-data", str(resolve(split["output_data"])), "--test-fraction", str(split["test_fraction"]), "--val-fraction", str(split["val_fraction"]), "--seed", str(split["split_seed"]), "--unlabeled-root", str(unlabeled), "--unlabeled-catalog", str(resolve(split["unlabeled_catalog"]))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
