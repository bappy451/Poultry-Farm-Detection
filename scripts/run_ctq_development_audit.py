"""Run the validation-only CTQ development audit from a YAML configuration."""

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


def load(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"training_data", "locked_test_data", "locked_test_split", "unlabeled_root", "publication_protocol", "teacher_method", "output_dir", "diagnostics", "test_v2"}
    if not isinstance(config, dict) or required - config.keys():
        missing = sorted(required - (config.keys() if isinstance(config, dict) else set()))
        raise ValueError(f"Invalid development config; missing {missing}")
    return config


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_development_v3.yaml")
    parser.add_argument("--phase", choices=("all", "leakage", "diagnostics", "create-test-v2"), default="all")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    config = load(resolve(args.config))
    output = resolve(config["output_dir"])
    training = resolve(config["training_data"])
    locked_test = resolve(config["locked_test_data"])
    unlabeled = resolve(config["unlabeled_root"])
    protocol = resolve(config["publication_protocol"])
    source_value = config.get("source_manifest")
    source_manifest = resolve(source_value) if source_value else None
    for path in (training, locked_test, unlabeled, protocol):
        if not path.exists():
            raise FileNotFoundError(path)
    if source_manifest is not None and not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    output.mkdir(parents=True, exist_ok=True)

    if args.phase in ("all", "leakage"):
        audit_output = output / "locked_test_unlabeled_audit.json"
        if not (args.reuse_existing and audit_output.is_file()):
            command = [sys.executable, "-m", "scripts.audit_ctq_unlabeled_leakage", "--training-data", str(training), "--final-test-data", str(locked_test), "--final-test-split", str(config["locked_test_split"]), "--unlabeled-root", str(unlabeled), "--output", str(audit_output)]
            if source_manifest is not None:
                command.extend(["--test-source-manifest", str(source_manifest)])
            run(command)

    if args.phase in ("all", "diagnostics"):
        diagnostics = config["diagnostics"]
        diagnostic_output = output / "validation_pseudo_diagnostics"
        if not (args.reuse_existing and (diagnostic_output / "diagnostic_summary.json").is_file()):
            command = [sys.executable, "-m", "scripts.diagnose_ctq_pseudo_labels", "--protocol-dir", str(protocol), "--data", str(training), "--split", str(diagnostics["split"]), "--method", str(config["teacher_method"]), "--output-dir", str(diagnostic_output), "--thresholds", str(diagnostics["thresholds"]), "--precision-target", str(diagnostics["precision_target"]), "--min-predictions", str(diagnostics["min_predictions"]), "--fallback", str(diagnostics["fallback"]), "--iou-match", str(diagnostics["iou_match"]), "--imgsz", str(diagnostics["imgsz"]), "--batch", str(diagnostics["batch"]), "--workers", str(diagnostics["workers"]), "--device", str(diagnostics["device"])]
            run(command)

    if args.phase in ("all", "create-test-v2"):
        if source_manifest is None:
            raise ValueError("create-test-v2 requires source_manifest in the development config.")
        split = config["test_v2"]
        command = [sys.executable, "-m", "scripts.create_ctq_source_disjoint_split", "--data", str(training), "--source-manifest", str(source_manifest), "--output-root", str(resolve(split["output_root"])), "--output-data", str(resolve(split["output_data"])), "--test-fraction", str(split["test_fraction"]), "--val-fraction", str(split["val_fraction"]), "--seed", str(split["split_seed"]), "--unlabeled-root", str(unlabeled), "--unlabeled-catalog", str(resolve(split["unlabeled_catalog"]))]
        run(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
