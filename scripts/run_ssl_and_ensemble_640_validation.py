"""Run the frozen 640 validation workflow in the required order.

The anchored-SSL validation protocol runs first.  If it succeeds, the
three-seed frozen ensemble evaluator runs second.  Both stages are
validation-only and inherit the same Python environment; a non-zero exit from
either stage stops the workflow immediately.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssl-config",
        default="configs/ssl_al_ms_anchored_dinov2_640_validation_v1.yaml",
        help="Anchored-SSL 640 validation config.",
    )
    parser.add_argument(
        "--ensemble-config",
        default="configs/full_al_640_seed_ensemble_validation_v1.yaml",
        help="Three-seed ensemble validation config.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed artifacts/checkpoints where each stage permits it.",
    )
    return parser.parse_args()


def run_stage(label: str, module: str, config: str, *, reuse_existing: bool) -> None:
    command = [sys.executable, "-m", module, "--config", config]
    if module == "scripts.run_ssl_al_ms_anchored_640_validation":
        command += ["--phase", "development"]
    if reuse_existing:
        command.append("--reuse-existing")
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    run_stage(
        "Anchored SSL 640 validation",
        "scripts.run_ssl_al_ms_anchored_640_validation",
        args.ssl_config,
        reuse_existing=args.reuse_existing,
    )
    run_stage(
        "Three-seed 640 ensemble validation",
        "scripts.evaluate_obb_seed_ensemble_validation",
        args.ensemble_config,
        reuse_existing=args.reuse_existing,
    )
    print("\n=== Workflow complete: both validation stages succeeded. ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
