"""Memory-safe AL/MS CTQ screen: 1024 calibration runs at a separate batch size."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _copy_baseline(source: Path, destination: Path) -> None:
    required = (source / "results.csv", source / "weights" / "best.pt")
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Resolution baseline is incomplete: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "weights").mkdir(exist_ok=True)
    shutil.copy2(source / "results.csv", destination / "results.csv")
    shutil.copy2(source / "weights" / "best.pt", destination / "weights" / "best.pt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--calibration-batch", type=int, default=1)
    parsed, remaining = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if parsed.calibration_batch <= 0:
        raise ValueError("--calibration-batch must be positive.")
    from scripts import run_ctq_reliable_v3_screen as base

    original_load, original_launch = base.load_config, base.launch
    box: dict[str, object] = {}

    def load_config(path):
        config = original_load(path)
        if config.get("unlabeled_catalog_mode") != "root_scan":
            raise ValueError("AL/MS runner requires unlabeled_catalog_mode: root_scan.")
        selection = config.get("resolution_selection")
        if not isinstance(selection, dict) or selection.get("selection_used_test") is not False or int(selection.get("n_seeds", 0)) != 3:
            raise ValueError("Use a three-seed validation-selected config from prepare_ctq_al_ms_config_from_resolution.")
        candidate = config["candidate"]
        needed = {"pseudo_background_weight", "pseudo_class_weight_min", "pseudo_class_weight_max"}
        if needed - candidate.keys():
            raise ValueError(f"Balanced candidate missing {sorted(needed - candidate.keys())}")
        source_ranking = Path(str(selection["source_ranking"])).expanduser().resolve()
        box.update({"candidate": candidate, "baseline_root": source_ranking.parent / str(selection["candidate_id"])})
        return config

    def launch(command: list[str]) -> None:
        command = list(command)
        if "scripts.train_supervised_obb" in command:
            output = Path(command[command.index("--output-dir") + 1])
            seed = command[command.index("--seed") + 1]
            _copy_baseline(Path(box["baseline_root"]) / f"seed{seed}", output)
            print(f"$ reused resolution baseline seed {seed}: {output}", flush=True)
            return
        if "scripts.calibrate_pseudo_thresholds" in command:
            index = command.index("--batch")
            command[index + 1] = str(parsed.calibration_batch)
            print(f"$ calibration batch forced to {parsed.calibration_batch}", flush=True)
        if "--unlabeled-catalog" in command:
            index = command.index("--unlabeled-catalog")
            del command[index:index + 2]
        if "scripts.train_ctq_reliable_obb" in command:
            candidate = box["candidate"]
            index = command.index("scripts.train_ctq_reliable_obb")
            command[index] = "scripts.train_ctq_reliable_balanced_obb"
            command.extend([
                "--pseudo-background-weight", str(candidate["pseudo_background_weight"]),
                "--pseudo-class-weight-min", str(candidate["pseudo_class_weight_min"]),
                "--pseudo-class-weight-max", str(candidate["pseudo_class_weight_max"]),
            ])
        original_launch(command)

    base.load_config, base.launch = load_config, launch
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        return base.main()
    finally:
        sys.argv = old_argv
        base.load_config, base.launch = original_load, original_launch


if __name__ == "__main__":
    raise SystemExit(main())
