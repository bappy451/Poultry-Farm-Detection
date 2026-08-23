"""Run the validation-only CTQ reliability-and-balance screen.

This reuses the audited v3 runner's sequencing (baseline -> validation-only
calibration -> matched control -> candidate) while selecting the standalone
balanced reliability trainer for the candidate.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from scripts import run_ctq_reliable_v3_screen as base

    original_launch = base.launch
    original_load_config = base.load_config
    config_box: dict[str, object] = {}

    def load_config(path):
        config = original_load_config(path)
        candidate = config["candidate"]
        required = {"pseudo_background_weight", "pseudo_class_weight_min", "pseudo_class_weight_max"}
        if required - candidate.keys():
            raise ValueError(f"Balanced reliability candidate missing {sorted(required - candidate.keys())}")
        config_box["candidate"] = candidate
        return config

    def launch(command: list[str]) -> None:
        if "scripts.train_ctq_reliable_obb" in command:
            candidate = config_box["candidate"]
            position = command.index("scripts.train_ctq_reliable_obb")
            command = list(command)
            command[position] = "scripts.train_ctq_reliable_balanced_obb"
            command.extend([
                "--pseudo-background-weight", str(candidate["pseudo_background_weight"]),
                "--pseudo-class-weight-min", str(candidate["pseudo_class_weight_min"]),
                "--pseudo-class-weight-max", str(candidate["pseudo_class_weight_max"]),
            ])
        original_launch(command)

    base.load_config = load_config
    base.launch = launch
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *(sys.argv[1:] if argv is None else argv)]
        return base.main()
    finally:
        sys.argv = old_argv
        base.load_config = original_load_config
        base.launch = original_launch


if __name__ == "__main__":
    raise SystemExit(main())
