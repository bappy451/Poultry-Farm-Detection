"""Run validation-only balanced CTQ with AL labels and MS unlabeled imagery.

The shared reliability runner normally requires an excluded-source catalog.
Here the user-declared AL/MS state separation is audited separately, so the
candidate streams the MS root directly and never reads the AL held-out test.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from scripts import run_ctq_reliable_v3_screen as base

    original_load, original_launch = base.load_config, base.launch
    config_box: dict[str, object] = {}

    def load_config(path):
        config = original_load(path)
        if config.get("unlabeled_catalog_mode") != "root_scan":
            raise ValueError("AL/MS runner requires unlabeled_catalog_mode: root_scan.")
        candidate = config["candidate"]
        required = {"pseudo_background_weight", "pseudo_class_weight_min", "pseudo_class_weight_max"}
        if required - candidate.keys():
            raise ValueError(f"Balanced candidate missing {sorted(required - candidate.keys())}")
        config_box["candidate"] = candidate
        return config

    def launch(command: list[str]) -> None:
        command = list(command)
        # The base runner validates a placeholder catalog. Remove it before the
        # training command so the native loader enumerates only the MS root.
        if "--unlabeled-catalog" in command:
            index = command.index("--unlabeled-catalog")
            del command[index:index + 2]
        if "scripts.train_ctq_reliable_obb" in command:
            candidate = config_box["candidate"]
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
        sys.argv = [old_argv[0], *(sys.argv[1:] if argv is None else argv)]
        return base.main()
    finally:
        sys.argv = old_argv
        base.load_config, base.launch = original_load, original_launch


if __name__ == "__main__":
    raise SystemExit(main())
