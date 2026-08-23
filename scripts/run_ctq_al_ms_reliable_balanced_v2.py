"""Run AL/MS CTQ after resolution selection, reusing the identical baselines."""

from __future__ import annotations

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
    (destination / "reused_baseline.json").write_text(
        '{"reused_from": ' + repr(str(source.resolve())).replace("'", '"') + ', "reason": "identical AL validation baseline selected by resolution screen"}\n',
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
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
        source_ranking = Path(str(selection["source_ranking"])).expanduser().resolve()
        baseline_root = source_ranking.parent / str(selection["candidate_id"])
        candidate = config["candidate"]
        required = {"pseudo_background_weight", "pseudo_class_weight_min", "pseudo_class_weight_max"}
        if required - candidate.keys():
            raise ValueError(f"Balanced candidate missing {sorted(required - candidate.keys())}")
        box.update({"candidate": candidate, "baseline_root": baseline_root})
        return config

    def launch(command: list[str]) -> None:
        command = list(command)
        if "scripts.train_supervised_obb" in command:
            output = Path(command[command.index("--output-dir") + 1])
            seed = command[command.index("--seed") + 1]
            _copy_baseline(Path(box["baseline_root"]) / f"seed{seed}", output)
            print(f"$ reused resolution baseline seed {seed}: {output}", flush=True)
            return
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
        sys.argv = [old_argv[0], *(sys.argv[1:] if argv is None else argv)]
        return base.main()
    finally:
        sys.argv = old_argv
        base.load_config, base.launch = original_load, original_launch


if __name__ == "__main__":
    raise SystemExit(main())
