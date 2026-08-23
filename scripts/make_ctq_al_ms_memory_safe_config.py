"""Create a resource-safe CTQ config from a frozen supervised resolution choice."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--adaptation-batch", type=int, default=4)
    parser.add_argument("--unlabeled-batch", type=int, default=4)
    parser.add_argument("--unlabeled-imgsz", type=int, default=640)
    args = parser.parse_args()
    source, output = resolve(args.source_config), resolve(args.output_config)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite config: {output}")
    if args.adaptation_batch <= 0 or args.unlabeled_batch <= 0 or args.unlabeled_imgsz <= 0 or args.unlabeled_imgsz % 32:
        raise ValueError("Batches must be positive and unlabeled-imgsz divisible by 32.")
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    selection = config.get("resolution_selection")
    if not isinstance(selection, dict) or selection.get("selection_used_test") is not False:
        raise ValueError("Source config must contain a validation-only resolution selection.")
    adaptation = config.get("adaptation")
    if not isinstance(adaptation, dict) or int(adaptation.get("imgsz", 0)) != int(selection.get("imgsz", 0)):
        raise ValueError("Source config has inconsistent selected resolution.")
    adaptation["batch"] = args.adaptation_batch
    adaptation["unlabeled_batch"] = args.unlabeled_batch
    adaptation["unlabeled_imgsz"] = args.unlabeled_imgsz
    config["output_dir"] = str(config["output_dir"]).rstrip("/\\") + f"_ctqbatch{args.adaptation_batch}"
    config["resource_policy"] = {
        "reason": "CTQ holds both student and EMA teacher; inference calibration at selected 1024 resolution OOMed at batch 8.",
        "supervised_baseline_batch": int(config["baseline"]["batch"]),
        "ctq_labeled_batch": args.adaptation_batch,
        "ctq_unlabeled_batch": args.unlabeled_batch,
        "ctq_unlabeled_imgsz": args.unlabeled_imgsz,
        "calibration_batch": 1,
        "selection_used_test": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote memory-safe CTQ config: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
