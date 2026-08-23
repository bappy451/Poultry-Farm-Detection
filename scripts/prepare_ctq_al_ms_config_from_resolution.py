"""Freeze the validation-selected AL baseline resolution into the AL/MS CTQ config."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def scaled_unlabeled_size(imgsz: int) -> int:
    return max(32, round(imgsz * 0.70 / 32) * 32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-screen-dir", required=True)
    parser.add_argument("--template", default="configs/ctq_al_ms_reliable_balanced_v1.yaml")
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()
    directory, template, output = (resolve(value) for value in (args.resolution_screen_dir, args.template, args.output_config))
    ranking_path, per_seed_path = directory / "validation_ranking.csv", directory / "per_seed_validation.csv"
    if not ranking_path.is_file() or not per_seed_path.is_file() or not template.is_file():
        raise FileNotFoundError("Resolution ranking, per-seed results, or template is missing.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selected config: {output}")
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))
    if not ranking or int(ranking[0].get("n_seeds", 0)) != 3:
        raise RuntimeError("Freeze a CTQ resolution only after the three-seed validation ranking is complete.")
    winner = ranking[0]
    with per_seed_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("candidate_id") == winner["candidate_id"]]
    settings = {(int(row["imgsz"]), int(row["batch"])) for row in rows}
    if len(rows) != 3 or len(settings) != 1:
        raise RuntimeError("Selected candidate must have exactly three consistent per-seed records.")
    imgsz, batch = settings.pop()
    config = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    config["baseline"]["imgsz"], config["baseline"]["batch"] = imgsz, batch
    config["adaptation"]["imgsz"], config["adaptation"]["batch"] = imgsz, batch
    config["adaptation"]["unlabeled_batch"] = batch
    config["adaptation"]["unlabeled_imgsz"] = scaled_unlabeled_size(imgsz)
    config["output_dir"] = f"runs/ctq_al_ms_v1/reliable_balanced_ctq_{winner['candidate_id']}"
    config["resolution_selection"] = {"source_ranking": str(ranking_path), "candidate_id": winner["candidate_id"], "imgsz": imgsz, "batch": batch, "mean_best_val_map50_95": float(winner["mean_best_val_map50_95"]), "n_seeds": 3, "selection_used_test": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Selected {winner['candidate_id']} (imgsz={imgsz}, batch={batch}); wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
