"""Freeze the validation-selected supervised resolution into a CTQ screen config."""

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
    """Keep the established 640->448 ratio, aligned to the YOLO stride."""
    return max(32, round((imgsz * 0.70) / 32) * 32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-screen-dir", required=True)
    parser.add_argument("--template", default="configs/ctq_reliable_balanced_v5_source_v2_ready.yaml")
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()
    ranking_path, template_path, output = (resolve(value) for value in (args.resolution_screen_dir, args.template, args.output_config))
    ranking_path = ranking_path / "validation_ranking.csv"
    if not ranking_path.is_file() or not template_path.is_file():
        raise FileNotFoundError("Resolution ranking or CTQ template config is missing.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite selected CTQ config: {output}")
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        ranking = list(csv.DictReader(handle))
    if not ranking:
        raise RuntimeError("Resolution ranking is empty.")
    selected = ranking[0]
    for field in ("candidate_id", "imgsz", "batch", "n_seeds"):
        if field not in selected:
            raise ValueError(f"Resolution ranking lacks {field!r}.")
    if int(selected["n_seeds"]) < 3:
        raise RuntimeError("Do not freeze a CTQ image size before all three resolution seeds complete.")
    config = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    imgsz, batch = int(selected["imgsz"]), int(selected["batch"])
    if imgsz % 32 or batch <= 0:
        raise ValueError("Selected resolution is invalid.")
    config["baseline"]["imgsz"] = imgsz
    config["baseline"]["batch"] = batch
    config["adaptation"]["imgsz"] = imgsz
    config["adaptation"]["batch"] = batch
    config["adaptation"]["unlabeled_batch"] = batch
    config["adaptation"]["unlabeled_imgsz"] = scaled_unlabeled_size(imgsz)
    config["output_dir"] = f"runs/ctq_reliable_balanced_v5_{selected['candidate_id']}_run1"
    config["resolution_selection"] = {"source_ranking": str(ranking_path), "candidate_id": selected["candidate_id"], "imgsz": imgsz, "batch": batch, "mean_best_val_map50_95": float(selected["mean_best_val_map50_95"]), "n_seeds": int(selected["n_seeds"]), "selection_used_test": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Selected {selected['candidate_id']} (imgsz={imgsz}, batch={batch}); wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
