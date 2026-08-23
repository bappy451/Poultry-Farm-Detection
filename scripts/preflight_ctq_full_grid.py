"""Fail-fast preflight for the staged CTQ-OBB grid specification."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def validate(config: dict, *, require_tile_manifest: bool) -> dict[str, int]:
    for field in ("data", "unlabeled_root", "model", "stages", "screen_seeds", "confirmation_seeds"):
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    for required in (path(config["data"]), path(config["unlabeled_root"]), path(config["model"])):
        if not required.exists():
            raise FileNotFoundError(required)
    stages = config["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("stages must be a non-empty list")
    ids = set()
    candidate_count = 0
    for stage in stages:
        identifier = stage.get("id")
        candidates = stage.get("candidates")
        if not isinstance(identifier, str) or identifier in ids or not isinstance(candidates, list) or not candidates:
            raise ValueError("Each stage needs a unique id and non-empty candidates")
        ids.add(identifier)
        candidate_count += len(candidates)
        if stage.get("requires_tile_manifest") and require_tile_manifest:
            manifest = path(config["tile_manifest"])
            if not manifest.is_file():
                raise FileNotFoundError(f"Cross-tile stage requires a validated manifest: {manifest}")
            from scripts.ctq_tile_manifest import validate_cross_tile_manifest

            validate_cross_tile_manifest(manifest, root=path(config["unlabeled_root"]))
    return {"stages": len(stages), "candidates": candidate_count, "screen_runs": candidate_count * len(config["screen_seeds"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ctq_obb_full_staged_grid.yaml")
    parser.add_argument("--skip-cross-tile", action="store_true", help="Validate the first five stages only while the manifest is being built.")
    args = parser.parse_args()
    config_path = path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    result = validate(config, require_tile_manifest=not args.skip_cross_tile)
    print(f"CTQ grid preflight passed: {result['stages']} stages, {result['candidates']} candidates, {result['screen_runs']} screening runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
