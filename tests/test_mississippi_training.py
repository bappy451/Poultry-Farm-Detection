from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

torch = pytest.importorskip("torch")

from poultry_geo.mississippi_adjacency import write_adjacency_index_csv
from poultry_geo.train_mississippifm import (
    MississippiFMTrainingConfig,
    load_mississippifm_config,
    train_mississippifm,
)


def _write_tile(path: Path, *, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 512), color=(value, value, value)).save(path)


def _prepare_dataset(root: Path) -> Path:
    _write_tile(root / "raster_a__row_0__col_0.jpg", value=30)
    _write_tile(root / "raster_a__row_0__col_256.jpg", value=50)
    _write_tile(root / "raster_b__row_0__col_0.jpg", value=70)
    _write_tile(root / "raster_b__row_0__col_256.jpg", value=90)
    adjacency_csv = root / "adjacency.csv"
    write_adjacency_index_csv(root, adjacency_csv, overlap_ratio=0.5, tile_size=512)
    return adjacency_csv


def test_load_mississippifm_config_reads_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "mississippi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "image_size": 256,
                "model_size": "vit_s",
                "epoch_size": 4,
            }
        ),
        encoding="utf-8",
    )

    config = load_mississippifm_config(config_path)

    assert config.epochs == 1
    assert config.batch_size == 2
    assert config.model_size == "vit_s"
    assert config.epoch_size == 4


def test_train_mississippifm_writes_checkpoints_and_metrics(tmp_path: Path) -> None:
    adjacency_csv = _prepare_dataset(tmp_path / "tiles")
    output_dir = tmp_path / "runs" / "mississippifm"
    config = MississippiFMTrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        image_size=256,
        model_size="vit_s",
        num_workers=0,
        epoch_size=4,
        output_dir=output_dir,
    )

    summary = train_mississippifm(
        config,
        data_root=adjacency_csv.parent,
        adjacency_csv=adjacency_csv,
    )

    assert summary.output_dir == str(output_dir)
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "metrics.json").exists()
    assert summary.final_total_loss is not None


def test_train_mississippifm_can_select_best_checkpoint_from_probe_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjacency_csv = _prepare_dataset(tmp_path / "tiles")
    output_dir = tmp_path / "runs" / "mississippifm_probe"
    probe_config = tmp_path / "probe_config.yaml"
    probe_config.write_text("tasks: []\nselection_metric: macro_f1\n", encoding="utf-8")
    config = MississippiFMTrainingConfig(
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        image_size=256,
        model_size="vit_s",
        num_workers=0,
        epoch_size=4,
        output_dir=output_dir,
        probe_config_path=probe_config,
    )

    import poultry_geo.train_mississippifm as mississippi_training

    call_count = {"count": 0}

    def fake_evaluate_mississippi_probes(*args, **kwargs):
        call_count["count"] += 1
        return {
            "aggregate_primary_score": 0.2 if call_count["count"] == 1 else 0.9,
            "tasks": {},
        }

    monkeypatch.setattr(
        mississippi_training,
        "evaluate_mississippi_probes",
        fake_evaluate_mississippi_probes,
    )

    train_mississippifm(
        config,
        data_root=adjacency_csv.parent,
        adjacency_csv=adjacency_csv,
    )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    assert best_checkpoint["epoch"] == 2
    assert best_checkpoint["best_probe_score"] == pytest.approx(0.9)


def test_train_mississippifm_uses_autocast_for_bf16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adjacency_csv = _prepare_dataset(tmp_path / "tiles")
    output_dir = tmp_path / "runs" / "mississippifm_bf16"
    config = MississippiFMTrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        image_size=256,
        model_size="vit_s",
        precision="bf16",
        num_workers=0,
        epoch_size=4,
        output_dir=output_dir,
    )

    import poultry_geo.train_mississippifm as mississippi_training

    calls = {"count": 0}

    class _CountingContext:
        def __enter__(self):
            calls["count"] += 1
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_autocast_context(*, device, precision):
        return _CountingContext()

    monkeypatch.setattr(mississippi_training, "_autocast_context", fake_autocast_context)

    train_mississippifm(
        config,
        data_root=adjacency_csv.parent,
        adjacency_csv=adjacency_csv,
    )

    assert calls["count"] > 0
