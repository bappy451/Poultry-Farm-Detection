from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")

from poultry_geo.evaluate_probes import (
    MississippiProbeConfig,
    ProbeTaskConfig,
    evaluate_mississippi_probes,
    load_probe_config,
)
from poultry_geo.geographic_benchmark import (
    BenchmarkTaskConfig,
    GeographicBenchmarkConfig,
    load_geographic_benchmark_config,
    run_zero_shot_transfer_evaluation,
    run_geographic_benchmark,
    sample_few_shot_subset,
)


class FakeFrozenEncoder(torch.nn.Module):
    def __init__(self, *, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.image_size = 32
        self.embed_dim = 8

    def forward_views(
        self,
        *,
        global_views: torch.Tensor,
        local_views: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled = global_views.mean(dim=(2, 3))
        mean_rgb = pooled.mean(dim=1, keepdim=True)
        max_rgb = global_views.amax(dim=(2, 3))
        global_cls = torch.cat([pooled, mean_rgb, max_rgb], dim=1)
        if global_cls.shape[1] < self.embed_dim:
            padding = torch.zeros(
                global_cls.shape[0],
                self.embed_dim - global_cls.shape[1],
                device=global_cls.device,
                dtype=global_cls.dtype,
            )
            global_cls = torch.cat([global_cls, padding], dim=1)

        intensity = global_views.mean(dim=1, keepdim=True)
        pooled_grid = torch.nn.functional.avg_pool2d(
            intensity,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        flat_grid = pooled_grid.flatten(2).transpose(1, 2)
        patch_features = torch.cat(
            [
                flat_grid,
                flat_grid**2,
                1.0 - flat_grid,
                flat_grid * 0.5,
                torch.ones_like(flat_grid) * 0.25,
                torch.zeros_like(flat_grid),
                torch.ones_like(flat_grid) * 0.75,
                flat_grid * 0.25,
            ],
            dim=-1,
        )
        return {
            "global_cls": global_cls[:, : self.embed_dim],
            "global_patches": patch_features[:, :, : self.embed_dim],
        }


def _write_rgb_image(path: Path, *, bright: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = 220 if bright else 25
    Image.new("RGB", (32, 32), color=(value, value, value)).save(path)


def _write_segmentation_example(image_path: Path, mask_path: Path, *, offset: int) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=(15, 15, 15))
    draw = ImageDraw.Draw(image)
    draw.rectangle((offset, offset, offset + 7, offset + 7), fill=(230, 230, 230))
    image.save(image_path)

    mask = Image.new("L", (32, 32), color=0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((offset, offset, offset + 7, offset + 7), fill=255)
    mask.save(mask_path)


def _write_detection_example(image_path: Path, *, boxes: list[list[int]]) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=(15, 15, 15))
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=(230, 230, 230))
    image.save(image_path)


def _write_classification_manifests(root: Path) -> tuple[Path, Path]:
    train_rows = []
    val_rows = []
    for index, bright in enumerate([False, False, True, True]):
        image_path = root / "cls_train" / f"example_{index}.png"
        _write_rgb_image(image_path, bright=bright)
        train_rows.append({"image_path": str(image_path), "label": int(bright)})
    for index, bright in enumerate([False, True]):
        image_path = root / "cls_val" / f"example_{index}.png"
        _write_rgb_image(image_path, bright=bright)
        val_rows.append({"image_path": str(image_path), "label": int(bright)})

    train_manifest = root / "classification_train.csv"
    val_manifest = root / "classification_val.csv"
    pd.DataFrame(train_rows).to_csv(train_manifest, index=False)
    pd.DataFrame(val_rows).to_csv(val_manifest, index=False)
    return train_manifest, val_manifest


def _write_context_manifests(root: Path) -> tuple[Path, Path]:
    train_rows = []
    val_rows = []
    for index, bright in enumerate([False, False, True, True]):
        image_path = root / "ctx_train" / f"image_{index}.png"
        context_path = root / "ctx_train_context" / f"context_{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)
        _write_rgb_image(context_path, bright=bright)
        train_rows.append(
            {
                "image_path": str(image_path),
                "context_image_path": str(context_path),
                "label": int(bright),
            }
        )
    for index, bright in enumerate([False, True]):
        image_path = root / "ctx_val" / f"image_{index}.png"
        context_path = root / "ctx_val_context" / f"context_{index}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(120, 120, 120)).save(image_path)
        _write_rgb_image(context_path, bright=bright)
        val_rows.append(
            {
                "image_path": str(image_path),
                "context_image_path": str(context_path),
                "label": int(bright),
            }
        )

    train_manifest = root / "context_train.csv"
    val_manifest = root / "context_val.csv"
    pd.DataFrame(train_rows).to_csv(train_manifest, index=False)
    pd.DataFrame(val_rows).to_csv(val_manifest, index=False)
    return train_manifest, val_manifest


def _write_segmentation_manifests(root: Path) -> tuple[Path, Path]:
    train_rows = []
    val_rows = []
    for index, offset in enumerate([0, 8, 16, 24]):
        image_path = root / "seg_train" / f"image_{index}.png"
        mask_path = root / "seg_train_masks" / f"mask_{index}.png"
        _write_segmentation_example(image_path, mask_path, offset=offset)
        train_rows.append({"image_path": str(image_path), "mask_path": str(mask_path)})
    for index, offset in enumerate([8, 16]):
        image_path = root / "seg_val" / f"image_{index}.png"
        mask_path = root / "seg_val_masks" / f"mask_{index}.png"
        _write_segmentation_example(image_path, mask_path, offset=offset)
        val_rows.append({"image_path": str(image_path), "mask_path": str(mask_path)})

    train_manifest = root / "segmentation_train.csv"
    val_manifest = root / "segmentation_val.csv"
    pd.DataFrame(train_rows).to_csv(train_manifest, index=False)
    pd.DataFrame(val_rows).to_csv(val_manifest, index=False)
    return train_manifest, val_manifest


def _write_benchmark_manifest(root: Path) -> Path:
    rows = []
    examples = [
        ("al", "county_a", 0.6, 2021, False),
        ("al", "county_b", 0.6, 2022, True),
        ("ga", "county_c", 0.3, 2021, False),
        ("ga", "county_d", 0.3, 2022, True),
        ("ms", "county_e", 0.6, 2023, False),
        ("ms", "county_f", 0.3, 2023, True),
    ]
    for index, (state, county, gsd, year, bright) in enumerate(examples):
        image_path = root / "benchmark" / f"image_{index}.png"
        _write_rgb_image(image_path, bright=bright)
        rows.append(
            {
                "image_path": str(image_path),
                "label": int(bright),
                "state": state,
                "county": county,
                "gsd": gsd,
                "year": year,
            }
        )
    manifest = root / "benchmark_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def _write_detection_manifests(root: Path) -> tuple[Path, Path]:
    train_rows = []
    val_rows = []
    train_examples = [
        ([[0, 0, 8, 8]], "train_0.png"),
        ([[8, 8, 16, 16]], "train_1.png"),
        ([[16, 16, 24, 24]], "train_2.png"),
        ([[24, 24, 32, 32]], "train_3.png"),
    ]
    val_examples = [
        ([[8, 8, 16, 16]], "val_0.png"),
        ([[16, 16, 24, 24]], "val_1.png"),
    ]
    for boxes, name in train_examples:
        image_path = root / "det_train" / name
        _write_detection_example(image_path, boxes=boxes)
        train_rows.append({"image_path": str(image_path), "boxes": json.dumps(boxes)})
    for boxes, name in val_examples:
        image_path = root / "det_val" / name
        _write_detection_example(image_path, boxes=boxes)
        val_rows.append({"image_path": str(image_path), "boxes": json.dumps(boxes)})

    train_manifest = root / "detection_train.csv"
    val_manifest = root / "detection_val.csv"
    pd.DataFrame(train_rows).to_csv(train_manifest, index=False)
    pd.DataFrame(val_rows).to_csv(val_manifest, index=False)
    return train_manifest, val_manifest


def test_load_probe_config_reads_yaml_tasks(tmp_path: Path) -> None:
    cls_train, cls_val = _write_classification_manifests(tmp_path)
    config_path = tmp_path / "probe_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "selection_metric": "macro_f1",
                "tasks": [
                    {
                        "name": "settlement_context",
                        "task_type": "classification",
                        "train_manifest": str(cls_train),
                        "val_manifest": str(cls_val),
                        "epochs": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_probe_config(config_path)

    assert config.selection_metric == "macro_f1"
    assert len(config.tasks) == 1
    assert config.tasks[0].name == "settlement_context"
    assert config.tasks[0].task_type == "classification"


def test_evaluate_mississippi_probes_writes_metrics_for_classification_and_segmentation(tmp_path: Path) -> None:
    cls_train, cls_val = _write_classification_manifests(tmp_path)
    seg_train, seg_val = _write_segmentation_manifests(tmp_path)
    output_dir = tmp_path / "probe_results"
    config = MississippiProbeConfig(
        selection_metric="macro_f1",
        tasks=[
            ProbeTaskConfig(
                name="settlement_context",
                task_type="classification",
                train_manifest=cls_train,
                val_manifest=cls_val,
                epochs=6,
                batch_size=2,
                learning_rate=0.2,
            ),
            ProbeTaskConfig(
                name="poultry_segmentation",
                task_type="segmentation",
                train_manifest=seg_train,
                val_manifest=seg_val,
                epochs=12,
                batch_size=2,
                learning_rate=0.2,
            ),
        ],
    )

    summary = evaluate_mississippi_probes(
        config,
        output_dir=output_dir,
        encoder=FakeFrozenEncoder(),
        device="cpu",
    )

    assert (output_dir / "probe_metrics.json").exists()
    assert "settlement_context" in summary["tasks"]
    assert "poultry_segmentation" in summary["tasks"]
    assert summary["tasks"]["settlement_context"]["macro_f1"] >= 0.99
    assert summary["tasks"]["poultry_segmentation"]["dice"] >= 0.80
    assert summary["aggregate_primary_score"] >= 0.80


def test_evaluate_mississippi_probes_writes_ap_metrics_for_detection(tmp_path: Path) -> None:
    det_train, det_val = _write_detection_manifests(tmp_path)
    output_dir = tmp_path / "probe_detection_results"
    config = MississippiProbeConfig(
        selection_metric="ap50_95",
        tasks=[
            ProbeTaskConfig(
                name="oriented_object_detection",
                task_type="detection",
                train_manifest=det_train,
                val_manifest=det_val,
                boxes_column="boxes",
                epochs=30,
                batch_size=2,
                learning_rate=0.2,
                threshold=0.5,
            )
        ],
    )

    summary = evaluate_mississippi_probes(
        config,
        output_dir=output_dir,
        encoder=FakeFrozenEncoder(),
        device="cpu",
    )

    task_summary = summary["tasks"]["oriented_object_detection"]
    assert task_summary["ap50"] >= 0.90
    assert task_summary["ap50_95"] >= 0.75


def test_evaluate_mississippi_probes_supports_texture_retrieval_metrics(tmp_path: Path) -> None:
    cls_train, cls_val = _write_classification_manifests(tmp_path)
    output_dir = tmp_path / "probe_retrieval_results"
    config = MississippiProbeConfig(
        selection_metric="recall_at_1",
        tasks=[
            ProbeTaskConfig(
                name="texture_retrieval",
                task_type="retrieval",
                train_manifest=cls_train,
                val_manifest=cls_val,
                epochs=1,
                batch_size=2,
            )
        ],
    )

    summary = evaluate_mississippi_probes(
        config,
        output_dir=output_dir,
        encoder=FakeFrozenEncoder(),
        device="cpu",
    )

    task_summary = summary["tasks"]["texture_retrieval"]
    assert task_summary["recall_at_1"] >= 0.99
    assert task_summary["recall_at_5"] >= 0.99


def test_evaluate_mississippi_probes_supports_context_augmentation_metrics(tmp_path: Path) -> None:
    ctx_train, ctx_val = _write_context_manifests(tmp_path)
    output_dir = tmp_path / "probe_context_results"
    config = MississippiProbeConfig(
        selection_metric="context_macro_f1",
        tasks=[
            ProbeTaskConfig(
                name="context_augmentation",
                task_type="context_augmentation",
                train_manifest=ctx_train,
                val_manifest=ctx_val,
                context_image_column="context_image_path",
                epochs=8,
                batch_size=2,
                learning_rate=0.2,
            )
        ],
    )

    summary = evaluate_mississippi_probes(
        config,
        output_dir=output_dir,
        encoder=FakeFrozenEncoder(),
        device="cpu",
    )

    task_summary = summary["tasks"]["context_augmentation"]
    assert task_summary["base_macro_f1"] <= 0.51
    assert task_summary["context_macro_f1"] >= 0.99
    assert task_summary["context_gain"] >= 0.40


def test_load_geographic_benchmark_config_reads_yaml(tmp_path: Path) -> None:
    manifest = _write_benchmark_manifest(tmp_path)
    config_path = tmp_path / "benchmark_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {
                        "name": "county_transfer",
                        "task_type": "classification",
                        "manifest": str(manifest),
                        "split_columns": ["county", "gsd", "year"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    config = load_geographic_benchmark_config(config_path)

    assert len(config.tasks) == 1
    assert config.tasks[0].split_columns == ("county", "gsd", "year")


def test_run_geographic_benchmark_writes_aggregate_split_metrics(tmp_path: Path) -> None:
    manifest = _write_benchmark_manifest(tmp_path)
    output_dir = tmp_path / "benchmark_results"
    config = GeographicBenchmarkConfig(
        tasks=[
            BenchmarkTaskConfig(
                name="state_transfer",
                task_type="classification",
                manifest=manifest,
                split_columns=("county", "gsd", "year"),
                epochs=6,
                batch_size=2,
                learning_rate=0.2,
            )
        ]
    )

    summary = run_geographic_benchmark(
        config,
        output_dir=output_dir,
        encoder=FakeFrozenEncoder(),
        device="cpu",
    )

    assert (output_dir / "geographic_benchmark.json").exists()
    task_summary = summary["tasks"]["state_transfer"]
    assert task_summary["county"]["num_splits"] >= 2
    assert task_summary["gsd"]["num_splits"] >= 2
    assert task_summary["year"]["num_splits"] >= 2


def test_zero_shot_transfer_utilities_support_few_shot_adaptation(tmp_path: Path) -> None:
    manifest = _write_benchmark_manifest(tmp_path)
    frame = pd.read_csv(manifest)
    source_frame = frame.loc[frame["state"] == "al"].copy()
    target_frame = frame.loc[frame["state"] != "al"].copy()
    benchmark_task = BenchmarkTaskConfig(
        name="state_transfer",
        task_type="classification",
        manifest=manifest,
        split_columns=("state",),
        epochs=6,
        batch_size=2,
        learning_rate=0.2,
    )

    few_shot = sample_few_shot_subset(
        target_frame,
        group_column="state",
        shots_per_group=1,
        seed=27,
    )
    summary = run_zero_shot_transfer_evaluation(
        benchmark_task,
        source_frame=source_frame,
        target_frame=target_frame,
        encoder=FakeFrozenEncoder(),
        device="cpu",
        few_shot_per_group=1,
        few_shot_group_column="state",
        seed=27,
    )

    assert len(few_shot) == 2
    assert "zero_shot" in summary
    assert "few_shot" in summary
    assert summary["few_shot"]["few_shot_samples"] == 2
