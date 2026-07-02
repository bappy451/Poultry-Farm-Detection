from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from poultry_geo.evaluate_probes import (
    ProbeTaskConfig,
    evaluate_probe_task,
    load_frozen_mississippifm_encoder,
)


@dataclass(slots=True)
class BenchmarkTaskConfig:
    name: str
    task_type: str
    manifest: Path
    split_columns: tuple[str, ...] = ("county", "gsd", "year")
    image_column: str = "image_path"
    label_column: str = "label"
    mask_column: str = "mask_path"
    boxes_column: str = "boxes"
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    hidden_dim: int = 0
    threshold: float = 0.5
    max_detections_per_image: int = 32
    nms_iou_threshold: float = 0.5
    min_train_samples: int = 2
    min_val_samples: int = 1


@dataclass(slots=True)
class GeographicBenchmarkConfig:
    tasks: list[BenchmarkTaskConfig]


def load_geographic_benchmark_config(path: str | Path) -> GeographicBenchmarkConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Geographic benchmark config must contain a top-level mapping.")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Geographic benchmark config must define a non-empty 'tasks' list.")
    return GeographicBenchmarkConfig(tasks=[_benchmark_task_from_mapping(task) for task in raw_tasks])


def run_geographic_benchmark(
    config: GeographicBenchmarkConfig | str | Path,
    *,
    output_dir: str | Path,
    encoder=None,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    resolved_config = load_geographic_benchmark_config(config) if isinstance(config, (str, Path)) else config
    if encoder is None and checkpoint_path is None:
        raise ValueError("Provide either encoder or checkpoint_path for benchmark evaluation.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "geographic_benchmark.json"
    if resume and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    frozen_encoder = encoder or load_frozen_mississippifm_encoder(checkpoint_path, device=device)
    task_summaries: dict[str, Any] = {}
    for task in resolved_config.tasks:
        frame = pd.read_csv(task.manifest)
        task_summaries[task.name] = _run_single_benchmark_task(
            task,
            frame=frame,
            encoder=frozen_encoder,
            device=device,
        )

    summary = {
        "output_dir": str(output_path),
        "tasks": task_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_zero_shot_transfer_evaluation(
    task: BenchmarkTaskConfig,
    *,
    source_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    encoder,
    device: str | None = None,
    few_shot_per_group: int | None = None,
    few_shot_group_column: str | None = None,
    seed: int = 27,
) -> dict[str, Any]:
    probe_task = _benchmark_task_to_probe_task(task)
    zero_shot = evaluate_probe_task(
        probe_task,
        encoder=encoder,
        device=device,
        train_frame=source_frame,
        val_frame=target_frame,
    )

    summary: dict[str, Any] = {"zero_shot": zero_shot}
    if few_shot_per_group is not None and few_shot_per_group > 0:
        adaptation_frame = sample_few_shot_subset(
            target_frame,
            group_column=few_shot_group_column,
            shots_per_group=few_shot_per_group,
            seed=seed,
        )
        evaluation_target = target_frame.drop(index=adaptation_frame.index)
        if evaluation_target.empty:
            evaluation_target = adaptation_frame.copy()
        few_shot_train = pd.concat([source_frame, adaptation_frame], ignore_index=True)
        few_shot_metrics = evaluate_probe_task(
            probe_task,
            encoder=encoder,
            device=device,
            train_frame=few_shot_train,
            val_frame=evaluation_target,
        )
        few_shot_metrics["few_shot_samples"] = int(len(adaptation_frame))
        summary["few_shot"] = few_shot_metrics
    return summary


def sample_few_shot_subset(
    frame: pd.DataFrame,
    *,
    group_column: str | None,
    shots_per_group: int,
    seed: int = 27,
) -> pd.DataFrame:
    if shots_per_group <= 0:
        raise ValueError("shots_per_group must be positive.")
    if frame.empty:
        return frame.copy()
    if group_column is None:
        return frame.sample(n=min(shots_per_group, len(frame)), random_state=seed).sort_index()
    if group_column not in frame.columns:
        raise ValueError(f"group_column '{group_column}' is not present in the frame.")
    groups = []
    for _, group_frame in frame.groupby(group_column, sort=True):
        groups.append(group_frame.sample(n=min(shots_per_group, len(group_frame)), random_state=seed))
    return pd.concat(groups, axis=0).sort_index()


def _run_single_benchmark_task(
    task: BenchmarkTaskConfig,
    *,
    frame: pd.DataFrame,
    encoder,
    device: str | None,
) -> dict[str, Any]:
    split_summaries: dict[str, Any] = {}
    for split_column in task.split_columns:
        if split_column not in frame.columns:
            continue
        splits: list[dict[str, Any]] = []
        for holdout_value in sorted(frame[split_column].dropna().unique().tolist(), key=str):
            train_frame = frame.loc[frame[split_column] != holdout_value].copy()
            val_frame = frame.loc[frame[split_column] == holdout_value].copy()
            if len(train_frame) < task.min_train_samples or len(val_frame) < task.min_val_samples:
                continue
            probe_task = _benchmark_task_to_probe_task(
                task,
                name=f"{task.name}:{split_column}:{holdout_value}",
            )
            metrics = evaluate_probe_task(
                probe_task,
                encoder=encoder,
                device=device,
                train_frame=train_frame,
                val_frame=val_frame,
            )
            splits.append({"holdout": _json_safe_value(holdout_value), **metrics})
        split_summaries[split_column] = _aggregate_split_metrics(splits)
    return split_summaries


def _aggregate_split_metrics(splits: list[dict[str, Any]]) -> dict[str, Any]:
    if not splits:
        return {"num_splits": 0, "splits": []}
    numeric_keys = sorted(
        key
        for key in splits[0]
        if key not in {"holdout", "task_type", "primary_metric_name"}
        and isinstance(splits[0][key], (int, float))
    )
    summary: dict[str, Any] = {
        "num_splits": len(splits),
        "splits": splits,
    }
    for key in numeric_keys:
        values = [float(split[key]) for split in splits]
        summary[f"mean_{key}"] = float(sum(values) / len(values))
        variance = sum((value - summary[f"mean_{key}"]) ** 2 for value in values) / len(values)
        summary[f"std_{key}"] = float(math.sqrt(variance))
    return summary


def _benchmark_task_from_mapping(payload: Any) -> BenchmarkTaskConfig:
    if not isinstance(payload, dict):
        raise ValueError("Each geographic benchmark task must be a mapping.")
    split_columns = payload.get("split_columns", ("county", "gsd", "year"))
    if isinstance(split_columns, str):
        split_columns = [split_columns]
    return BenchmarkTaskConfig(
        name=str(payload["name"]),
        task_type=str(payload["task_type"]).strip().lower(),
        manifest=Path(payload["manifest"]).expanduser().resolve(),
        split_columns=tuple(str(column) for column in split_columns),
        image_column=str(payload.get("image_column", "image_path")),
        label_column=str(payload.get("label_column", "label")),
        mask_column=str(payload.get("mask_column", "mask_path")),
        boxes_column=str(payload.get("boxes_column", "boxes")),
        epochs=int(payload.get("epochs", 5)),
        batch_size=int(payload.get("batch_size", 4)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        hidden_dim=int(payload.get("hidden_dim", 0)),
        threshold=float(payload.get("threshold", 0.5)),
        max_detections_per_image=int(payload.get("max_detections_per_image", 32)),
        nms_iou_threshold=float(payload.get("nms_iou_threshold", 0.5)),
        min_train_samples=int(payload.get("min_train_samples", 2)),
        min_val_samples=int(payload.get("min_val_samples", 1)),
    )


def _benchmark_task_to_probe_task(
    task: BenchmarkTaskConfig,
    *,
    name: str | None = None,
) -> ProbeTaskConfig:
    return ProbeTaskConfig(
        name=name or task.name,
        task_type=task.task_type,
        train_manifest=task.manifest,
        val_manifest=task.manifest,
        image_column=task.image_column,
        label_column=task.label_column,
        mask_column=task.mask_column,
        boxes_column=task.boxes_column,
        epochs=task.epochs,
        batch_size=task.batch_size,
        learning_rate=task.learning_rate,
        hidden_dim=task.hidden_dim,
        threshold=task.threshold,
        max_detections_per_image=task.max_detections_per_image,
        nms_iou_threshold=task.nms_iou_threshold,
    )


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
