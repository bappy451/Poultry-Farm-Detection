from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from shapely.geometry import Polygon, box as shapely_box
from torch import nn

from poultry_geo.mississippi_models import MississippiFMEncoder


_VALID_TASK_TYPES = frozenset({"classification", "regression", "segmentation", "detection", "retrieval", "context_augmentation"})
_DEFAULT_PRIMARY_METRICS = {
    "classification": "macro_f1",
    "regression": "neg_rmse",
    "segmentation": "dice",
    "detection": "ap50_95",
    "retrieval": "recall_at_1",
    "context_augmentation": "context_macro_f1",
}


@dataclass(slots=True)
class ProbeTaskConfig:
    name: str
    task_type: str
    train_manifest: Path
    val_manifest: Path
    image_column: str = "image_path"
    context_image_column: str = "context_image_path"
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


@dataclass(slots=True)
class MississippiProbeConfig:
    tasks: list[ProbeTaskConfig]
    selection_metric: str = "mean_primary_metric"


def load_probe_config(path: str | Path) -> MississippiProbeConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Probe config must contain a top-level mapping.")
    tasks_payload = payload.get("tasks")
    if not isinstance(tasks_payload, list) or not tasks_payload:
        raise ValueError("Probe config must define a non-empty 'tasks' list.")
    tasks = [_probe_task_from_mapping(item) for item in tasks_payload]
    return MississippiProbeConfig(
        tasks=tasks,
        selection_metric=str(payload.get("selection_metric", "mean_primary_metric")),
    )


def load_frozen_mississippifm_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> MississippiFMEncoder:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    raw_config = checkpoint.get("config", {})
    if not isinstance(raw_config, dict):
        raw_config = {}
    patch_size = int(raw_config.get("patch_size", 16))
    model = MississippiFMEncoder(
        image_size=int(raw_config.get("image_size", 256)),
        patch_size=patch_size,
        model_size=str(raw_config.get("model_size", "vit_b")),
        precision=str(raw_config.get("precision", "fp32")),
        projection_dim=int(raw_config.get("projection_dim", max(64, patch_size * 4))),
        backbone_impl=str(raw_config.get("backbone_impl", "custom")),
        backbone_name=raw_config.get("backbone_name"),
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    resolved_device = torch.device(device or "cpu")
    model = model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    return model


def evaluate_mississippi_probes(
    config: MississippiProbeConfig | str | Path,
    *,
    output_dir: str | Path,
    encoder: MississippiFMEncoder | nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    resolved_config = load_probe_config(config) if isinstance(config, (str, Path)) else config
    if encoder is None and checkpoint_path is None:
        raise ValueError("Provide either encoder or checkpoint_path for probe evaluation.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path / "probe_metrics.json"
    if resume and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    loaded_encoder = encoder is None
    frozen_encoder = encoder or load_frozen_mississippifm_encoder(checkpoint_path, device=device)
    torch_device = torch.device(device or "cpu")
    frozen_encoder = frozen_encoder.to(torch_device)
    frozen_encoder.eval()
    if loaded_encoder:
        frozen_encoder.requires_grad_(False)

    task_summaries: dict[str, dict[str, Any]] = {}
    selection_scores: list[float] = []
    for task in resolved_config.tasks:
        metrics = evaluate_probe_task(
            task,
            encoder=frozen_encoder,
            device=torch_device,
        )
        task_summaries[task.name] = metrics
        metric_name, metric_value = _resolve_selection_metric(metrics, requested_metric=resolved_config.selection_metric)
        task_summaries[task.name]["selection_metric_name"] = metric_name
        task_summaries[task.name]["selection_metric"] = metric_value
        selection_scores.append(metric_value)

    summary = {
        "output_dir": str(output_path),
        "selection_metric": resolved_config.selection_metric,
        "aggregate_primary_score": float(sum(selection_scores) / max(1, len(selection_scores))),
        "tasks": task_summaries,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_probe_task(
    task: ProbeTaskConfig,
    *,
    encoder: nn.Module,
    device: str | torch.device | None = None,
    train_frame: pd.DataFrame | None = None,
    val_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    normalized_type = task.task_type.strip().lower()
    if normalized_type not in _VALID_TASK_TYPES:
        raise ValueError(f"Unsupported task_type '{task.task_type}'. Expected one of {sorted(_VALID_TASK_TYPES)}.")

    train_data = train_frame if train_frame is not None else pd.read_csv(task.train_manifest)
    val_data = val_frame if val_frame is not None else pd.read_csv(task.val_manifest)
    if train_data.empty or val_data.empty:
        raise ValueError(f"Task '{task.name}' requires non-empty train and validation manifests.")

    resolved_device = torch.device(device or "cpu")
    if normalized_type == "classification":
        return _evaluate_classification_task(task, train_data, val_data, encoder=encoder, device=resolved_device)
    if normalized_type == "regression":
        return _evaluate_regression_task(task, train_data, val_data, encoder=encoder, device=resolved_device)
    if normalized_type == "detection":
        return _evaluate_detection_task(task, train_data, val_data, encoder=encoder, device=resolved_device)
    if normalized_type == "retrieval":
        return _evaluate_retrieval_task(task, train_data, val_data, encoder=encoder, device=resolved_device)
    if normalized_type == "context_augmentation":
        return _evaluate_context_augmentation_task(task, train_data, val_data, encoder=encoder, device=resolved_device)
    return _evaluate_segmentation_task(task, train_data, val_data, encoder=encoder, device=resolved_device)


def _evaluate_classification_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    train_features = _extract_global_features(
        encoder,
        train_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    val_features = _extract_global_features(
        encoder,
        val_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )

    label_values = list(train_frame[task.label_column].tolist()) + list(val_frame[task.label_column].tolist())
    label_to_index = {value: index for index, value in enumerate(sorted(set(label_values)))}
    train_labels = torch.tensor([label_to_index[value] for value in train_frame[task.label_column]], dtype=torch.long, device=device)
    val_labels = torch.tensor([label_to_index[value] for value in val_frame[task.label_column]], dtype=torch.long, device=device)

    scored = _fit_and_score_classifier(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        learning_rate=task.learning_rate,
        epochs=task.epochs,
        hidden_dim=task.hidden_dim,
        device=device,
    )
    metrics = {
        "task_type": "classification",
        "accuracy": scored["accuracy"],
        "macro_f1": scored["macro_f1"],
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["classification"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _evaluate_regression_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    train_features = _extract_global_features(
        encoder,
        train_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    val_features = _extract_global_features(
        encoder,
        val_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    train_targets = torch.tensor(train_frame[task.label_column].astype(float).to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)
    val_targets = torch.tensor(val_frame[task.label_column].astype(float).to_numpy(), dtype=torch.float32, device=device).unsqueeze(1)

    head = _build_mlp_head(train_features.shape[1], 1, hidden_dim=task.hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=task.learning_rate)
    for _ in range(int(task.epochs)):
        optimizer.zero_grad(set_to_none=True)
        predictions = head(train_features)
        loss = F.mse_loss(predictions, train_targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_predictions = head(val_features)
        absolute_error = torch.abs(val_predictions - val_targets)
        squared_error = (val_predictions - val_targets) ** 2

    rmse = float(torch.sqrt(squared_error.mean()).cpu())
    metrics = {
        "task_type": "regression",
        "mae": float(absolute_error.mean().cpu()),
        "rmse": rmse,
        "neg_rmse": float(-rmse),
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["regression"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _evaluate_segmentation_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    train_patches, train_mask, grid_size, mask_size = _extract_patch_features_and_masks(
        encoder,
        train_frame,
        image_column=task.image_column,
        mask_column=task.mask_column,
        batch_size=task.batch_size,
        device=device,
    )
    val_patches, val_mask, _, _ = _extract_patch_features_and_masks(
        encoder,
        val_frame,
        image_column=task.image_column,
        mask_column=task.mask_column,
        batch_size=task.batch_size,
        device=device,
    )

    head = _SegmentationProbeHead(
        embed_dim=int(train_patches.shape[-1]),
        hidden_dim=int(task.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=task.learning_rate)
    positive = float(train_mask.sum().item())
    negative = float(train_mask.numel() - train_mask.sum().item())
    pos_weight = None
    if positive > 0.0:
        pos_weight = torch.tensor(max(1.0, negative / positive), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(int(task.epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_patches, grid_size=grid_size, output_size=mask_size)
        loss = loss_fn(logits, train_mask)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_logits = head(val_patches, grid_size=grid_size, output_size=mask_size)
        val_probs = torch.sigmoid(val_logits)
        predicted = (val_probs >= float(task.threshold)).float()

    dice = _dice_score(predicted, val_mask)
    miou = _mean_iou(predicted, val_mask)
    boundary_f1 = _boundary_f1(predicted, val_mask)
    metrics = {
        "task_type": "segmentation",
        "dice": dice,
        "miou": miou,
        "boundary_f1": boundary_f1,
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["segmentation"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _evaluate_detection_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    (
        train_patches,
        train_objectness,
        train_boxes,
        _train_gt_boxes,
        grid_size,
        image_size,
    ) = _extract_patch_features_and_boxes(
        encoder,
        train_frame,
        image_column=task.image_column,
        boxes_column=task.boxes_column,
        batch_size=task.batch_size,
        device=device,
    )
    train_patches = _augment_patch_features_with_positions(train_patches, grid_size=grid_size)
    (
        val_patches,
        _val_objectness,
        _val_boxes,
        val_gt_boxes,
        _,
        _,
    ) = _extract_patch_features_and_boxes(
        encoder,
        val_frame,
        image_column=task.image_column,
        boxes_column=task.boxes_column,
        batch_size=task.batch_size,
        device=device,
    )
    val_patches = _augment_patch_features_with_positions(val_patches, grid_size=grid_size)

    head = _DetectionProbeHead(
        embed_dim=int(train_patches.shape[-1]),
        hidden_dim=int(task.hidden_dim),
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=task.learning_rate)
    positive = float(train_objectness.sum().item())
    negative = float(train_objectness.numel() - train_objectness.sum().item())
    pos_weight = None
    if positive > 0.0:
        pos_weight = torch.tensor(max(1.0, negative / positive), dtype=torch.float32, device=device)
    objectness_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(int(task.epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits, box_regression = head(train_patches)
        loss = objectness_loss(logits, train_objectness)
        positive_mask = train_objectness > 0.5
        if positive_mask.any():
            predicted_boxes = torch.sigmoid(box_regression[positive_mask])
            loss = loss + F.l1_loss(predicted_boxes, train_boxes[positive_mask])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, box_regression = head(val_patches)
        predictions = _decode_detection_predictions(
            logits=logits,
            box_regression=box_regression,
            image_size=image_size,
            grid_size=grid_size,
            score_threshold=float(task.threshold),
            max_detections=int(task.max_detections_per_image),
            nms_iou_threshold=float(task.nms_iou_threshold),
        )

    ap50 = _average_precision_at_iou(predictions, val_gt_boxes, iou_threshold=0.50)
    ap50_95 = float(
        np.mean(
            [
                _average_precision_at_iou(predictions, val_gt_boxes, iou_threshold=threshold)
                for threshold in np.arange(0.50, 1.00, 0.05)
            ]
        )
    )
    metrics = {
        "task_type": "detection",
        "ap50": ap50,
        "ap50_95": ap50_95,
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["detection"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _evaluate_retrieval_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    gallery_features = F.normalize(
        _extract_global_features(
            encoder,
            train_frame[task.image_column].tolist(),
            batch_size=task.batch_size,
            device=device,
        ).float(),
        dim=-1,
    )
    query_features = F.normalize(
        _extract_global_features(
            encoder,
            val_frame[task.image_column].tolist(),
            batch_size=task.batch_size,
            device=device,
        ).float(),
        dim=-1,
    )

    gallery_labels = train_frame[task.label_column].tolist()
    query_labels = val_frame[task.label_column].tolist()
    similarity = query_features @ gallery_features.transpose(0, 1)
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)

    def _recall_at(k: int) -> float:
        hits = 0
        limit = min(k, sorted_indices.shape[1])
        for query_index, target_label in enumerate(query_labels):
            top_indices = sorted_indices[query_index, :limit].tolist()
            if any(gallery_labels[index] == target_label for index in top_indices):
                hits += 1
        return float(hits / max(1, len(query_labels)))

    metrics = {
        "task_type": "retrieval",
        "recall_at_1": _recall_at(1),
        "recall_at_5": _recall_at(5),
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["retrieval"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _evaluate_context_augmentation_task(
    task: ProbeTaskConfig,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    *,
    encoder: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    base_train = _extract_global_features(
        encoder,
        train_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    base_val = _extract_global_features(
        encoder,
        val_frame[task.image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    context_train = _extract_global_features(
        encoder,
        train_frame[task.context_image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )
    context_val = _extract_global_features(
        encoder,
        val_frame[task.context_image_column].tolist(),
        batch_size=task.batch_size,
        device=device,
    )

    label_values = list(train_frame[task.label_column].tolist()) + list(val_frame[task.label_column].tolist())
    label_to_index = {value: index for index, value in enumerate(sorted(set(label_values)))}
    train_labels = torch.tensor([label_to_index[value] for value in train_frame[task.label_column]], dtype=torch.long, device=device)
    val_labels = torch.tensor([label_to_index[value] for value in val_frame[task.label_column]], dtype=torch.long, device=device)

    base_metrics = _fit_and_score_classifier(
        train_features=base_train,
        train_labels=train_labels,
        val_features=base_val,
        val_labels=val_labels,
        learning_rate=task.learning_rate,
        epochs=task.epochs,
        hidden_dim=task.hidden_dim,
        device=device,
    )
    context_metrics = _fit_and_score_classifier(
        train_features=torch.cat([base_train, context_train], dim=1),
        train_labels=train_labels,
        val_features=torch.cat([base_val, context_val], dim=1),
        val_labels=val_labels,
        learning_rate=task.learning_rate,
        epochs=task.epochs,
        hidden_dim=task.hidden_dim,
        device=device,
    )
    metrics = {
        "task_type": "context_augmentation",
        "base_accuracy": base_metrics["accuracy"],
        "base_macro_f1": base_metrics["macro_f1"],
        "context_accuracy": context_metrics["accuracy"],
        "context_macro_f1": context_metrics["macro_f1"],
        "context_gain": context_metrics["macro_f1"] - base_metrics["macro_f1"],
        "primary_metric_name": _DEFAULT_PRIMARY_METRICS["context_augmentation"],
    }
    metrics["primary_metric"] = metrics[metrics["primary_metric_name"]]
    return metrics


def _extract_global_features(
    encoder: nn.Module,
    image_paths: Sequence[str | Path],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    features: list[torch.Tensor] = []
    for paths in _batched(image_paths, batch_size):
        batch = torch.stack([_load_rgb_tensor(path) for path in paths], dim=0).to(device)
        with torch.no_grad():
            outputs = encoder.forward_views(global_views=batch)
        features.append(outputs["global_cls"].detach().to(device))
    return torch.cat(features, dim=0)


def _extract_patch_features_and_masks(
    encoder: nn.Module,
    frame: pd.DataFrame,
    *,
    image_column: str,
    mask_column: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    patch_features: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    grid_size: tuple[int, int] | None = None
    mask_size: tuple[int, int] | None = None
    rows = frame[[image_column, mask_column]].to_dict("records")
    for chunk in _batched(rows, batch_size):
        images = torch.stack([_load_rgb_tensor(item[image_column]) for item in chunk], dim=0).to(device)
        batch_masks = torch.stack([_load_mask_tensor(item[mask_column]) for item in chunk], dim=0).to(device)
        with torch.no_grad():
            outputs = encoder.forward_views(global_views=images)
        patch_batch = outputs["global_patches"].detach().to(device)
        patch_features.append(patch_batch)
        masks.append(batch_masks)
        current_grid = (int(images.shape[2] // getattr(encoder, "patch_size", 16)), int(images.shape[3] // getattr(encoder, "patch_size", 16)))
        current_mask_size = (int(batch_masks.shape[2]), int(batch_masks.shape[3]))
        if grid_size is None:
            grid_size = current_grid
            mask_size = current_mask_size
        elif grid_size != current_grid or mask_size != current_mask_size:
            raise ValueError("Segmentation probe currently expects consistent image and mask sizes within each task.")
    if grid_size is None or mask_size is None:
        raise ValueError("Segmentation probe received no samples.")
    return (
        torch.cat(patch_features, dim=0),
        torch.cat(masks, dim=0),
        grid_size,
        mask_size,
    )


def _extract_patch_features_and_boxes(
    encoder: nn.Module,
    frame: pd.DataFrame,
    *,
    image_column: str,
    boxes_column: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[tuple[float, float, float, float]]], tuple[int, int], tuple[int, int]]:
    patch_features: list[torch.Tensor] = []
    objectness_targets: list[torch.Tensor] = []
    box_targets: list[torch.Tensor] = []
    ground_truth_boxes: list[list[tuple[float, float, float, float]]] = []
    grid_size: tuple[int, int] | None = None
    image_size: tuple[int, int] | None = None

    rows = frame[[image_column, boxes_column]].to_dict("records")
    for chunk in _batched(rows, batch_size):
        images = torch.stack([_load_rgb_tensor(item[image_column]) for item in chunk], dim=0).to(device)
        with torch.no_grad():
            outputs = encoder.forward_views(global_views=images)
        patch_batch = outputs["global_patches"].detach().to(device)
        patch_features.append(patch_batch)

        current_grid = (
            int(images.shape[2] // getattr(encoder, "patch_size", 16)),
            int(images.shape[3] // getattr(encoder, "patch_size", 16)),
        )
        current_image_size = (int(images.shape[2]), int(images.shape[3]))
        if grid_size is None:
            grid_size = current_grid
            image_size = current_image_size
        elif grid_size != current_grid or image_size != current_image_size:
            raise ValueError("Detection probe currently expects consistent image sizes within each task.")

        objectness_batch = torch.zeros(
            patch_batch.shape[0],
            patch_batch.shape[1],
            dtype=torch.float32,
            device=device,
        )
        box_batch = torch.zeros(
            patch_batch.shape[0],
            patch_batch.shape[1],
            4,
            dtype=torch.float32,
            device=device,
        )
        patch_size = getattr(encoder, "patch_size", 16)
        for sample_index, item in enumerate(chunk):
            boxes = _parse_box_annotations(item[boxes_column])
            ground_truth_boxes.append([annotation["bbox_xyxy"] for annotation in boxes])
            for annotation in boxes:
                x1, y1, x2, y2 = annotation["bbox_xyxy"]
                center_x = max(0.0, min(current_image_size[1] - 1.0, (x1 + x2) / 2.0))
                center_y = max(0.0, min(current_image_size[0] - 1.0, (y1 + y2) / 2.0))
                col_index = min(current_grid[1] - 1, int(center_x // patch_size))
                row_index = min(current_grid[0] - 1, int(center_y // patch_size))
                patch_index = row_index * current_grid[1] + col_index
                cell_left = col_index * patch_size
                cell_top = row_index * patch_size
                objectness_batch[sample_index, patch_index] = 1.0
                box_batch[sample_index, patch_index] = torch.tensor(
                    [
                        (x1 - cell_left) / patch_size,
                        (y1 - cell_top) / patch_size,
                        (x2 - cell_left) / patch_size,
                        (y2 - cell_top) / patch_size,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
        objectness_targets.append(objectness_batch)
        box_targets.append(box_batch)

    if grid_size is None or image_size is None:
        raise ValueError("Detection probe received no samples.")
    return (
        torch.cat(patch_features, dim=0),
        torch.cat(objectness_targets, dim=0),
        torch.cat(box_targets, dim=0),
        ground_truth_boxes,
        grid_size,
        image_size,
    )


def _build_mlp_head(in_dim: int, out_dim: int, *, hidden_dim: int) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _fit_and_score_classifier(
    *,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    learning_rate: float,
    epochs: int,
    hidden_dim: int,
    device: torch.device,
) -> dict[str, float]:
    num_classes = int(train_labels.max().item()) + 1
    head = _build_mlp_head(train_features.shape[1], num_classes, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_features)
        loss = F.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_logits = head(val_features)
        predicted = val_logits.argmax(dim=1).cpu().numpy()
        truth = val_labels.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


class _SegmentationProbeHead(nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
        else:
            self.net = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(
        self,
        patch_features: torch.Tensor,
        *,
        grid_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        outputs = patch_features.transpose(1, 2).reshape(
            patch_features.shape[0],
            patch_features.shape[2],
            grid_size[0],
            grid_size[1],
        )
        logits = self.net(outputs)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class _DetectionProbeHead(nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.objectness_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.box_head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 4),
            )
        else:
            self.objectness_head = nn.Linear(embed_dim, 1)
            self.box_head = nn.Linear(embed_dim, 4)

    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        objectness_logits = self.objectness_head(patch_features).squeeze(-1)
        box_regression = self.box_head(patch_features)
        return objectness_logits, box_regression


def _augment_patch_features_with_positions(
    patch_features: torch.Tensor,
    *,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    grid_h, grid_w = grid_size
    row_positions = torch.linspace(0.0, 1.0, steps=grid_h, device=patch_features.device, dtype=patch_features.dtype)
    col_positions = torch.linspace(0.0, 1.0, steps=grid_w, device=patch_features.device, dtype=patch_features.dtype)
    row_grid, col_grid = torch.meshgrid(row_positions, col_positions, indexing="ij")
    position_features = torch.stack([row_grid, col_grid], dim=-1).reshape(1, grid_h * grid_w, 2)
    position_features = position_features.expand(patch_features.shape[0], -1, -1)
    return torch.cat([patch_features, position_features], dim=-1)


def _probe_task_from_mapping(payload: Any) -> ProbeTaskConfig:
    if not isinstance(payload, dict):
        raise ValueError("Each probe task must be a mapping.")
    task_type = str(payload.get("task_type", "")).strip().lower()
    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(f"Unsupported probe task_type '{task_type}'.")
    return ProbeTaskConfig(
        name=str(payload["name"]),
        task_type=task_type,
        train_manifest=Path(payload["train_manifest"]).expanduser().resolve(),
        val_manifest=Path(payload["val_manifest"]).expanduser().resolve(),
        image_column=str(payload.get("image_column", "image_path")),
        context_image_column=str(payload.get("context_image_column", "context_image_path")),
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
    )


def _load_rgb_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(Path(path)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_mask_tensor(path: str | Path) -> torch.Tensor:
    with Image.open(Path(path)) as image:
        array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy((array >= 0.5).astype(np.float32)).unsqueeze(0)


def _parse_box_annotations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, float) and math.isnan(payload):
        return []
    parsed = json.loads(payload) if isinstance(payload, str) else payload
    if not parsed:
        return []

    annotations: list[dict[str, Any]] = []
    for entry in parsed:
        polygon = _entry_to_polygon(entry)
        bounds = polygon.bounds
        annotations.append(
            {
                "bbox_xyxy": (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])),
                "polygon": polygon,
            }
        )
    return annotations


def _batched(sequence: Sequence[Any], batch_size: int) -> Sequence[Sequence[Any]]:
    for start in range(0, len(sequence), max(1, int(batch_size))):
        yield sequence[start : start + max(1, int(batch_size))]


def _resolve_selection_metric(metrics: dict[str, Any], *, requested_metric: str) -> tuple[str, float]:
    if requested_metric and requested_metric not in {"", "mean_primary_metric"} and requested_metric in metrics:
        return requested_metric, float(metrics[requested_metric])
    return str(metrics["primary_metric_name"]), float(metrics["primary_metric"])


def _decode_detection_predictions(
    *,
    logits: torch.Tensor,
    box_regression: torch.Tensor,
    image_size: tuple[int, int],
    grid_size: tuple[int, int],
    score_threshold: float,
    max_detections: int,
    nms_iou_threshold: float,
) -> list[list[dict[str, Any]]]:
    scores = torch.sigmoid(logits)
    normalized_boxes = torch.sigmoid(box_regression)
    height, width = image_size
    grid_h, grid_w = grid_size
    cell_width = width / grid_w
    cell_height = height / grid_h
    predictions: list[list[dict[str, Any]]] = []
    for sample_index in range(scores.shape[0]):
        candidate_indices = torch.nonzero(scores[sample_index] >= score_threshold, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            predictions.append([])
            continue
        sample_predictions: list[dict[str, Any]] = []
        for patch_index in candidate_indices.tolist():
            score = float(scores[sample_index, patch_index].item())
            row_index = patch_index // grid_w
            col_index = patch_index % grid_w
            cell_left = col_index * cell_width
            cell_top = row_index * cell_height
            x1, y1, x2, y2 = normalized_boxes[sample_index, patch_index].tolist()
            left = float(cell_left + (min(x1, x2) * cell_width))
            top = float(cell_top + (min(y1, y2) * cell_height))
            right = float(cell_left + (max(x1, x2) * cell_width))
            bottom = float(cell_top + (max(y1, y2) * cell_height))
            if right <= left or bottom <= top:
                continue
            sample_predictions.append(
                {
                    "score": score,
                    "bbox_xyxy": (left, top, right, bottom),
                    "polygon": shapely_box(left, top, right, bottom),
                }
            )
        sample_predictions.sort(key=lambda item: item["score"], reverse=True)
        predictions.append(_nms_detection_predictions(sample_predictions, iou_threshold=nms_iou_threshold)[:max_detections])
    return predictions


def _nms_detection_predictions(
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in predictions:
        if all(_polygon_iou(candidate["polygon"], existing["polygon"]) <= iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def _average_precision_at_iou(
    predictions: list[list[dict[str, Any]]],
    ground_truth_boxes: list[list[tuple[float, float, float, float]]],
    *,
    iou_threshold: float,
) -> float:
    total_gt = sum(len(items) for items in ground_truth_boxes)
    if total_gt == 0:
        return 0.0

    flattened_predictions: list[tuple[float, int, Polygon]] = []
    for image_index, image_predictions in enumerate(predictions):
        for prediction in image_predictions:
            flattened_predictions.append((float(prediction["score"]), image_index, prediction["polygon"]))
    flattened_predictions.sort(key=lambda item: item[0], reverse=True)

    matched: dict[int, set[int]] = {index: set() for index in range(len(ground_truth_boxes))}
    true_positives: list[float] = []
    false_positives: list[float] = []
    gt_polygons = [[shapely_box(*bbox) for bbox in items] for items in ground_truth_boxes]
    for _score, image_index, polygon in flattened_predictions:
        best_iou = 0.0
        best_match = None
        for gt_index, gt_polygon in enumerate(gt_polygons[image_index]):
            if gt_index in matched[image_index]:
                continue
            iou = _polygon_iou(polygon, gt_polygon)
            if iou > best_iou:
                best_iou = iou
                best_match = gt_index
        if best_match is not None and best_iou >= iou_threshold:
            matched[image_index].add(best_match)
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    if not true_positives:
        return 0.0
    tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recall = tp / max(1.0, float(total_gt))
    precision = tp / np.maximum(tp + fp, 1e-9)

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _entry_to_polygon(entry: Any) -> Polygon:
    if isinstance(entry, dict):
        if "polygon" in entry:
            coords = entry["polygon"]
            return Polygon([(float(coords[index]), float(coords[index + 1])) for index in range(0, len(coords), 2)])
        if "bbox_xyxy" in entry:
            x1, y1, x2, y2 = entry["bbox_xyxy"]
            return shapely_box(float(x1), float(y1), float(x2), float(y2))
        if "bbox" in entry:
            x1, y1, x2, y2 = entry["bbox"]
            return shapely_box(float(x1), float(y1), float(x2), float(y2))
    if isinstance(entry, (list, tuple)):
        if len(entry) == 4:
            x1, y1, x2, y2 = entry
            return shapely_box(float(x1), float(y1), float(x2), float(y2))
        if len(entry) == 8:
            return Polygon([(float(entry[index]), float(entry[index + 1])) for index in range(0, 8, 2)])
    raise ValueError(f"Unsupported detection annotation entry: {entry!r}")


def _dice_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.float()
    truth = target.float()
    intersection = (pred * truth).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denom + 1e-6)
    return float(dice.mean().cpu())


def _mean_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.float()
    truth = target.float()
    intersection = (pred * truth).sum(dim=(1, 2, 3))
    union = ((pred + truth) > 0).float().sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return float(iou.mean().cpu())


def _boundary_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred_boundary = _extract_boundary_map(prediction.float())
    truth_boundary = _extract_boundary_map(target.float())
    true_positive = (pred_boundary * truth_boundary).sum(dim=(1, 2, 3))
    precision = (true_positive + 1e-6) / (pred_boundary.sum(dim=(1, 2, 3)) + 1e-6)
    recall = (true_positive + 1e-6) / (truth_boundary.sum(dim=(1, 2, 3)) + 1e-6)
    score = (2.0 * precision * recall + 1e-6) / (precision + recall + 1e-6)
    return float(score.mean().cpu())


def _extract_boundary_map(mask: torch.Tensor) -> torch.Tensor:
    dilation = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    erosion = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return ((dilation - erosion) > 0).float()


def _polygon_iou(left: Polygon, right: Polygon) -> float:
    if left.is_empty or right.is_empty:
        return 0.0
    union = left.union(right).area
    if union <= 0.0:
        return 0.0
    return float(left.intersection(right).area / union)
