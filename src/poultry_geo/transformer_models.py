from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(slots=True)
class TrainingRunSummary:
    output_dir: str
    epochs: int
    train_batches: int
    val_batches: int
    final_train_loss: float | None
    final_val_loss: float | None


def _select_torch_device(device: str | None = None):
    import torch

    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_batch_to_device(batch: dict[str, Any], device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [
                {nested_key: nested_value.to(device) if hasattr(nested_value, "to") else nested_value for nested_key, nested_value in item.items()}
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            moved[key] = value
    return moved


def build_detr_processor(model_name: str = "facebook/detr-resnet-50"):
    from transformers import DetrImageProcessor

    return DetrImageProcessor.from_pretrained(model_name)


def build_mask2former_processor(
    model_name: str = "facebook/mask2former-swin-base-coco-instance",
):
    from transformers import Mask2FormerImageProcessor

    return Mask2FormerImageProcessor.from_pretrained(model_name)


class CocoDetectionHFDataset:
    def __init__(self, root: str | Path, ann_file: str | Path, processor: Any) -> None:
        from torchvision.datasets import CocoDetection

        self.coco_dataset = CocoDetection(str(root), str(ann_file))
        self.processor = processor

    def __len__(self) -> int:
        return len(self.coco_dataset)

    def __getitem__(self, idx: int):
        image, target = self.coco_dataset[idx]
        annotations = [
            {
                "bbox": annotation["bbox"],
                "category_id": annotation["category_id"],
                "area": annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3]),
                "iscrowd": annotation.get("iscrowd", 0),
            }
            for annotation in target
        ]
        return image, {"image_id": idx, "annotations": annotations}


def detr_collate_fn(batch, processor: Any):
    images, targets = zip(*batch)
    return processor(images=list(images), annotations=list(targets), return_tensors="pt")


class CocoSegmentationDataset:
    def __init__(self, img_folder: str | Path, ann_file: str | Path, processor: Any) -> None:
        from pycocotools.coco import COCO

        self.img_folder = Path(img_folder)
        self.coco = COCO(str(ann_file))
        self.img_ids = self.coco.getImgIds()
        self.processor = processor

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        import torch

        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        image_info = self.coco.loadImgs(img_id)[0]
        image = Image.open(self.img_folder / image_info["file_name"]).convert("RGB")

        if hasattr(self.processor, "post_process_instance_segmentation"):
            mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            for annotation in annotations:
                ann_mask = self.coco.annToMask(annotation)
                mask[ann_mask == 1] = annotation["category_id"]
            inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "mask_labels": inputs["mask_labels"][0],
                "class_labels": inputs["class_labels"][0],
            }

        width, height = image.size
        boxes: list[list[float]] = []
        class_labels: list[int] = []
        for annotation in annotations:
            x_coord, y_coord, box_width, box_height = annotation["bbox"]
            boxes.append([x_coord / width, y_coord / height, box_width / width, box_height / height])
            class_labels.append(annotation["category_id"])

        labels = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
        encoding = self.processor(images=image, return_tensors="pt")
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}
        encoding["labels"] = labels
        return encoding


def mask2former_collate_fn(batch):
    import torch

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    mask_labels = [item["mask_labels"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    max_num_masks = max(mask.shape[0] for mask in mask_labels)

    padded_mask_labels = []
    padded_class_labels = []
    for masks, classes in zip(mask_labels, class_labels):
        mask_padding = (0, 0, 0, 0, 0, max_num_masks - masks.shape[0])
        padded_mask_labels.append(torch.nn.functional.pad(masks, mask_padding, value=0))
        padded_class_labels.append(
            torch.cat(
                [classes, torch.zeros(max_num_masks - classes.shape[0], dtype=classes.dtype)],
            ),
        )

    return {
        "pixel_values": pixel_values,
        "mask_labels": torch.stack(padded_mask_labels),
        "class_labels": torch.stack(padded_class_labels),
    }


def train_hf_detr(
    train_image_dir: str | Path,
    train_annotations: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "facebook/detr-resnet-50",
    num_labels: int = 2,
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    val_image_dir: str | Path | None = None,
    val_annotations: str | Path | None = None,
    device: str | None = None,
) -> TrainingRunSummary:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DetrForObjectDetection

    processor = build_detr_processor(model_name)
    train_dataset = CocoDetectionHFDataset(train_image_dir, train_annotations, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: detr_collate_fn(batch, processor),
    )
    val_loader = None
    if val_image_dir and val_annotations:
        val_dataset = CocoDetectionHFDataset(val_image_dir, val_annotations, processor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: detr_collate_fn(batch, processor),
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _select_torch_device(device)
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_train_loss: float | None = None
    final_val_loss: float | None = None
    for _epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, resolved_device)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(train_losses)) if train_losses else None

        if val_loader is not None:
            model.eval()
            val_losses: list[float] = []
            with torch.inference_mode():
                for batch in val_loader:
                    batch = _move_batch_to_device(batch, resolved_device)
                    outputs = model(**batch)
                    val_losses.append(float(outputs.loss.detach().cpu()))
            final_val_loss = float(np.mean(val_losses)) if val_losses else None

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingRunSummary(
        output_dir=str(output_dir),
        epochs=epochs,
        train_batches=len(train_loader),
        val_batches=0 if val_loader is None else len(val_loader),
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
    )


def train_mask2former(
    train_image_dir: str | Path,
    train_annotations: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "facebook/mask2former-swin-base-coco-instance",
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | None = None,
) -> TrainingRunSummary:
    import torch
    from torch.utils.data import DataLoader
    from transformers import Mask2FormerForUniversalSegmentation

    processor = build_mask2former_processor(model_name)
    train_dataset = CocoSegmentationDataset(train_image_dir, train_annotations, processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=mask2former_collate_fn,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _select_torch_device(device)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    final_train_loss: float | None = None
    for _epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = _move_batch_to_device(batch, resolved_device)
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        final_train_loss = float(np.mean(train_losses)) if train_losses else None

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "final_train_loss": final_train_loss,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainingRunSummary(
        output_dir=str(output_dir),
        epochs=epochs,
        train_batches=len(train_loader),
        val_batches=0,
        final_train_loss=final_train_loss,
        final_val_loss=None,
    )


def predict_hf_detr(
    image_path: str | Path,
    model_path: str | Path,
    *,
    threshold: float = 0.5,
    device: str | None = None,
) -> list[dict[str, float]]:
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    resolved_device = _select_torch_device(device)
    model = DetrForObjectDetection.from_pretrained(str(model_path)).to(resolved_device)
    processor = DetrImageProcessor.from_pretrained(str(model_path))
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = _move_batch_to_device(inputs, resolved_device)

    with torch.inference_mode():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=resolved_device)
    processed = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

    detections: list[dict[str, float]] = []
    for score, label, box in zip(processed["scores"], processed["labels"], processed["boxes"]):
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        detections.append(
            {
                "score": float(score.item()),
                "label": int(label.item()),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        )
    return detections
