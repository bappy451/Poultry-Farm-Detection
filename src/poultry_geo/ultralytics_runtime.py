from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_ultralytics_family(model_name: str | Path, family: str = "auto") -> str:
    if family != "auto":
        normalized = family.lower()
        if normalized not in {"yolo", "rtdetr"}:
            raise ValueError("family must be one of: auto, yolo, rtdetr")
        return normalized

    model_str = str(model_name).lower()
    if "rtdetr" in model_str:
        return "rtdetr"
    return "yolo"


def load_ultralytics_model(model_name: str | Path, family: str = "auto") -> Any:
    resolved_family = infer_ultralytics_family(model_name, family=family)
    if resolved_family == "rtdetr":
        from ultralytics import RTDETR

        return RTDETR(str(model_name))

    from ultralytics import YOLO

    return YOLO(str(model_name))

