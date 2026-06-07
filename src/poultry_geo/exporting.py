from __future__ import annotations

from pathlib import Path
from typing import Any

from poultry_geo.ultralytics_runtime import load_ultralytics_model


def export_ultralytics_model(
    model_path: str | Path,
    export_format: str = "tflite",
    family: str = "auto",
    device: str | None = None,
    image_size: int | None = None,
    extra_export_kwargs: dict[str, Any] | None = None,
) -> str:
    model = load_ultralytics_model(model_path, family=family)
    kwargs: dict[str, Any] = {"format": export_format}
    if device:
        kwargs["device"] = device
    if image_size:
        kwargs["imgsz"] = image_size
    if extra_export_kwargs:
        kwargs.update(extra_export_kwargs)
    result = model.export(**kwargs)
    return str(result)

