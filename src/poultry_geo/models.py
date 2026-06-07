from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class TileSlice:
    """One edge-snapped tile extracted from a GeoTIFF."""

    image_array: np.ndarray
    transform: Any
    crs: Any
    dtype: str
    count: int
    position: tuple[int, int]
    offset_xy: tuple[int, int]
    # Maintained for compatibility with earlier code; it now matches the real tile size.
    valid_shape: tuple[int, int]
    source_path: Path | None = None
