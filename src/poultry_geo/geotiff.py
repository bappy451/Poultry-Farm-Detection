from __future__ import annotations

import os
import math
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.windows import Window

from poultry_geo.models import TileSlice


SLICE_NAME_RE = re.compile(r"slice_(?P<row>\d+)_(?P<col>\d+)\.tif{1,2}$", re.IGNORECASE)
DEFAULT_JPEG_QUALITY = 95


def find_geotiffs(folder_path: str | Path) -> list[Path]:
    folder = Path(folder_path)
    if folder.is_file():
        return [folder] if folder.suffix.lower() in {".tif", ".tiff"} else []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


def _resolve_stride(tile_size: int, stride: int | None) -> int:
    resolved_stride = tile_size if stride is None else stride
    if tile_size <= 0 or resolved_stride <= 0:
        raise ValueError("tile_size and stride must be positive integers.")
    if resolved_stride > tile_size:
        raise ValueError("stride must be less than or equal to tile_size to guarantee full coverage.")
    return resolved_stride


def compute_tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return tile starts that fully cover a dimension while snapping the last window to the edge."""
    if length <= 0:
        return []
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    if not starts:
        starts = [0]

    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iter_tile_windows(
    width: int,
    height: int,
    tile_size: int,
    stride: int | None = None,
) -> Iterator[tuple[int, int, int, int, int, int]]:
    stride = _resolve_stride(tile_size, stride)
    y_starts = compute_tile_starts(height, tile_size, stride)
    x_starts = compute_tile_starts(width, tile_size, stride)

    for row_index, y_start in enumerate(y_starts):
        for col_index, x_start in enumerate(x_starts):
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            yield row_index, col_index, x_start, y_start, x_end, y_end


def _normalize_rgb(image_array: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(_ensure_three_channels(image_array), 0, -1)
    if rgb.dtype == np.uint8:
        return np.ascontiguousarray(rgb)
    rgb = rgb.astype(np.float32)
    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    if math.isclose(rgb_min, rgb_max):
        return np.zeros_like(rgb, dtype=np.uint8)
    scaled = (rgb - rgb_min) / (rgb_max - rgb_min)
    return np.ascontiguousarray(np.clip(scaled * 255.0, 0, 255).astype(np.uint8))


def _ensure_three_channels(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim != 3 or image_array.shape[0] == 0:
        raise ValueError("Expected image data in (bands, height, width) format.")
    if image_array.shape[0] >= 3:
        return image_array[:3]
    if image_array.shape[0] == 1:
        return np.repeat(image_array, 3, axis=0)
    return np.concatenate([image_array, image_array[-1:, :, :]], axis=0)


def _resolve_output_format(output_format: str | None, to_jpeg: bool) -> str:
    if output_format is None:
        return "jpg" if to_jpeg else "tif"

    normalized = output_format.lower()
    if normalized == "tiff":
        normalized = "tif"
    if normalized not in {"tif", "png", "jpg", "jpeg"}:
        raise ValueError("output_format must be one of: tif, png, jpg, jpeg")
    if to_jpeg and normalized not in {"jpg", "jpeg"}:
        raise ValueError("to_jpeg=True cannot be combined with a non-JPEG output_format.")
    return normalized


def _write_image_tile(
    image_array: np.ndarray,
    output_path: Path,
    output_format: str,
    jpeg_quality: int,
) -> None:
    image = Image.fromarray(_normalize_rgb(image_array))
    if output_format in {"jpg", "jpeg"} and image.mode == "RGBA":
        image = image.convert("RGB")
    if output_format in {"jpg", "jpeg"}:
        image.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return
    image.save(output_path, format="PNG", optimize=True)


def iter_slice_geotiff(
    geotiff_path: str | Path,
    tile_size: int,
    stride: int | None = None,
) -> Iterator[TileSlice]:
    """Yield edge-snapped windows from a GeoTIFF without zero padding."""
    stride = _resolve_stride(tile_size, stride)
    geotiff_path = Path(geotiff_path)

    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        for row_index, col_index, x_start, y_start, x_end, y_end in iter_tile_windows(
            src.width,
            src.height,
            tile_size,
            stride,
        ):
            window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
            image_array = src.read(window=window)
            valid_shape = (image_array.shape[1], image_array.shape[2])
            new_transform = from_origin(
                transform.c + x_start * transform.a,
                transform.f + y_start * transform.e,
                transform.a,
                -transform.e,
            )
            yield TileSlice(
                image_array=image_array,
                transform=new_transform,
                crs=src.crs,
                dtype=src.dtypes[0],
                count=src.count,
                position=(row_index, col_index),
                offset_xy=(x_start, y_start),
                valid_shape=valid_shape,
                source_path=geotiff_path,
            )


def slice_geotiff_to_memory(
    geotiff_path: str | Path,
    tile_size: int,
    stride: int | None = None,
) -> list[TileSlice]:
    """Read a GeoTIFF into edge-snapped windows without zero padding."""
    return list(iter_slice_geotiff(geotiff_path, tile_size, stride))


def slice_geotiff_to_files(
    geotiff_path: str | Path,
    output_dir: str | Path,
    tile_size: int,
    stride: int | None = None,
    to_jpeg: bool = False,
    output_format: str | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    workers: int | str | None = 1,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output_format = _resolve_output_format(output_format, to_jpeg)

    # File-level multiprocessing is orchestrated by slice_geotiff_collection_to_files.
    # A single GeoTIFF is always processed inline so one worker owns the full file
    # from open through final tile writes.
    written_paths: list[Path] = []
    for tile in iter_slice_geotiff(geotiff_path, tile_size=tile_size, stride=stride):
        written_paths.append(_write_single_tile(tile, output_dir, resolved_output_format, jpeg_quality))
    return written_paths


def _write_single_tile(
    tile: TileSlice,
    output_dir: Path,
    resolved_output_format: str,
    jpeg_quality: int,
) -> Path:
    row_index, col_index = tile.position
    if resolved_output_format == "tif":
        output_path = output_dir / f"slice_{row_index}_{col_index}.tif"
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=tile.image_array.shape[1],
            width=tile.image_array.shape[2],
            count=tile.count,
            dtype=tile.dtype,
            crs=tile.crs,
            transform=tile.transform,
        ) as dst:
            dst.write(tile.image_array)
            dst.update_tags(
                offset_x=tile.offset_xy[0],
                offset_y=tile.offset_xy[1],
                valid_height=tile.valid_shape[0],
                valid_width=tile.valid_shape[1],
            )
    else:
        suffix = ".jpg" if resolved_output_format in {"jpg", "jpeg"} else ".png"
        output_path = output_dir / f"slice_{row_index}_{col_index}{suffix}"
        _write_image_tile(
            tile.image_array,
            output_path,
            resolved_output_format,
            jpeg_quality=jpeg_quality,
        )
    return output_path


def _write_tile_job(job: tuple) -> str:
    (
        geotiff_path,
        output_dir,
        row_index,
        col_index,
        x_start,
        y_start,
        x_end,
        y_end,
        output_format,
        jpeg_quality,
    ) = job

    with rasterio.open(geotiff_path) as src:
        window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
        image_array = src.read(window=window)
        valid_shape = (image_array.shape[1], image_array.shape[2])
        transform = src.transform
        new_transform = from_origin(
            transform.c + x_start * transform.a,
            transform.f + y_start * transform.e,
            transform.a,
            -transform.e,
        )
        tile = TileSlice(
            image_array=image_array,
            transform=new_transform,
            crs=src.crs,
            dtype=src.dtypes[0],
            count=src.count,
            position=(row_index, col_index),
            offset_xy=(x_start, y_start),
            valid_shape=valid_shape,
            source_path=Path(geotiff_path),
        )
        output_path = _write_single_tile(tile, Path(output_dir), output_format, jpeg_quality)
        return str(output_path)


def _run_slice_jobs_serial(
    jobs: list[tuple[str, str, int, int | None, str | None, int]],
    *,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    written: list[Path] = []
    for index, job in enumerate(jobs, start=1):
        written.extend(Path(path) for path in _slice_single_geotiff_job(job))
        if progress_callback is not None:
            progress_callback(index, len(jobs), Path(job[0]).name)
    return written


def slice_geotiff_collection_to_files(
    input_path: str | Path,
    output_dir: str | Path,
    tile_size: int,
    stride: int | None = None,
    *,
    output_format: str | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    workers: int | str | None = "auto",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    """Slice one GeoTIFF or a directory tree of GeoTIFFs into tile files."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_path.is_file():
        written = slice_geotiff_to_files(
            input_path,
            output_dir,
            tile_size=tile_size,
            stride=stride,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            workers=workers,
        )
        if progress_callback is not None:
            progress_callback(1, 1, input_path.name)
        return written

    geotiff_paths = find_geotiffs(input_path)
    if not geotiff_paths:
        raise ValueError(f"No GeoTIFF files found under {input_path}.")

    resolved_workers = _resolve_worker_count(workers, len(geotiff_paths))
    jobs = [
        (
            str(geotiff_path),
            str((output_dir / geotiff_path.relative_to(input_path).parent / geotiff_path.stem)),
            tile_size,
            stride,
            output_format,
            jpeg_quality,
        )
        for geotiff_path in geotiff_paths
    ]

    if resolved_workers == 1:
        return _run_slice_jobs_serial(jobs, progress_callback=progress_callback)

    written: list[Path] = []
    try:
        executor = ProcessPoolExecutor(max_workers=resolved_workers)
    except (OSError, PermissionError) as exc:
        warnings.warn(
            "ProcessPoolExecutor is unavailable in this environment; "
            "falling back to serial file-level slicing. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _run_slice_jobs_serial(jobs, progress_callback=progress_callback)

    with executor:
        # Each job is a full GeoTIFF from open to final write. We do not split a single
        # GeoTIFF across processes because the user wants one worker to own the entire file.
        for index, result_paths in enumerate(executor.map(_slice_single_geotiff_job, jobs), start=1):
            written.extend(Path(path) for path in result_paths)
            if progress_callback is not None:
                progress_callback(index, len(jobs), Path(jobs[index - 1][0]).name)
    return written


def _resolve_worker_count(workers: int | str | None, num_geotiffs: int) -> int:
    """Normalize a worker request into a concrete file-level process count."""
    if num_geotiffs <= 0:
        return 1

    if workers is None or workers == "auto":
        return max(1, min(num_geotiffs, os.cpu_count() or 1))

    if isinstance(workers, str):
        try:
            parsed_workers = int(workers)
        except ValueError as exc:
            raise ValueError("workers must be a positive integer or 'auto'.") from exc
    else:
        parsed_workers = int(workers)

    if parsed_workers <= 0:
        raise ValueError("workers must be a positive integer or 'auto'.")
    return min(parsed_workers, num_geotiffs)


def _slice_single_geotiff_job(
    job: tuple[str, str, int, int | None, str | None, int],
) -> list[str]:
    """Worker entrypoint: one process owns one full GeoTIFF from open to final write."""
    geotiff_path, output_dir, tile_size, stride, output_format, jpeg_quality = job
    try:
        written = slice_geotiff_to_files(
            geotiff_path,
            output_dir,
            tile_size=tile_size,
            stride=stride,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            workers=1,
        )
        return [str(path) for path in written]
    except Exception as exc:
        raise RuntimeError(f"Failed while slicing GeoTIFF '{geotiff_path}': {exc}") from exc


def stitch_geotiff_tiles(input_dir: str | Path, output_geotiff: str | Path) -> Path:
    """
    Reconstructs a full GeoTIFF from a directory of TIFF tile slices.
    
    This implementation is memory-efficient; it uses windowed writing to process 
    tiles one by one, ensuring that the entire final raster is never loaded 
    into RAM at once. This is critical for high-resolution datasets where 
    the final image can exceed available memory.
    
    Args:
        input_dir: Directory containing 'slice_row_col.tif' files.
        output_geotiff: Destination path for the stitched raster.
    """
    input_dir = Path(input_dir)
    output_geotiff = Path(output_geotiff)
    slice_files = sorted(path for path in input_dir.glob("slice_*_*.tif") if path.is_file())
    if not slice_files:
        raise ValueError(f"No TIFF slice files found in {input_dir}.")

    def extract_indices(path: Path) -> tuple[int, int]:
        match = SLICE_NAME_RE.match(path.name)
        if not match:
            raise ValueError(f"Unsupported tile filename: {path.name}")
        return int(match.group("row")), int(match.group("col"))

    # Read metadata from the first slice to initialize the destination raster
    with rasterio.open(slice_files[0]) as src:
        tile_height = src.height
        tile_width = src.width
        count = src.count
        dtype = src.dtypes[0]
        crs = src.crs
        transform = src.transform

    tile_layouts: list[tuple[Path, int, int, int, int, Any]] = []
    full_height = 0
    full_width = 0

    # First pass: Determine final raster dimensions and tile placements
    for path in slice_files:
        row_index, col_index = extract_indices(path)
        with rasterio.open(path) as src:
            tags = src.tags()
            # Use tags if available (for exact alignment), otherwise fallback to grid-based offsets
            offset_x = int(tags.get("offset_x", col_index * tile_width))
            offset_y = int(tags.get("offset_y", row_index * tile_height))
            valid_height = int(tags.get("valid_height", src.height))
            valid_width = int(tags.get("valid_width", src.width))
            tile_transform = src.transform
        tile_layouts.append((path, offset_x, offset_y, valid_width, valid_height, tile_transform))
        full_width = max(full_width, offset_x + valid_width)
        full_height = max(full_height, offset_y + valid_height)

    # Calculate global transform based on the top-left tile
    top_left_transform = min(tile_layouts, key=lambda item: (item[2], item[1]))[5]

    # Second pass: Progressively write tiles into the destination using Rasterio windows
    with rasterio.open(
        output_geotiff,
        "w",
        driver="GTiff",
        height=full_height,
        width=full_width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=from_origin(top_left_transform.c, top_left_transform.f, top_left_transform.a, -top_left_transform.e),
    ) as dst:
        for path, offset_x, offset_y, valid_width, valid_height, _ in tile_layouts:
            with rasterio.open(path) as src:
                # Read valid pixel data from the slice
                tile_data = src.read(window=Window(0, 0, valid_width, valid_height))
            # Write directly to the corresponding window in the destination raster
            dst.write(tile_data, window=Window(offset_x, offset_y, valid_width, valid_height))

    return output_geotiff
