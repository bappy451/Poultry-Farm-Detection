from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence
import xml.etree.ElementTree as ET

import fiona
import numpy as np
import pandas as pd
import pyproj
import rasterio
from PIL import Image
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from shapely.wkt import dumps as wkt_dumps

from poultry_geo.geotiff import DEFAULT_JPEG_QUALITY, find_geotiffs


def _normalize_rgb(image: np.ndarray, out_dtype: np.dtype = np.uint8) -> np.ndarray:
    rgb = np.moveaxis(_ensure_three_channels(image), 0, -1)
    if rgb.dtype == out_dtype:
        return np.ascontiguousarray(rgb)
    rgb = rgb.astype(np.float32)
    rgb_min = float(rgb.min())
    rgb_max = float(rgb.max())
    if np.isclose(rgb_min, rgb_max):
        return np.ascontiguousarray(np.zeros_like(rgb, dtype=out_dtype))
    scaled = (rgb - rgb_min) / (rgb_max - rgb_min)
    if np.issubdtype(out_dtype, np.integer):
        return np.ascontiguousarray(
            np.clip(scaled * np.iinfo(out_dtype).max, 0, np.iinfo(out_dtype).max).astype(out_dtype)
        )
    return np.ascontiguousarray(scaled.astype(out_dtype))


def _ensure_three_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] == 0:
        raise ValueError("Expected image data in (bands, height, width) format.")
    if image.shape[0] >= 3:
        return image[:3]
    if image.shape[0] == 1:
        return np.repeat(image, 3, axis=0)
    return np.concatenate([image, image[-1:, :, :]], axis=0)


def convert_geotiff_to_png_or_jpg(
    input_dir: str | Path,
    output_dir: str | Path = "converted_images",
    out_ext: str = "png",
    out_dtype: np.dtype = np.uint8,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_ext = out_ext.lower()
    if normalized_ext not in {"png", "jpg", "jpeg"}:
        raise ValueError("out_ext must be one of: png, jpg, jpeg")

    written: list[Path] = []
    tif_paths = find_geotiffs(input_dir)
    if not tif_paths:
        raise ValueError(f"No GeoTIFF files found under {input_dir}.")

    for index, tif_path in enumerate(tif_paths, start=1):
        with rasterio.open(tif_path) as src:
            band_indexes = list(range(1, min(3, src.count) + 1))
            image = src.read(band_indexes)
        rgb = _normalize_rgb(image, out_dtype=out_dtype)
        suffix = ".jpg" if normalized_ext in {"jpg", "jpeg"} else ".png"
        if input_dir.is_file():
            output_path = output_dir / f"{tif_path.stem}{suffix}"
        else:
            relative_parent = tif_path.relative_to(input_dir).parent
            output_path = output_dir / relative_parent / f"{tif_path.stem}{suffix}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image = Image.fromarray(rgb)
        if suffix == ".jpg" and pil_image.mode == "RGBA":
            pil_image = pil_image.convert("RGB")
        if suffix == ".jpg":
            pil_image.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
        else:
            pil_image.save(output_path, format="PNG", optimize=True)
        written.append(output_path)
        if progress_callback is not None:
            progress_callback(index, len(tif_paths), tif_path.name)
    return written


def sample_random_rows(
    input_path: str | Path,
    n_rows: int,
    output_csv: str | Path,
    seed: int = 27,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    if n_rows > len(df):
        raise ValueError(f"Requested {n_rows} rows, but file has only {len(df)} rows.")

    rng = np.random.default_rng(seed)
    remaining_idx = df.index.to_numpy()
    selected_idx: list[int] = []
    for _ in range(n_rows):
        pick = int(rng.choice(remaining_idx))
        selected_idx.append(pick)
        remaining_idx = remaining_idx[remaining_idx != pick]

    sampled = df.loc[selected_idx].reset_index(drop=True)
    sampled.to_csv(output_csv, index=False)
    return sampled


def shapefile_to_csv(
    shapefile_path: str | Path,
    output_csv_path: str | Path,
) -> Path:
    shapefile_path = Path(shapefile_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with fiona.open(shapefile_path, "r") as src:
        crs_obj = pyproj.CRS(src.crs) if src.crs else pyproj.CRS("EPSG:4326")
        is_projected = crs_obj.is_projected
        transformer = None
        if is_projected:
            transformer = pyproj.Transformer.from_crs(crs_obj, pyproj.CRS("EPSG:4326"), always_xy=True)

        records: list[dict[str, object]] = []
        for feature in src:
            props = dict(feature["properties"])
            geom = shape(feature["geometry"])
            geom_latlon = (
                shapely_transform(transformer.transform, geom) if transformer is not None else geom
            )

            props["geometry_type"] = geom.geom_type
            props["wkt"] = wkt_dumps(geom)
            props["area"] = geom.area
            props["bbox_xmin"], props["bbox_ymin"], props["bbox_xmax"], props["bbox_ymax"] = geom.bounds
            props["centroid_lon"] = geom_latlon.centroid.x
            props["centroid_lat"] = geom_latlon.centroid.y

            if geom.geom_type == "Polygon":
                props["coords"] = json.dumps(list(geom.exterior.coords))
            elif geom.geom_type == "MultiPolygon":
                props["coords"] = json.dumps([list(part.exterior.coords) for part in geom.geoms])
            else:
                props["coords"] = json.dumps(list(getattr(geom, "coords", [])))
            records.append(props)

    pd.DataFrame(records).to_csv(output_csv_path, index=False)
    return output_csv_path


def shapefile_to_projected_csv(
    shapefile_path: str | Path,
    output_csv_path: str | Path,
    target_epsg: str | int = "26916",
) -> Path:
    shapefile_path = Path(shapefile_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = pyproj.CRS.from_user_input(target_epsg)

    with fiona.open(shapefile_path, "r") as src:
        source_crs = pyproj.CRS(src.crs) if src.crs else target_crs
        transformer = None
        if source_crs != target_crs:
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

        records: list[dict[str, object]] = []
        for feature in src:
            props = dict(feature["properties"])
            geom = shape(feature["geometry"])
            geom_proj = shapely_transform(transformer.transform, geom) if transformer is not None else geom

            props["geometry_type"] = geom_proj.geom_type
            props["wkt"] = wkt_dumps(geom_proj)
            props["area"] = geom_proj.area
            props["bbox_xmin"], props["bbox_ymin"], props["bbox_xmax"], props["bbox_ymax"] = geom_proj.bounds
            props["centroid_x"] = geom_proj.centroid.x
            props["centroid_y"] = geom_proj.centroid.y

            if geom_proj.geom_type == "Polygon":
                props["coords"] = json.dumps(list(geom_proj.exterior.coords))
            elif geom_proj.geom_type == "MultiPolygon":
                props["coords"] = json.dumps([list(part.exterior.coords) for part in geom_proj.geoms])
            else:
                props["coords"] = json.dumps(list(getattr(geom_proj, "coords", [])))
            records.append(props)

    pd.DataFrame(records).to_csv(output_csv_path, index=False)
    return output_csv_path


def _ring_to_kml_coords(
    xy_list: Iterable[Sequence[float]],
    transformer: pyproj.Transformer,
) -> list[str]:
    coords: list[str] = []
    for xy in xy_list:
        if len(xy) < 2:
            continue
        lon, lat = transformer.transform(float(xy[0]), float(xy[1]))
        coords.append(f"{lon},{lat},0")
    if len(coords) >= 3 and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def csv_to_kml_polygons(
    input_csv: str | Path,
    output_kml: str | Path,
    source_epsg: str | int = 26916,
    name_col: str = "tile_name",
    polygon_col: str = "mask_points_gis",
    centroid_cols: tuple[str, str] = ("centroid_x", "centroid_y"),
    fallback_xy_cols: tuple[str, str] = ("longitude", "latitude"),
    desc_cols: tuple[str, ...] = ("class", "confidence", "area_geo", "length", "width"),
    include_extended_data: bool = True,
) -> Path:
    input_csv = Path(input_csv)
    output_kml = Path(output_kml)
    output_kml.parent.mkdir(parents=True, exist_ok=True)

    src = pyproj.CRS.from_user_input(source_epsg)
    dst = pyproj.CRS.from_epsg(4326)
    transformer = pyproj.Transformer.from_crs(src, dst, always_xy=True)

    kml = ET.Element("kml", {"xmlns": "http://www.opengis.net/kml/2.2"})
    document = ET.SubElement(kml, "Document")

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for index, row in enumerate(reader, start=1):
            placemark = ET.SubElement(document, "Placemark")
            name = row.get(name_col, "").strip() if name_col in fields else f"Feature {index}"
            ET.SubElement(placemark, "name").text = name

            description_parts = [f"{column}: {row[column]}" for column in desc_cols if column in fields and row.get(column)]
            if description_parts:
                ET.SubElement(placemark, "description").text = "\n".join(description_parts)

            if include_extended_data:
                extended = ET.SubElement(placemark, "ExtendedData")
                for key, value in row.items():
                    if key == polygon_col:
                        continue
                    data = ET.SubElement(extended, "Data", name=key)
                    ET.SubElement(data, "value").text = "" if value is None else str(value)

            wrote_geometry = False
            polygon_value = row.get(polygon_col, "")
            if isinstance(polygon_value, str) and polygon_value.strip().startswith("["):
                try:
                    polygon_object = ast.literal_eval(polygon_value)
                except (ValueError, SyntaxError):
                    polygon_object = None

                if polygon_object:
                    if isinstance(polygon_object[0][0], (int, float)):
                        coords = _ring_to_kml_coords(polygon_object, transformer)
                        if len(coords) >= 4:
                            polygon = ET.SubElement(placemark, "Polygon")
                            outer = ET.SubElement(polygon, "outerBoundaryIs")
                            ring = ET.SubElement(outer, "LinearRing")
                            ET.SubElement(ring, "coordinates").text = " ".join(coords)
                            wrote_geometry = True
                    elif isinstance(polygon_object[0][0], (list, tuple)):
                        multi_geometry = ET.SubElement(placemark, "MultiGeometry")
                        any_written = False
                        for ring_like in polygon_object:
                            coords = _ring_to_kml_coords(ring_like, transformer)
                            if len(coords) < 4:
                                continue
                            polygon = ET.SubElement(multi_geometry, "Polygon")
                            outer = ET.SubElement(polygon, "outerBoundaryIs")
                            ring = ET.SubElement(outer, "LinearRing")
                            ET.SubElement(ring, "coordinates").text = " ".join(coords)
                            any_written = True
                        wrote_geometry = any_written

            if not wrote_geometry:
                point = ET.SubElement(placemark, "Point")
                x_col, y_col = centroid_cols
                if x_col in fields and y_col in fields and row.get(x_col) and row.get(y_col):
                    lon, lat = transformer.transform(float(row[x_col]), float(row[y_col]))
                else:
                    lon_col, lat_col = fallback_xy_cols
                    lon = float(row[lon_col])
                    lat = float(row[lat_col])
                ET.SubElement(point, "coordinates").text = f"{lon},{lat},0"

    ET.ElementTree(kml).write(output_kml, encoding="utf-8", xml_declaration=True)
    return output_kml
