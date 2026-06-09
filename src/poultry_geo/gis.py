from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable

import fiona
import numpy as np
import pandas as pd
import pyproj
import rasterio
from shapely.geometry import Point, Polygon, box, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree

from poultry_geo.postprocess import _parse_polygon, _select_geometry_column


SQ_M_TO_SQ_FT = 10.7639
M_TO_FT = 3.28084


def create_boundary_shapefiles_from_geotiffs(
    geotiff_folder: str | Path,
    output_folder: str | Path,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[Path]:
    geotiff_folder = Path(geotiff_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    geotiff_paths = sorted(
        path
        for path in geotiff_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    for index, geotiff_path in enumerate(geotiff_paths, start=1):
        with rasterio.open(geotiff_path) as src:
            bounds = src.bounds
            crs = src.crs.to_dict() if src.crs is not None else None

        schema = {"geometry": "Polygon", "properties": {"filename": "str"}}
        geometry = box(*bounds)
        shapefile_path = output_folder / f"{geotiff_path.stem}.shp"

        with fiona.open(
            shapefile_path,
            "w",
            driver="ESRI Shapefile",
            crs=crs,
            schema=schema,
        ) as layer:
            layer.write(
                {
                    "geometry": mapping(geometry),
                    "properties": {"filename": geotiff_path.name},
                }
            )

        written_files.append(shapefile_path)
        if progress_callback is not None:
            progress_callback(index, len(geotiff_paths), geotiff_path.name)

    return written_files


def merge_csv_files(
    csv_path1: str | Path,
    csv_path2: str | Path,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([pd.read_csv(csv_path1), pd.read_csv(csv_path2)], ignore_index=True)
    merged.to_csv(output_path, index=False)
    return output_path


def save_polygon_shapefile_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    geometry_column: str = "mask_points_gis",
    reference_crs: str = "EPSG:26916",
) -> Path:
    df = pd.read_csv(csv_path)
    geometry_column = _select_geometry_column(df, geometry_column)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        polygon = _parse_polygon(row[geometry_column])
        if polygon.exterior.is_empty:
            continue
        rows.append(
            {
                "geometry": polygon,
                "latitude": float(row.get("latitude", 0.0)),
                "longitude": float(row.get("longitude", 0.0)),
                "class": int(row.get("class", 0)),
                "confidence": float(row.get("confidence", 0.0)),
                "tile_name": str(row.get("tile_name", ""))[:80],
                "area_geo": float(row.get("area_geo", polygon.area)),
            }
        )

    output_dir = Path(output_dir)
    shapefile_path = output_dir / "polygon_shapes.shp"
    _write_features(shapefile_path, "Polygon", reference_crs, rows)
    return shapefile_path


save_polygon_shapefile_from_mask = save_polygon_shapefile_from_csv


def save_bbox_shapefile_from_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    reference_crs: str = "EPSG:26916",
    x1_col: str = "bbox_lon_x1",
    y1_col: str = "bbox_lat_y1",
    x2_col: str = "bbox_lon_x2",
    y2_col: str = "bbox_lat_y2",
) -> Path:
    df = pd.read_csv(csv_path)

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        minx = float(row[x1_col])
        miny = float(row[y1_col])
        maxx = float(row[x2_col])
        maxy = float(row[y2_col])
        polygon = box(min(minx, maxx), min(miny, maxy), max(minx, maxx), max(miny, maxy))
        rows.append(
            {
                "geometry": polygon,
                "latitude": float(row.get("latitude", polygon.centroid.y)),
                "longitude": float(row.get("longitude", polygon.centroid.x)),
                "class": int(row.get("class", 0)),
                "confidence": float(row.get("confidence", 0.0)),
                "tile_name": str(row.get("tile_name", ""))[:80],
                "width_geo": float(abs(maxx - minx)),
                "height_geo": float(abs(maxy - miny)),
                "area_geo": float(row.get("area_geo", polygon.area)),
            }
        )

    output_dir = Path(output_dir)
    shapefile_path = output_dir / "bbox_shapes.shp"
    _write_features(shapefile_path, "Polygon", reference_crs, rows)
    return shapefile_path


def merge_overlapping_polygons_with_largest_area_attr(
    shapefile_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:26916",
) -> Path:
    df, crs = _read_vector_rows(shapefile_path)
    if df.empty:
        return _write_features(Path(output_path), "Polygon", target_crs, [])

    projected_df, projected_crs = _ensure_projected(df, crs, target_crs=target_crs)
    polygons = projected_df["geometry"].tolist()
    tree = STRtree(polygons)
    visited: set[int] = set()
    rows: list[dict[str, Any]] = []

    for index, polygon in enumerate(polygons):
        if index in visited:
            continue
        stack = [index]
        group: set[int] = set()
        while stack:
            current = stack.pop()
            if current in group:
                continue
            group.add(current)
            for candidate_index in _query_indices(tree, polygons, polygons[current]):
                if candidate_index not in group and polygons[current].intersects(polygons[candidate_index]):
                    stack.append(candidate_index)

        visited.update(group)
        merged = unary_union([polygons[group_index] for group_index in group])
        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda geom: geom.area)
        area_series = projected_df.iloc[list(group)]["geometry"].apply(lambda geom: geom.area)
        best_row = projected_df.iloc[list(group)].loc[area_series.idxmax()].copy()
        best_row["geometry"] = merged
        best_row["area_gis"] = float(merged.area)
        rows.append(best_row.to_dict())

    return _write_features(Path(output_path), "Polygon", projected_crs, rows)


def process_shapefile_with_bbox_dimensions(
    shapefile_path: str | Path,
    output_path: str | Path,
    target_crs: str = "EPSG:26916",
) -> Path:
    df, crs = _read_vector_rows(shapefile_path)
    projected_df, projected_crs = _ensure_projected(df, crs, target_crs=target_crs)
    if projected_df.empty:
        return _write_features(Path(output_path), "Polygon", projected_crs, [])

    lengths: list[float] = []
    widths: list[float] = []
    areas: list[float] = []
    for geometry in projected_df["geometry"]:
        rect = _safe_minimum_rotated_rectangle(geometry)
        coords = list(rect.exterior.coords)
        edge1 = math.dist(coords[0], coords[1]) * M_TO_FT
        edge2 = math.dist(coords[1], coords[2]) * M_TO_FT
        lengths.append(max(edge1, edge2))
        widths.append(min(edge1, edge2))
        areas.append(float(geometry.area) * SQ_M_TO_SQ_FT)

    output_df = projected_df.copy()
    output_df["area_gis"] = areas
    output_df["length"] = lengths
    output_df["width"] = widths
    return _write_features(Path(output_path), "Polygon", projected_crs, output_df.to_dict(orient="records"))


def filter_shapefile_by_length_width_area_proximity(
    shapefile_path: str | Path,
    output_path: str | Path,
    *,
    length_min: float = 370.0,
    width_min: float = 39.0,
    width_max: float = 100.0,
    area_min: float = 15000.0,
    area_max: float = 60000.0,
    proximity_m: float = 50.0,
    target_crs: str = "EPSG:26916",
    always_keep_class_zero: bool = True,
) -> Path:
    df, crs = _read_vector_rows(shapefile_path)
    projected_df, projected_crs = _ensure_projected(df, crs, target_crs=target_crs)
    if projected_df.empty:
        return _write_features(Path(output_path), "Polygon", projected_crs, [])

    projected_df = _ensure_dimension_columns(projected_df)
    area_ok = projected_df["area_gis"].between(area_min, area_max)
    width_ok = projected_df["width"].between(width_min, width_max)
    length_ok = projected_df["length"] >= length_min
    main_mask = ((length_ok) | (width_ok)) & area_ok

    class0_mask = pd.Series(False, index=projected_df.index)
    if always_keep_class_zero and "class" in projected_df.columns:
        class0_mask = pd.to_numeric(projected_df["class"], errors="coerce").fillna(1).eq(0)

    filtered = projected_df[main_mask | class0_mask].copy().reset_index(drop=True)
    if filtered.empty:
        return _write_features(Path(output_path), "Polygon", projected_crs, [])

    centroids = filtered["geometry"].apply(lambda geom: geom.centroid)
    coords = np.array([[point.x, point.y] for point in centroids], dtype=float)
    keep_indices: set[int] = set()
    class0_after_filter = pd.Series(False, index=filtered.index)
    if always_keep_class_zero and "class" in filtered.columns:
        class0_after_filter = pd.to_numeric(filtered["class"], errors="coerce").fillna(1).eq(0)
        keep_indices.update(filtered.index[class0_after_filter].tolist())

    for index, point in enumerate(centroids):
        if index in keep_indices:
            continue
        distances = np.linalg.norm(coords - np.array([point.x, point.y]), axis=1)
        has_neighbor = bool(np.any((distances <= proximity_m) & (distances > 0)))
        if has_neighbor:
            keep_indices.add(index)

    result = filtered.loc[sorted(keep_indices)].copy()
    return _write_features(Path(output_path), "Polygon", projected_crs, result.to_dict(orient="records"))


def identify_poultry_farms_from_shapefile(
    shapefile_path: str | Path,
    output_path: str | Path,
    *,
    proximity_m: float = 250.0,
    min_samples: int = 2,
    target_crs: str = "EPSG:26916",
) -> Path:
    """
    Identifies active poultry farms by clustering individual house detections.
    
    This function uses the DBSCAN (Density-Based Spatial Clustering of 
    Applications with Noise) algorithm to group detections that are within 
    a specific proximity of each other. Individual houses that do not meet 
    the minimum cluster size (min_samples) are treated as noise/outliers.
    
    Args:
        shapefile_path: Input shapefile containing poultry house polygons.
        output_path: Destination path for the farm centroid shapefile (EPSG:4326).
        proximity_m: Maximum distance (meters) between houses in the same farm.
        min_samples: Minimum number of houses to constitute a 'farm'.
    """
    from sklearn.cluster import DBSCAN

    # 1. Read source vector data and project to a metric CRS for accurate distance calculation
    df, crs = _read_vector_rows(shapefile_path)
    projected_df, projected_crs = _ensure_projected(df, crs, target_crs=target_crs)
    if projected_df.empty:
        return _write_features(Path(output_path), "Point", "EPSG:4326", [])

    # 2. Extract centroids for clustering
    centroids = projected_df["geometry"].apply(lambda geom: geom.centroid)
    coords = np.array([[point.x, point.y] for point in centroids], dtype=float)
    
    # 3. Execute DBSCAN clustering
    # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    labels = DBSCAN(eps=proximity_m, min_samples=min_samples, metric="euclidean").fit(coords).labels_
    projected_df["farm_id"] = labels
    
    # Filter out noise (-1)
    grouped = projected_df[projected_df["farm_id"] != -1].copy()
    if grouped.empty:
        return _write_features(Path(output_path), "Point", "EPSG:4326", [])

    # 4. Compute farm-level metrics and centroids
    transformer = pyproj.Transformer.from_crs(projected_crs, "EPSG:4326", always_xy=True)
    rows: list[dict[str, Any]] = []
    for farm_id, farm_df in grouped.groupby("farm_id"):
        # Mean center of all houses in the farm
        center_x = float(farm_df["geometry"].apply(lambda geom: geom.centroid.x).mean())
        center_y = float(farm_df["geometry"].apply(lambda geom: geom.centroid.y).mean())
        
        # Project back to geographic coordinates (WGS84) for easy visualization
        longitude, latitude = transformer.transform(center_x, center_y)
        rows.append(
            {
                "geometry": Point(longitude, latitude),
                "farm_id": int(farm_id),
                "houses": int(len(farm_df)),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        )

    # 5. Write the resulting farm centroids to a new shapefile
    return _write_features(Path(output_path), "Point", "EPSG:4326", rows)


def _read_vector_rows(shapefile_path: str | Path) -> tuple[pd.DataFrame, pyproj.CRS | None]:
    rows: list[dict[str, Any]] = []
    crs: pyproj.CRS | None = None
    with fiona.open(shapefile_path, "r") as src:
        if src.crs_wkt:
            crs = pyproj.CRS.from_wkt(src.crs_wkt)
        elif src.crs:
            crs = pyproj.CRS(src.crs)
        for feature in src:
            geometry = feature.get("geometry")
            if geometry is None:
                continue
            row = dict(feature["properties"])
            row["geometry"] = shape(geometry)
            rows.append(row)
    return pd.DataFrame(rows), crs


def _ensure_projected(
    df: pd.DataFrame,
    crs: pyproj.CRS | None,
    *,
    target_crs: str = "EPSG:26916",
) -> tuple[pd.DataFrame, pyproj.CRS]:
    target = pyproj.CRS.from_user_input(target_crs)
    if df.empty:
        return df.copy(), target
    if crs is None:
        return df.copy(), target
    if crs.is_geographic:
        transformer = pyproj.Transformer.from_crs(crs, target, always_xy=True)
        projected = df.copy()
        projected["geometry"] = projected["geometry"].apply(
            lambda geom: shapely_transform(transformer.transform, geom),
        )
        return projected, target
    return df.copy(), crs


def _ensure_dimension_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "area_gis" not in output.columns:
        output["area_gis"] = output["geometry"].apply(lambda geom: float(geom.area) * SQ_M_TO_SQ_FT)
    if "length" not in output.columns or "width" not in output.columns:
        lengths: list[float] = []
        widths: list[float] = []
        for geometry in output["geometry"]:
            rect = _safe_minimum_rotated_rectangle(geometry)
            coords = list(rect.exterior.coords)
            edge1 = math.dist(coords[0], coords[1]) * M_TO_FT
            edge2 = math.dist(coords[1], coords[2]) * M_TO_FT
            lengths.append(max(edge1, edge2))
            widths.append(min(edge1, edge2))
        output["length"] = lengths
        output["width"] = widths
    return output


def _query_indices(tree: STRtree, geometries: list[Polygon], polygon: Polygon) -> list[int]:
    result = tree.query(polygon)
    if len(result) == 0:
        return []
    if hasattr(result[0], "geom_type"):
        lookup = {geom.wkt: index for index, geom in enumerate(geometries)}
        return [lookup[item.wkt] for item in result if item.wkt in lookup]
    return [int(index) for index in result]


def _write_features(
    output_path: Path,
    geometry_type: str,
    crs: str | pyproj.CRS,
    rows: Iterable[dict[str, Any]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    property_columns = [
        column
        for column in row_list[0].keys() if column != "geometry"
    ] if row_list else []
    field_map = _build_field_name_map(property_columns)
    schema = {
        "geometry": geometry_type,
        "properties": {
            field_map[column]: _infer_fiona_type([row.get(column) for row in row_list])
            for column in property_columns
        },
    }
    crs_obj = pyproj.CRS.from_user_input(crs)

    with fiona.open(
        output_path,
        mode="w",
        driver="ESRI Shapefile",
        crs_wkt=crs_obj.to_wkt(),
        schema=schema,
    ) as layer:
        for row in row_list:
            geometry = row.get("geometry")
            if geometry is None:
                continue
            properties = {
                field_map[column]: _coerce_property_value(
                    row.get(column),
                    schema["properties"][field_map[column]],
                )
                for column in property_columns
            }
            layer.write({"geometry": mapping(geometry), "properties": properties})
    return output_path


def _build_field_name_map(columns: list[str]) -> dict[str, str]:
    used: set[str] = set()
    mapping_out: dict[str, str] = {}
    for column in columns:
        candidate = column[:10]
        if candidate not in used:
            mapping_out[column] = candidate
            used.add(candidate)
            continue
        for index in range(1, 100):
            suffix = str(index)
            alt = f"{candidate[: 10 - len(suffix)]}{suffix}"
            if alt not in used:
                mapping_out[column] = alt
                used.add(alt)
                break
    return mapping_out


def _infer_fiona_type(values: list[Any]) -> str:
    filtered = [value for value in values if value is not None and not (isinstance(value, float) and np.isnan(value))]
    if not filtered:
        return "str:120"
    if all(isinstance(value, (bool, np.bool_)) for value in filtered):
        return "int"
    if all(isinstance(value, (int, np.integer)) for value in filtered):
        return "int"
    if all(isinstance(value, (int, float, np.integer, np.floating)) for value in filtered):
        return "float"
    return "str:120"


def _coerce_property_value(value: Any, fiona_type: str) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        if fiona_type.startswith("str"):
            return ""
        if fiona_type == "int":
            return 0
        return 0.0
    if fiona_type == "int":
        return int(value)
    if fiona_type == "float":
        return float(value)
    return str(value)


def _safe_minimum_rotated_rectangle(polygon: Polygon) -> Polygon:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*oriented_envelope.*",
            category=RuntimeWarning,
        )
        rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return polygon.envelope
    return rectangle
