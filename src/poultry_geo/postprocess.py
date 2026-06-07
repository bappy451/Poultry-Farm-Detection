from __future__ import annotations

import ast
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from sklearn.cluster import DBSCAN


def load_detection_csv(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _select_geometry_column(df: pd.DataFrame, geometry_column: str | None = None) -> str:
    if geometry_column:
        if geometry_column not in df.columns:
            raise ValueError(f"Geometry column '{geometry_column}' not found in dataframe.")
        return geometry_column
    for candidate in ("mask_points_gis", "rect_mask_points"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No supported geometry column found. Expected mask_points_gis or rect_mask_points.")


def _parse_polygon(raw_value: str | list[list[float]]) -> Polygon:
    if isinstance(raw_value, list):
        coords = raw_value
    else:
        try:
            coords = json.loads(raw_value)
        except json.JSONDecodeError:
            coords = ast.literal_eval(raw_value)
    polygon = Polygon(coords)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Invalid polygon geometry.")
    return polygon


def centroid_clustering_suppression(
    df: pd.DataFrame,
    distance_threshold: float = 10.0,
    geometry_column: str | None = None,
) -> pd.DataFrame:
    geometry_column = _select_geometry_column(df, geometry_column)
    working = df.copy()
    working["polygon"] = working[geometry_column].apply(_parse_polygon)
    working["centroid"] = working["polygon"].apply(lambda poly: (poly.centroid.x, poly.centroid.y))

    centroids = np.array(working["centroid"].tolist(), dtype=float)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1, metric="euclidean").fit(centroids)
    working["cluster"] = clustering.labels_

    keep_indices = []
    for _, group in working.groupby("cluster"):
        keep_indices.append(group["confidence"].idxmax())

    deduped = working.loc[keep_indices].copy()
    return deduped.drop(columns=["polygon", "centroid", "cluster"]).reset_index(drop=True)


def _compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    if poly1.is_empty or poly2.is_empty or not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union else 0.0


def mask_nms_by_area(
    df: pd.DataFrame,
    iou_threshold: float = 0.5,
    geometry_column: str | None = None,
) -> pd.DataFrame:
    geometry_column = _select_geometry_column(df, geometry_column)
    working = df.copy()
    working["polygon"] = working[geometry_column].apply(_parse_polygon)
    if "area_geo" not in working.columns:
        working["area_geo"] = working["polygon"].apply(lambda poly: poly.area)

    working = working.sort_values("area_geo", ascending=False).reset_index(drop=True)
    kept_rows: list[int] = []
    suppressed: set[int] = set()

    for index, row in working.iterrows():
        if index in suppressed:
            continue
        kept_rows.append(index)
        polygon = row["polygon"]
        for candidate_index in range(index + 1, len(working)):
            if candidate_index in suppressed:
                continue
            candidate_polygon = working.loc[candidate_index, "polygon"]
            if _compute_iou(polygon, candidate_polygon) > iou_threshold:
                suppressed.add(candidate_index)

    filtered = working.loc[kept_rows].drop(columns=["polygon"])
    return filtered.reset_index(drop=True)


def detect_poultry_farms(
    df: pd.DataFrame,
    eps_house: float = 50.0,
    min_samples: int = 2,
    eps_merge: float = 100.0,
) -> pd.DataFrame:
    required = {"latitude", "longitude"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Input dataframe missing required columns: {missing}")

    working = df.copy()
    coords = working[["longitude", "latitude"]].to_numpy(dtype=float)
    house_clusters = DBSCAN(eps=eps_house, min_samples=min_samples).fit(coords)
    working["initial_cluster"] = house_clusters.labels_

    clustered = working[working["initial_cluster"] != -1].copy()
    if clustered.empty:
        return pd.DataFrame(columns=["merged_farm_id", "longitude", "latitude", "num_houses"])

    centroids = (
        clustered.groupby("initial_cluster")[["longitude", "latitude"]]
        .mean()
        .reset_index()
    )
    centroid_coords = centroids[["longitude", "latitude"]].to_numpy(dtype=float)
    merge_clusters = DBSCAN(eps=eps_merge, min_samples=1).fit(centroid_coords)
    centroids["merged_farm_id"] = merge_clusters.labels_

    cluster_map = dict(zip(centroids["initial_cluster"], centroids["merged_farm_id"]))
    clustered["merged_farm_id"] = clustered["initial_cluster"].map(cluster_map)

    farms = (
        clustered.groupby("merged_farm_id")[["longitude", "latitude"]]
        .mean()
        .reset_index()
    )
    farms["num_houses"] = clustered.groupby("merged_farm_id").size().values
    return farms


def maximum_likelihood_suppression(
    results: Any,
    conf_threshold: float = 0.7,
    iou_threshold: float = 0.5,
) -> Any:
    try:
        import torch
        from torchvision.ops import nms
    except ImportError as exc:
        raise RuntimeError(
            "maximum_likelihood_suppression requires torch and torchvision. Install requirements-ml.txt.",
        ) from exc

    if results.masks is None or results.boxes is None or results.boxes.data is None:
        return results

    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    keep = scores >= conf_threshold
    if keep.sum() == 0:
        results.boxes.data = results.boxes.data[[]]
        if results.masks is not None and results.masks.data is not None:
            results.masks.data = results.masks.data[[]]
        return results

    keep_indices = nms(boxes[keep], scores[keep], iou_threshold)
    indices_to_keep = torch.nonzero(keep).squeeze(1)[keep_indices]
    results.boxes.data = results.boxes.data[indices_to_keep]
    if results.masks is not None and results.masks.data is not None:
        results.masks.data = results.masks.data[indices_to_keep]
    return results


def edges_from_polygon(poly: Polygon) -> list[LineString]:
    coords = list(poly.exterior.coords)
    return [LineString([coords[index], coords[index + 1]]) for index in range(len(coords) - 1)]


def polygons_edges_are_close(poly1: Polygon, poly2: Polygon, threshold: float = 2.0) -> bool:
    for edge1 in edges_from_polygon(poly1):
        for edge2 in edges_from_polygon(poly2):
            if edge1.distance(edge2) < threshold:
                return True
    return False


def furthest_four_points(all_points: list[list[float]]) -> list[list[float]]:
    poly = Polygon(all_points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or not poly.is_valid:
        xs, ys = zip(*all_points)
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        return [[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny]]
    minimum_rectangle = _safe_minimum_rotated_rectangle(poly)
    return [list(point) for point in list(minimum_rectangle.exterior.coords)[:-1]]


def make_rotated_rectangles(
    df_or_csv: pd.DataFrame | str | Path,
    geometry_column: str = "mask_points_gis",
) -> pd.DataFrame:
    df = load_detection_csv(df_or_csv) if isinstance(df_or_csv, (str, Path)) else df_or_csv.copy()

    rect_points: list[str] = []
    rect_bboxes: list[str] = []
    rect_areas: list[float] = []
    rect_x_centers: list[float] = []
    rect_y_centers: list[float] = []

    for raw_value in df[geometry_column]:
        polygon = _parse_polygon(raw_value)
        rect = _safe_minimum_rotated_rectangle(polygon)
        coords = [list(point) for point in list(rect.exterior.coords)[:-1]]
        minx, miny, maxx, maxy = rect.bounds
        rect_points.append(json.dumps(coords))
        rect_bboxes.append(str((minx, miny, maxx, maxy)))
        rect_areas.append(float(rect.area))
        rect_x_centers.append(float(rect.centroid.x))
        rect_y_centers.append(float(rect.centroid.y))

    output = df.copy()
    output["rect_mask_points"] = rect_points
    output["rect_bbox"] = rect_bboxes
    output["rect_area"] = rect_areas
    output["rect_x_center"] = rect_x_centers
    output["rect_y_center"] = rect_y_centers
    return output


def merge_adjacent_polygons(
    df_or_csv: pd.DataFrame | str | Path,
    eps: float = 0.0,
    geometry_column: str = "mask_points_gis",
) -> pd.DataFrame:
    df = load_detection_csv(df_or_csv) if isinstance(df_or_csv, (str, Path)) else df_or_csv.copy()
    df["polygon"] = df[geometry_column].apply(_parse_polygon)
    polygons = df["polygon"].tolist()
    buffered = [polygon.buffer(eps) if eps else polygon for polygon in polygons]

    adjacency = {index: set() for index in range(len(polygons))}
    tree = STRtree(buffered)
    for index, buffered_polygon in enumerate(buffered):
        for candidate_index in _get_candidates_from_tree(tree, buffered, buffered_polygon, 0.0):
            if candidate_index <= index:
                continue
            if buffered_polygon.intersects(buffered[candidate_index]):
                adjacency[index].add(candidate_index)
                adjacency[candidate_index].add(index)

    merged_rows: list[pd.Series] = []
    for group in _connected_components(adjacency):
        group_indices = list(group)
        merged_geometry = unary_union([polygons[index] for index in group_indices])
        if merged_geometry.is_empty:
            continue
        if merged_geometry.geom_type == "MultiPolygon":
            merged_geometry = max(merged_geometry.geoms, key=lambda polygon: polygon.area)
        elif merged_geometry.geom_type != "Polygon":
            merged_geometry = merged_geometry.convex_hull

        group_rows = df.loc[group_indices]
        row = group_rows.loc[group_rows["confidence"].idxmax()].copy()
        minx, miny, maxx, maxy = merged_geometry.bounds
        row["mask_points_gis"] = json.dumps([list(point) for point in merged_geometry.exterior.coords])
        row["bbox_lon_x1"], row["bbox_lat_y1"] = minx, miny
        row["bbox_lon_x2"], row["bbox_lat_y2"] = maxx, maxy
        row["bbox_width_geo"] = maxx - minx
        row["bbox_height_geo"] = maxy - miny
        row["area_geo"] = float(merged_geometry.area)
        row["longitude"] = float(merged_geometry.centroid.x)
        row["latitude"] = float(merged_geometry.centroid.y)
        merged_rows.append(row)

    output = pd.DataFrame(merged_rows)
    if "polygon" in output.columns:
        output = output.drop(columns=["polygon"])
    return output.reset_index(drop=True)


def _get_candidates_from_tree(tree: STRtree, polygons: list[Polygon], query_polygon: Polygon, distance: float) -> list[int]:
    result = tree.query(query_polygon.buffer(distance))
    if len(result) == 0:
        return []
    if hasattr(result[0], "wkt"):
        lookup = {polygon.wkt: index for index, polygon in enumerate(polygons)}
        return [lookup.get(poly.wkt, -1) for poly in result if lookup.get(poly.wkt, -1) >= 0]
    return [int(index) for index in result]


def merge_adjacent_masks(
    df_or_csv: pd.DataFrame | str | Path,
    distance_threshold: float = 2.0,
    geometry_column: str = "rect_mask_points",
) -> pd.DataFrame:
    df = load_detection_csv(df_or_csv) if isinstance(df_or_csv, (str, Path)) else df_or_csv.copy()
    df["polygon"] = df[geometry_column].apply(lambda value: Polygon(ast.literal_eval(value)))
    polygons = df["polygon"].tolist()

    adjacency_list = {index: set() for index in range(len(polygons))}
    center_lines = [LineString(_middle_short_edge_points(poly)) for poly in polygons]

    for left_index in range(len(center_lines)):
        for right_index in range(left_index + 1, len(center_lines)):
            if center_lines[left_index].distance(center_lines[right_index]) <= distance_threshold:
                adjacency_list[left_index].add(right_index)
                adjacency_list[right_index].add(left_index)

    groups = _connected_components(adjacency_list)
    merged_rows: list[pd.Series] = []
    for group_indices in groups:
        group_polygons = [polygons[index] for index in group_indices]
        merged_geometry = unary_union(group_polygons)
        geometries = list(merged_geometry.geoms) if merged_geometry.geom_type == "MultiPolygon" else [merged_geometry]
        for geometry in geometries:
            if geometry.is_empty or not geometry.is_valid:
                continue
            group_rows = df.loc[list(group_indices)]
            row = group_rows.loc[group_rows["confidence"].idxmax()].copy()
            minx, miny, maxx, maxy = geometry.bounds
            row["mask_points_gis"] = json.dumps(list(geometry.exterior.coords))
            row["bbox_lon_x1"], row["bbox_lat_y1"] = minx, miny
            row["bbox_lon_x2"], row["bbox_lat_y2"] = maxx, maxy
            row["bbox_width_geo"] = maxx - minx
            row["bbox_height_geo"] = maxy - miny
            row["rect_area"] = geometry.area
            centroid = geometry.centroid
            row["rect_x_center"], row["rect_y_center"] = centroid.x, centroid.y
            row["latitude"], row["longitude"] = centroid.y, centroid.x
            row["bbox_width"], row["bbox_height"] = maxx - minx, maxy - miny
            merged_rows.append(row)

    output = pd.DataFrame(merged_rows)
    if "polygon" in output.columns:
        output = output.drop(columns=["polygon"])
    return output.reset_index(drop=True)


def merge_rect_masks(
    df_or_csv: pd.DataFrame | str | Path,
    geometry_column: str = "rect_mask_points",
) -> pd.DataFrame:
    df = load_detection_csv(df_or_csv) if isinstance(df_or_csv, (str, Path)) else df_or_csv.copy()
    polygons = [Polygon(ast.literal_eval(value)) for value in df[geometry_column]]
    areas = [polygon.area for polygon in polygons]

    overlap_graph = {index: set() for index in range(len(polygons))}
    for left_index in range(len(polygons)):
        for right_index in range(left_index + 1, len(polygons)):
            if polygons[left_index].intersects(polygons[right_index]) and polygons[left_index].intersection(polygons[right_index]).area > 0:
                overlap_graph[left_index].add(right_index)
                overlap_graph[right_index].add(left_index)

    keep_indices: set[int] = set()
    for group in _connected_components(overlap_graph):
        group_indices = list(group)
        largest_index = group_indices[int(np.argmax([areas[index] for index in group_indices]))]
        keep_indices.add(largest_index)

    return df.iloc[sorted(keep_indices)].reset_index(drop=True)


def merge_masks_by_edge_proximity(
    df_or_csv: pd.DataFrame | str | Path,
    edge_threshold: float = 2.0,
    fill_gap: bool = True,
    geometry_column: str = "rect_mask_points",
) -> pd.DataFrame:
    df0 = load_detection_csv(df_or_csv) if isinstance(df_or_csv, (str, Path)) else df_or_csv.copy()

    polygons: list[Polygon] = []
    rows: list[pd.Series] = []
    rect_points_list: list[list[list[float]]] = []
    for _, source_row in df0.iterrows():
        rect_points = json.loads(source_row[geometry_column])
        polygon = Polygon(rect_points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if isinstance(polygon, MultiPolygon):
            for sub_polygon in polygon.geoms:
                if sub_polygon.is_empty or not sub_polygon.is_valid:
                    continue
                polygons.append(sub_polygon)
                rows.append(source_row.copy())
                rect_points_list.append([list(point) for point in list(sub_polygon.exterior.coords)[:-1]])
        elif not polygon.is_empty and polygon.is_valid:
            polygons.append(polygon)
            rows.append(source_row.copy())
            rect_points_list.append(rect_points)

    df = pd.DataFrame(rows).reset_index(drop=True)
    tree = STRtree(polygons)
    adjacency = {index: set() for index in range(len(polygons))}
    for index, polygon in enumerate(polygons):
        for candidate_index in _get_candidates_from_tree(tree, polygons, polygon, edge_threshold):
            if candidate_index <= index:
                continue
            if polygons_edges_are_close(polygon, polygons[candidate_index], edge_threshold):
                adjacency[index].add(candidate_index)
                adjacency[candidate_index].add(index)

    merged_rows: list[pd.Series] = []
    for group in _connected_components(adjacency):
        group_indices = list(group)
        try:
            polygons_to_merge = [polygons[index] for index in group_indices]
            group_rect_points = [point for index in group_indices for point in rect_points_list[index]]
            if fill_gap:
                buffered = [polygon.buffer(edge_threshold) for polygon in polygons_to_merge]
                merged_polygon = unary_union(buffered).buffer(-edge_threshold)
            else:
                merged_polygon = unary_union(polygons_to_merge)
        except Exception:
            continue
        if merged_polygon.is_empty:
            continue
        geometries = list(merged_polygon.geoms) if merged_polygon.geom_type == "MultiPolygon" else [merged_polygon]
        for geometry in geometries:
            if geometry.is_empty or not geometry.is_valid:
                continue
            group_rows = df.loc[group_indices]
            row = group_rows.loc[group_rows["confidence"].idxmax()].copy()
            minx, miny, maxx, maxy = geometry.bounds
            mask_points = [list(point) for point in geometry.exterior.coords]
            merged_rect_points = furthest_four_points(group_rect_points)
            row["rect_mask_points"] = json.dumps(merged_rect_points)
            row["rect_bbox"] = str(tuple(tuple(point) for point in merged_rect_points))
            row["rect_area"] = Polygon(merged_rect_points).area
            row["mask_points_gis"] = json.dumps(mask_points)
            row["bbox_lon_x1"], row["bbox_lat_y1"] = minx, miny
            row["bbox_lon_x2"], row["bbox_lat_y2"] = maxx, maxy
            row["bbox_width_geo"] = maxx - minx
            row["bbox_height_geo"] = maxy - miny
            centroid = geometry.centroid
            row["rect_x_center"], row["rect_y_center"] = centroid.x, centroid.y
            row["latitude"], row["longitude"] = centroid.y, centroid.x
            row["bbox_width"], row["bbox_height"] = maxx - minx, maxy - miny
            merged_rows.append(row)
    return pd.DataFrame(merged_rows).reset_index(drop=True)


def _middle_short_edge_points(poly: Polygon) -> list[tuple[float, float]]:
    coords = list(poly.exterior.coords)
    edges = [(coords[index], coords[index + 1]) for index in range(len(coords) - 1)]
    lengths = [LineString([point1, point2]).length for point1, point2 in edges]
    short_edge_indices = np.argsort(lengths)[:2]
    midpoints = [LineString(edges[index]).interpolate(0.5, normalized=True) for index in short_edge_indices]
    return [(point.x, point.y) for point in midpoints]


def _connected_components(adjacency: dict[int, set[int]]) -> list[set[int]]:
    visited: set[int] = set()
    groups: list[set[int]] = []

    def dfs(index: int, group: set[int]) -> None:
        visited.add(index)
        group.add(index)
        for neighbor in adjacency[index]:
            if neighbor not in visited:
                dfs(neighbor, group)

    for index in adjacency:
        if index not in visited:
            group: set[int] = set()
            dfs(index, group)
            groups.append(group)
    return groups


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
