from __future__ import annotations

import json

import pandas as pd

from poultry_geo.postprocess import (
    centroid_clustering_suppression,
    detect_poultry_farms,
    make_rotated_rectangles,
    mask_nms_by_area,
    merge_adjacent_polygons,
)


def test_centroid_clustering_suppression_keeps_highest_confidence() -> None:
    df = pd.DataFrame(
        [
            {
                "confidence": 0.60,
                "mask_points_gis": json.dumps([[0, 0], [2, 0], [2, 1], [0, 1]]),
            },
            {
                "confidence": 0.95,
                "mask_points_gis": json.dumps([[0.5, 0], [2.5, 0], [2.5, 1], [0.5, 1]]),
            },
            {
                "confidence": 0.80,
                "mask_points_gis": json.dumps([[20, 20], [22, 20], [22, 21], [20, 21]]),
            },
        ]
    )

    filtered = centroid_clustering_suppression(df, distance_threshold=5.0)

    assert len(filtered) == 2
    assert 0.95 in filtered["confidence"].tolist()
    assert 0.60 not in filtered["confidence"].tolist()


def test_mask_nms_by_area_suppresses_overlapping_polygon() -> None:
    df = pd.DataFrame(
        [
            {
                "confidence": 0.9,
                "area_geo": 10.0,
                "mask_points_gis": json.dumps([[0, 0], [4, 0], [4, 4], [0, 4]]),
            },
            {
                "confidence": 0.8,
                "area_geo": 9.0,
                "mask_points_gis": json.dumps([[0.5, 0.5], [4.5, 0.5], [4.5, 4.5], [0.5, 4.5]]),
            },
            {
                "confidence": 0.7,
                "area_geo": 8.0,
                "mask_points_gis": json.dumps([[10, 10], [12, 10], [12, 12], [10, 12]]),
            },
        ]
    )

    filtered = mask_nms_by_area(df, iou_threshold=0.5)

    assert len(filtered) == 2


def test_detect_poultry_farms_groups_close_houses() -> None:
    df = pd.DataFrame(
        [
            {"longitude": 0.0, "latitude": 0.0},
            {"longitude": 10.0, "latitude": 0.0},
            {"longitude": 12.0, "latitude": 1.0},
            {"longitude": 300.0, "latitude": 300.0},
            {"longitude": 305.0, "latitude": 302.0},
        ]
    )

    farms = detect_poultry_farms(df, eps_house=20.0, min_samples=2, eps_merge=40.0)

    assert len(farms) == 2
    assert sorted(farms["num_houses"].tolist()) == [2, 3]


def test_make_rotated_rectangles_adds_rectangle_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "confidence": 0.9,
                "mask_points_gis": json.dumps([[0, 0], [4, 0], [4, 2], [0, 2]]),
            },
        ]
    )

    output = make_rotated_rectangles(df)

    assert "rect_mask_points" in output.columns
    assert "rect_area" in output.columns
    rect_points = json.loads(output.loc[0, "rect_mask_points"])
    assert len(rect_points) == 4
    assert output.loc[0, "rect_area"] > 0


def test_merge_adjacent_polygons_merges_overlapping_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "confidence": 0.70,
                "mask_points_gis": json.dumps([[0, 0], [3, 0], [3, 2], [0, 2]]),
                "bbox_lon_x1": 0.0,
                "bbox_lat_y1": 0.0,
                "bbox_lon_x2": 3.0,
                "bbox_lat_y2": 2.0,
            },
            {
                "confidence": 0.95,
                "mask_points_gis": json.dumps([[2, 0], [5, 0], [5, 2], [2, 2]]),
                "bbox_lon_x1": 2.0,
                "bbox_lat_y1": 0.0,
                "bbox_lon_x2": 5.0,
                "bbox_lat_y2": 2.0,
            },
        ]
    )

    output = merge_adjacent_polygons(df)

    assert len(output) == 1
    assert output.loc[0, "confidence"] == 0.95
    assert output.loc[0, "area_geo"] > 0
