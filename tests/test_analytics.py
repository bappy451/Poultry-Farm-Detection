from __future__ import annotations

import math

import pandas as pd

from poultry_geo.analytics import (
    aggregate_detections_by_county,
    apply_nass_farm_correction,
    bootstrap_adjusted_runoff,
    compute_county_runoff_screening,
)


def test_aggregate_detections_by_county_counts_houses_and_area() -> None:
    detections = pd.DataFrame(
        [
            {"county": "A", "area_geo": 10.0},
            {"county": "A", "area_geo": 12.5},
            {"county": "B", "area_geo": 5.0},
        ]
    )

    summary = aggregate_detections_by_county(detections)

    assert summary.loc[summary["county"] == "A", "detected_houses"].item() == 2
    assert math.isclose(summary.loc[summary["county"] == "A", "roof_area_m2"].item(), 22.5)
    assert summary.loc[summary["county"] == "B", "detected_farms"].item() == 1


def test_apply_nass_farm_correction_handles_zero_nass() -> None:
    county_summary = pd.DataFrame(
        [
            {"county": "A", "nass_farms": 10, "detected_farms": 5},
            {"county": "B", "nass_farms": 0, "detected_farms": 3},
        ]
    )

    corrected = apply_nass_farm_correction(county_summary)

    assert math.isclose(corrected.loc[corrected["county"] == "A", "nass_correction_factor"].item(), 2.0)
    assert math.isclose(corrected.loc[corrected["county"] == "B", "nass_correction_factor"].item(), 0.0)


def test_compute_county_runoff_screening_uses_mm_to_m3_formula() -> None:
    county_summary = pd.DataFrame(
        [
            {
                "county": "A",
                "roof_area_m2": 1000.0,
                "detected_farms": 10,
                "nass_farms": 5,
                "nass_correction_factor": 0.5,
            }
        ]
    )
    precipitation = pd.DataFrame([{"county": "A", "jan_mm": 100.0, "feb_mm": 50.0}])

    runoff = compute_county_runoff_screening(
        county_summary,
        precipitation,
        runoff_coefficient=0.8,
    )

    assert math.isclose(runoff["jan_mm_runoff_m3"].item(), 80.0)
    assert math.isclose(runoff["feb_mm_runoff_m3"].item(), 40.0)
    assert math.isclose(runoff["annual_runoff_m3"].item(), 120.0)
    assert math.isclose(runoff["nass_adjusted_annual_runoff_m3"].item(), 60.0)


def test_bootstrap_adjusted_runoff_returns_interval() -> None:
    runoff = pd.DataFrame([{"nass_adjusted_annual_runoff_m3": 100.0}])

    summary = bootstrap_adjusted_runoff(
        runoff,
        adjustment_factors=[0.8, 1.0, 1.2],
        n_bootstrap=200,
        seed=7,
    )

    assert 0.8 <= summary.mean_adjustment_factor <= 1.2
    assert summary.lower_95_m3 <= summary.bootstrap_mean_volume_m3 <= summary.upper_95_m3
    assert summary.n_bootstrap == 200

