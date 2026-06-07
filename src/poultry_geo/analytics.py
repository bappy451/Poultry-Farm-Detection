from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BootstrapSummary:
    mean_adjustment_factor: float
    bootstrap_mean_volume_m3: float
    lower_95_m3: float
    upper_95_m3: float
    n_bootstrap: int


def aggregate_detections_by_county(
    detections_df: pd.DataFrame,
    county_column: str = "county",
    roof_area_column: str = "area_geo",
) -> pd.DataFrame:
    required = {county_column, roof_area_column}
    if not required.issubset(detections_df.columns):
        missing = ", ".join(sorted(required - set(detections_df.columns)))
        raise ValueError(f"Detection dataframe missing required columns: {missing}")

    grouped = detections_df.groupby(county_column, dropna=False)
    summary = grouped.agg(
        detected_houses=(county_column, "size"),
        roof_area_m2=(roof_area_column, "sum"),
    )
    summary = summary.reset_index()
    summary["detected_farms"] = summary["detected_houses"]
    return summary


def compute_nass_correction_factor(nass_farms: float, detected_farms: float) -> float:
    if pd.isna(nass_farms) or pd.isna(detected_farms):
        return float("nan")
    if nass_farms <= 0:
        return 0.0
    if detected_farms <= 0:
        return 0.0
    return float(nass_farms) / float(detected_farms)


def apply_nass_farm_correction(
    county_summary_df: pd.DataFrame,
    nass_farms_column: str = "nass_farms",
    detected_farms_column: str = "detected_farms",
) -> pd.DataFrame:
    required = {nass_farms_column, detected_farms_column}
    if not required.issubset(county_summary_df.columns):
        missing = ", ".join(sorted(required - set(county_summary_df.columns)))
        raise ValueError(f"County summary missing required columns: {missing}")

    corrected = county_summary_df.copy()
    corrected["nass_correction_factor"] = corrected.apply(
        lambda row: compute_nass_correction_factor(
            row[nass_farms_column],
            row[detected_farms_column],
        ),
        axis=1,
    )
    return corrected


def _auto_month_columns(precipitation_df: pd.DataFrame, county_column: str) -> list[str]:
    return [column for column in precipitation_df.columns if column != county_column]


def compute_county_runoff_screening(
    county_summary_df: pd.DataFrame,
    precipitation_df: pd.DataFrame,
    county_column: str = "county",
    roof_area_column: str = "roof_area_m2",
    runoff_coefficient: float = 0.8,
    month_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if county_column not in county_summary_df.columns or roof_area_column not in county_summary_df.columns:
        raise ValueError(f"County summary must contain '{county_column}' and '{roof_area_column}'.")
    if county_column not in precipitation_df.columns:
        raise ValueError(f"Precipitation dataframe must contain '{county_column}'.")

    month_columns = list(month_columns) if month_columns is not None else _auto_month_columns(
        precipitation_df,
        county_column,
    )
    merged = county_summary_df.merge(precipitation_df, on=county_column, how="left", validate="1:1")
    if merged[month_columns].isna().any().any():
        raise ValueError("Missing precipitation values after joining county summary to precipitation dataframe.")

    for month_column in month_columns:
        merged[f"{month_column}_runoff_m3"] = (
            merged[month_column].astype(float) / 1000.0
            * merged[roof_area_column].astype(float)
            * float(runoff_coefficient)
        )

    runoff_columns = [f"{month_column}_runoff_m3" for month_column in month_columns]
    merged["annual_runoff_m3"] = merged[runoff_columns].sum(axis=1)
    if "nass_correction_factor" in merged.columns:
        merged["nass_adjusted_annual_runoff_m3"] = (
            merged["annual_runoff_m3"] * merged["nass_correction_factor"]
        )
    return merged


def bootstrap_adjusted_runoff(
    county_runoff_df: pd.DataFrame,
    adjustment_factors: Sequence[float],
    annual_volume_column: str = "nass_adjusted_annual_runoff_m3",
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> BootstrapSummary:
    if annual_volume_column not in county_runoff_df.columns:
        raise ValueError(f"County runoff dataframe missing '{annual_volume_column}'.")
    if not adjustment_factors:
        raise ValueError("adjustment_factors must contain at least one value.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    rng = np.random.default_rng(seed)
    factors = np.asarray(adjustment_factors, dtype=float)
    base_volume = float(county_runoff_df[annual_volume_column].sum())
    sampled = rng.choice(factors, size=(n_bootstrap, len(factors)), replace=True)
    mean_factors = sampled.mean(axis=1)
    bootstrap_volumes = mean_factors * base_volume
    lower_95, upper_95 = np.percentile(bootstrap_volumes, [2.5, 97.5])
    return BootstrapSummary(
        mean_adjustment_factor=float(mean_factors.mean()),
        bootstrap_mean_volume_m3=float(bootstrap_volumes.mean()),
        lower_95_m3=float(lower_95),
        upper_95_m3=float(upper_95),
        n_bootstrap=n_bootstrap,
    )


def load_adjustment_factors(
    csv_path: str | Path,
    column: str = "adjustment_factor",
) -> list[float]:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Adjustment factor CSV missing column '{column}'.")
    return df[column].dropna().astype(float).tolist()

