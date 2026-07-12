from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import AMENITY_NAMES, MODEL_FEATURES, REFERENCE_YEAR

STRING_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status",
]
NUMERIC_COLUMNS = [
    "ID", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    "Year_Built", "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals",
]


def _clean_string(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned else "Unknown"


def parse_amenities(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        item.strip().title()
        for item in str(value).split(",")
        if item.strip()
    }


def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw records and enforce domain-valid values.

    The supplied dataset is complete, but this routine is intentionally defensive so
    the project also handles future files containing missing values or inconsistent
    floor counts.
    """
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    data = data.drop_duplicates(subset=["ID"], keep="first").reset_index(drop=True)

    for col in STRING_COLUMNS:
        if col in data:
            data[col] = data[col].map(_clean_string)

    for col in NUMERIC_COLUMNS:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    numeric_defaults = {
        "BHK": 2,
        "Size_in_SqFt": 1000,
        "Price_in_Lakhs": 50,
        "Year_Built": REFERENCE_YEAR - 10,
        "Floor_No": 0,
        "Total_Floors": 1,
        "Nearby_Schools": 0,
        "Nearby_Hospitals": 0,
    }
    for col, default in numeric_defaults.items():
        median = data[col].median() if col in data and data[col].notna().any() else default
        data[col] = data[col].fillna(median).fillna(default)

    data["BHK"] = data["BHK"].clip(1, 10).round().astype(int)
    data["Size_in_SqFt"] = data["Size_in_SqFt"].clip(lower=100)
    data["Price_in_Lakhs"] = data["Price_in_Lakhs"].clip(lower=1)
    data["Year_Built"] = data["Year_Built"].clip(1950, REFERENCE_YEAR).round().astype(int)
    data["Floor_No"] = data["Floor_No"].clip(lower=0).round().astype(int)
    data["Total_Floors"] = data["Total_Floors"].clip(lower=1).round().astype(int)

    # A property cannot be located above the building's total floor count.
    data["Total_Floors"] = np.maximum(data["Total_Floors"], data["Floor_No"])
    data["Age_of_Property"] = (REFERENCE_YEAR - data["Year_Built"]).clip(lower=0)
    data["Nearby_Schools"] = data["Nearby_Schools"].clip(0, 20).round().astype(int)
    data["Nearby_Hospitals"] = data["Nearby_Hospitals"].clip(0, 20).round().astype(int)

    # The source column stores lakhs per square foot rounded to two decimals.
    # Recompute a precise and interpretable INR-per-square-foot measure.
    data["Price_per_SqFt_Source_Lakhs"] = pd.to_numeric(
        data.get("Price_per_SqFt"), errors="coerce"
    )
    data["Price_per_SqFt"] = (
        data["Price_in_Lakhs"] * 100_000 / data["Size_in_SqFt"]
    ).replace([np.inf, -np.inf], np.nan)
    data["Price_per_SqFt"] = data["Price_per_SqFt"].fillna(
        data["Price_per_SqFt"].median()
    )
    return data


def add_amenity_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    parsed = data["Amenities"].map(parse_amenities)
    data["Amenity_Count"] = parsed.map(len).astype(int)
    data["Amenity_Density_Score"] = (data["Amenity_Count"] / len(AMENITY_NAMES)).clip(0, 1)
    for amenity in AMENITY_NAMES:
        data[f"Has_{amenity}"] = parsed.map(lambda items, a=amenity: int(a in items))
    return data


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    denominator = data["Total_Floors"].replace(0, 1)
    data["Floor_Ratio"] = (data["Floor_No"] / denominator).clip(0, 1)
    transport_score = data["Public_Transport_Accessibility"].map(
        {"Low": 0.0, "Medium": 0.5, "High": 1.0}
    ).fillna(0.0)
    data["Infrastructure_Score"] = 100 * (
        0.30 * transport_score
        + 0.25 * (data["Nearby_Schools"] / 10).clip(0, 1)
        + 0.20 * (data["Nearby_Hospitals"] / 10).clip(0, 1)
        + 0.25 * data["Amenity_Density_Score"].clip(0, 1)
    )
    return data


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the two project targets using documented, reproducible rules.

    Future value uses a domain-informed compound growth rate. Good Investment is a
    balanced multi-factor label that rewards below-local-market pricing, growth
    potential, transport, amenities, readiness, security, parking, and nearby social
    infrastructure. Target-derived columns are excluded from model inputs.
    """
    data = df.copy()

    city_median = data.groupby("City")["Price_per_SqFt"].transform("median")
    segment_median = data.groupby(
        ["City", "Locality", "Property_Type"], observed=True
    )["Price_per_SqFt"].transform("median")
    overall_median = float(data["Price_per_SqFt"].median())

    city_factor = (((city_median / overall_median) - 1) * 0.008).clip(-0.006, 0.012)
    transport = data["Public_Transport_Accessibility"].map(
        {"Low": 0.000, "Medium": 0.008, "High": 0.016}
    ).fillna(0.0)
    age_bonus = np.select(
        [data["Age_of_Property"] <= 10, data["Age_of_Property"] <= 20, data["Age_of_Property"] >= 30],
        [0.008, 0.004, -0.004],
        default=0.0,
    )
    property_bonus = data["Property_Type"].map(
        {"Apartment": 0.004, "Independent House": 0.002, "Villa": 0.003}
    ).fillna(0.0)

    annual_growth = (
        0.055
        + city_factor
        + transport
        + data["Availability_Status"].eq("Ready_to_Move").astype(float) * 0.005
        + data["Security"].eq("Yes").astype(float) * 0.004
        + data["Parking_Space"].eq("Yes").astype(float) * 0.004
        + data["Amenity_Density_Score"] * 0.010
        + (data["Nearby_Schools"] / 10).clip(0, 1) * 0.006
        + (data["Nearby_Hospitals"] / 10).clip(0, 1) * 0.005
        + age_bonus
        + property_bonus
        + data["Owner_Type"].eq("Builder").astype(float) * 0.002
    ).clip(0.045, 0.12)

    data["Growth_Rate_Annual"] = annual_growth
    data["Future_Price_5Y"] = data["Price_in_Lakhs"] * np.power(1 + annual_growth, 5)
    data["Expected_ROI_5Y"] = (
        (data["Future_Price_5Y"] / data["Price_in_Lakhs"]) - 1
    ) * 100
    data["Below_Segment_Median"] = (
        data["Price_per_SqFt"] <= segment_median
    ).astype(int)

    score = (
        data["Below_Segment_Median"] * 2.0
        + (annual_growth >= annual_growth.median()).astype(float) * 2.0
        + data["Public_Transport_Accessibility"].eq("High").astype(float)
        + (data["Amenity_Count"] >= 3).astype(float)
        + data["Availability_Status"].eq("Ready_to_Move").astype(float)
        + data["Security"].eq("Yes").astype(float) * 0.5
        + data["Parking_Space"].eq("Yes").astype(float) * 0.5
        + (
            (data["Nearby_Schools"] >= 5)
            & (data["Nearby_Hospitals"] >= 3)
        ).astype(float)
    )
    data["Investment_Score"] = score
    data["Good_Investment"] = (score >= 5.0).astype(int)
    return data


def engineer_features(df: pd.DataFrame, include_targets: bool = True) -> pd.DataFrame:
    data = clean_base_data(df)
    data = add_amenity_features(data)
    data = add_structural_features(data)
    if include_targets:
        data = add_targets(data)
    return data


def prepare_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready columns from either raw or already engineered records."""
    data = df.copy()
    missing_engineered = any(c not in data.columns for c in ["Amenity_Count", "Floor_Ratio"])
    if missing_engineered:
        data = engineer_features(data, include_targets=False)
    else:
        # Ensure key derived values remain consistent for user-entered records.
        data = clean_base_data(data)
        data = add_amenity_features(data)
        data = add_structural_features(data)
    for col in MODEL_FEATURES:
        if col not in data.columns:
            data[col] = 0 if col.startswith("Has_") else "Unknown"
    return data[MODEL_FEATURES].copy()
