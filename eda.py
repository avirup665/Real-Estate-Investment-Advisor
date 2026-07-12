from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EDA_SUMMARY_PATH, REPORTS_DIR


def build_eda_summary(df: pd.DataFrame, output_path: str | Path = EDA_SUMMARY_PATH) -> dict:
    numeric = df.select_dtypes(include=np.number)
    top_cities = (
        df.groupby("City", observed=True)["Price_in_Lakhs"]
        .mean().sort_values(ascending=False).head(10)
    )
    top_states = (
        df.groupby("State", observed=True)["Price_per_SqFt"]
        .mean().sort_values(ascending=False).head(10)
    )
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ids": int(df["ID"].duplicated().sum()),
        "states": int(df["State"].nunique()),
        "cities": int(df["City"].nunique()),
        "localities": int(df["Locality"].nunique()),
        "average_price_lakhs": float(df["Price_in_Lakhs"].mean()),
        "median_price_lakhs": float(df["Price_in_Lakhs"].median()),
        "average_price_per_sqft": float(df["Price_per_SqFt"].mean()),
        "good_investment_share": float(df["Good_Investment"].mean()),
        "average_roi_5y": float(df["Expected_ROI_5Y"].mean()),
        "top_cities_by_average_price": top_cities.round(3).to_dict(),
        "top_states_by_average_price_per_sqft": top_states.round(3).to_dict(),
        "numeric_correlation": numeric.corr(numeric_only=True).round(4).to_dict(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    target_distribution = (
        df["Good_Investment"].value_counts(dropna=False)
        .rename_axis("Good_Investment")
        .reset_index(name="Count")
    )
    target_distribution["Share"] = target_distribution["Count"] / len(df)
    target_distribution.to_csv(REPORTS_DIR / "target_distribution.csv", index=False)

    city_market = (
        df.groupby(["State", "City"], observed=True)
        .agg(
            Listings=("ID", "count"),
            Average_Price_Lakhs=("Price_in_Lakhs", "mean"),
            Median_Price_per_SqFt=("Price_per_SqFt", "median"),
            Good_Investment_Rate=("Good_Investment", "mean"),
            Average_ROI_5Y=("Expected_ROI_5Y", "mean"),
        )
        .reset_index()
        .sort_values("Average_Price_Lakhs", ascending=False)
    )
    city_market.to_csv(REPORTS_DIR / "city_market_summary.csv", index=False)
    return summary


def build_data_quality_report(raw: pd.DataFrame, processed: pd.DataFrame) -> dict:
    report = {
        "raw": {
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "missing_cells": int(raw.isna().sum().sum()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "duplicate_ids": int(raw["ID"].duplicated().sum()),
            "floor_above_total_count": int((raw["Floor_No"] > raw["Total_Floors"]).sum()),
            "nonpositive_price_per_sqft_source_count": int((raw["Price_per_SqFt"] <= 0).sum()),
        },
        "processed": {
            "rows": int(len(processed)),
            "columns": int(processed.shape[1]),
            "missing_cells": int(processed.isna().sum().sum()),
            "duplicate_ids": int(processed["ID"].duplicated().sum()),
            "floor_above_total_count": int((processed["Floor_No"] > processed["Total_Floors"]).sum()),
            "good_investment_rate": float(processed["Good_Investment"].mean()),
        },
        "actions": [
            "Validated required columns",
            "Removed duplicate IDs",
            "Standardized categorical text",
            "Coerced and imputed numeric values defensively",
            "Corrected total floors when floor number was higher",
            "Recomputed price per square foot in INR",
            "Engineered amenities, infrastructure, targets, ROI, and investment score",
        ],
    }
    (REPORTS_DIR / "data_quality_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
