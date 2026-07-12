from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import PROCESSED_DATA_PATH, RAW_DATA_PATH

REQUIRED_COLUMNS = {
    "ID", "State", "City", "Locality", "Property_Type", "BHK",
    "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type",
    "Availability_Status",
}


def validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Place india_housing_prices.csv "
            "inside data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    validate_columns(df.columns)
    return df


@lru_cache(maxsize=2)
def _load_processed_cached(path_string: str) -> pd.DataFrame:
    return pd.read_csv(path_string, low_memory=False)


def load_processed_data(path: str | Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run: python scripts/run_pipeline.py"
        )
    # Shared read-only cache prevents every Streamlit page from loading another
    # 250,000-row copy into memory.
    return _load_processed_cached(str(path))
