from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "india_housing_prices.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cleaned_housing_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"
CLASSIFICATION_LEADERBOARD_PATH = REPORTS_DIR / "classification_leaderboard.csv"
REGRESSION_LEADERBOARD_PATH = REPORTS_DIR / "regression_leaderboard.csv"
DIAGNOSTICS_PATH = REPORTS_DIR / "prediction_diagnostics.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
EXPERIMENT_RUNS_PATH = REPORTS_DIR / "experiment_runs.csv"

REFERENCE_YEAR = 2025
RANDOM_STATE = 42
DEFAULT_TRAINING_ROWS = 30_000
APP_TITLE = "Real Estate Investment Advisor"

AMENITY_NAMES = ["Gym", "Pool", "Clubhouse", "Garden", "Playground"]

MODEL_FEATURES = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Facing",
    "Owner_Type",
    "Availability_Status",
    "Amenity_Count",
    "Amenity_Density_Score",
    "Has_Gym",
    "Has_Pool",
    "Has_Clubhouse",
    "Has_Garden",
    "Has_Playground",
    "Floor_Ratio",
    "Infrastructure_Score",
]

CATEGORICAL_FEATURES = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "Furnished_Status",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]

NUMERIC_FEATURES = [c for c in MODEL_FEATURES if c not in CATEGORICAL_FEATURES]
