from __future__ import annotations

from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import CLASSIFIER_PATH, REGRESSOR_PATH
from src.feature_engineering import prepare_model_features


@lru_cache(maxsize=1)
def load_model_bundles() -> tuple[dict[str, Any], dict[str, Any]]:
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run: python scripts/run_pipeline.py"
        )
    return joblib.load(CLASSIFIER_PATH), joblib.load(REGRESSOR_PATH)


def predict_property(raw_record: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    frame = raw_record if isinstance(raw_record, pd.DataFrame) else pd.DataFrame([raw_record])
    X = prepare_model_features(frame)
    classifier_bundle, regressor_bundle = load_model_bundles()
    classifier = classifier_bundle["model"]
    regressor = regressor_bundle["model"]

    probability = float(classifier.predict_proba(X)[:, 1][0])
    classification = int(probability >= 0.5)
    future_price = max(float(regressor.predict(X)[0]), 0.0)
    current_price = float(frame.iloc[0]["Price_in_Lakhs"])
    gain = future_price - current_price
    roi = ((future_price / current_price) - 1) * 100 if current_price > 0 else 0.0
    annualized = ((future_price / current_price) ** (1 / 5) - 1) * 100 if current_price > 0 else 0.0

    return {
        "good_investment": bool(classification),
        "confidence": probability if classification else 1 - probability,
        "good_investment_probability": probability,
        "future_price_lakhs": future_price,
        "estimated_gain_lakhs": gain,
        "roi_5y_percent": roi,
        "annualized_return_percent": annualized,
        "classifier_name": classifier_bundle.get("model_name", "Model"),
        "regressor_name": regressor_bundle.get("model_name", "Model"),
    }
