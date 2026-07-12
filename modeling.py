from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.config import (
    CLASSIFICATION_LEADERBOARD_PATH,
    CLASSIFIER_PATH,
    DEFAULT_TRAINING_ROWS,
    DIAGNOSTICS_PATH,
    EXPERIMENT_RUNS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_LEADERBOARD_PATH,
    REGRESSOR_PATH,
    REPORTS_DIR,
)
from src.evaluation import (
    classification_diagnostics,
    classification_metrics,
    regression_diagnostics,
    regression_metrics,
)
from src.feature_engineering import prepare_model_features
from src.mlflow_tracking import mlflow_run, setup_mlflow
from src.preprocessing import build_one_hot_preprocessor, build_ordinal_preprocessor


@dataclass(frozen=True)
class Candidate:
    name: str
    factory: Callable[[], BaseEstimator]
    encoding: str = "ordinal"


def classification_candidates() -> list[Candidate]:
    candidates = [
        Candidate(
            "Logistic Regression",
            lambda: LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear"),
            "ordinal",
        ),
        Candidate(
            "Decision Tree",
            lambda: DecisionTreeClassifier(
                max_depth=14, min_samples_leaf=12, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=50, max_depth=16, min_samples_leaf=4,
                class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            "Extra Trees",
            lambda: ExtraTreesClassifier(
                n_estimators=60, max_depth=18, min_samples_leaf=3,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            "Gaussian Naive Bayes",
            lambda: GaussianNB(var_smoothing=1e-8),
        ),
    ]
    return candidates


def regression_candidates() -> list[Candidate]:
    candidates = [
        Candidate("Linear Regression", lambda: LinearRegression(n_jobs=-1), "ordinal"),
        Candidate("Ridge Regression", lambda: Ridge(alpha=1.0), "ordinal"),
        Candidate(
            "Decision Tree",
            lambda: DecisionTreeRegressor(
                max_depth=16, min_samples_leaf=8, random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            "Random Forest",
            lambda: RandomForestRegressor(
                n_estimators=50, max_depth=18, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
        Candidate(
            "Extra Trees",
            lambda: ExtraTreesRegressor(
                n_estimators=60, max_depth=20, min_samples_leaf=3,
                n_jobs=-1, random_state=RANDOM_STATE
            ),
        ),
    ]
    return candidates


def _pipeline(candidate: Candidate) -> Pipeline:
    preprocessor = (
        build_one_hot_preprocessor()
        if candidate.encoding == "onehot"
        else build_ordinal_preprocessor()
    )
    return Pipeline([("preprocessor", preprocessor), ("model", candidate.factory())])


def _sample_training_data(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    # Stratify the sample by target to preserve class balance.
    positive = df[df["Good_Investment"] == 1]
    negative = df[df["Good_Investment"] == 0]
    positive_n = int(max_rows * len(positive) / len(df))
    negative_n = max_rows - positive_n
    sampled = pd.concat([
        positive.sample(positive_n, random_state=RANDOM_STATE),
        negative.sample(negative_n, random_state=RANDOM_STATE),
    ])
    return sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def _extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{i}" for i in range(getattr(model, "n_features_in_", 0))])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).reshape(-1)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    n = min(len(names), len(values))
    frame = pd.DataFrame({"feature": names[:n], "importance": values[:n]})
    frame["feature"] = (
        frame["feature"].astype(str).str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    total = frame["importance"].sum()
    if total > 0:
        frame["importance"] = frame["importance"] / total
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def train_all_models(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_TRAINING_ROWS,
    enable_mlflow: bool = False,
) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _sample_training_data(df, max_rows=max_rows)
    X = prepare_model_features(train_df)
    y_class = train_df["Good_Investment"].astype(int)
    y_reg = train_df["Future_Price_5Y"].astype(float)

    indices = np.arange(len(train_df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_class,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train, yc_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    class_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    best_classifier = None
    best_classifier_name = None
    best_classifier_score = -np.inf
    best_classifier_diag = None
    best_regressor = None
    best_regressor_name = None
    best_regressor_score = np.inf
    best_regressor_diag = None

    for candidate in classification_candidates():
        print(f"  Training classifier: {candidate.name}", flush=True)
        pipeline = _pipeline(candidate)
        started = time.perf_counter()
        with mlflow_run(f"classification-{candidate.name}", enabled=enable_mlflow) as mlflow:
            pipeline.fit(X_train, yc_train)
            pred = pipeline.predict(X_test)
            probability = pipeline.predict_proba(X_test)[:, 1]
            metrics = classification_metrics(yc_test, pred, probability)
            elapsed = time.perf_counter() - started
            row = {
                "model": candidate.name,
                **metrics,
                "training_seconds": elapsed,
                "training_rows": len(X_train),
            }
            class_rows.append(row)
            experiment_rows.append({"task": "classification", **row})
            if mlflow is not None:
                mlflow.log_params({"model": candidate.name, "encoding": candidate.encoding})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, name="model")
            if metrics["f1"] > best_classifier_score:
                best_classifier_score = metrics["f1"]
                best_classifier = pipeline
                best_classifier_name = candidate.name
                best_classifier_diag = classification_diagnostics(yc_test, pred, probability)

    for candidate in regression_candidates():
        print(f"  Training regressor: {candidate.name}", flush=True)
        pipeline = _pipeline(candidate)
        started = time.perf_counter()
        with mlflow_run(f"regression-{candidate.name}", enabled=enable_mlflow) as mlflow:
            pipeline.fit(X_train, yr_train)
            pred = pipeline.predict(X_test)
            metrics = regression_metrics(yr_test, pred)
            elapsed = time.perf_counter() - started
            row = {
                "model": candidate.name,
                **metrics,
                "training_seconds": elapsed,
                "training_rows": len(X_train),
            }
            reg_rows.append(row)
            experiment_rows.append({"task": "regression", **row})
            if mlflow is not None:
                mlflow.log_params({"model": candidate.name, "encoding": candidate.encoding})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, name="model")
            if metrics["rmse"] < best_regressor_score:
                best_regressor_score = metrics["rmse"]
                best_regressor = pipeline
                best_regressor_name = candidate.name
                best_regressor_diag = regression_diagnostics(yr_test, pred)

    class_leaderboard = pd.DataFrame(class_rows).sort_values(
        ["f1", "roc_auc"], ascending=False
    )
    reg_leaderboard = pd.DataFrame(reg_rows).sort_values("rmse", ascending=True)
    class_leaderboard.to_csv(CLASSIFICATION_LEADERBOARD_PATH, index=False)
    reg_leaderboard.to_csv(REGRESSION_LEADERBOARD_PATH, index=False)
    pd.DataFrame(experiment_rows).to_csv(EXPERIMENT_RUNS_PATH, index=False)

    joblib.dump(
        {"model": best_classifier, "model_name": best_classifier_name},
        CLASSIFIER_PATH,
        compress=3,
    )
    joblib.dump(
        {"model": best_regressor, "model_name": best_regressor_name},
        REGRESSOR_PATH,
        compress=3,
    )

    class_importance = _extract_feature_importance(best_classifier)
    reg_importance = _extract_feature_importance(best_regressor)
    class_importance.to_csv(REPORTS_DIR / "feature_importance_classifier.csv", index=False)
    reg_importance.to_csv(REPORTS_DIR / "feature_importance_regressor.csv", index=False)

    if enable_mlflow:
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name="best-model-registration"):
            mlflow.log_params({
                "best_classifier": best_classifier_name,
                "best_regressor": best_regressor_name,
            })
            mlflow.sklearn.log_model(
                best_classifier,
                name="best_classifier",
                registered_model_name="RealEstateInvestmentAdvisorClassifier",
            )
            mlflow.sklearn.log_model(
                best_regressor,
                name="best_regressor",
                registered_model_name="RealEstateInvestmentAdvisorRegressor",
            )

    diagnostics = {
        "classification": {
            "best_model": best_classifier_name,
            "metrics": class_leaderboard.iloc[0].to_dict(),
            **(best_classifier_diag or {}),
        },
        "regression": {
            "best_model": best_regressor_name,
            "metrics": reg_leaderboard.iloc[0].to_dict(),
            **(best_regressor_diag or {}),
        },
    }
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    metadata = {
        "project_version": "1.0.0",
        "reference_year": 2025,
        "training_rows_total": int(len(train_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classification_target": "Good_Investment",
        "regression_target": "Future_Price_5Y",
        "best_classifier": best_classifier_name,
        "best_regressor": best_regressor_name,
        "classifier_f1": float(best_classifier_score),
        "regressor_rmse_lakhs": float(best_regressor_score),
        "features": list(X.columns),
        "mlflow_enabled_for_packaged_run": bool(enable_mlflow),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
