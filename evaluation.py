from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, y_pred, y_probability) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_diagnostics(y_true, y_pred, y_probability) -> dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, y_probability)
    # Store at most 250 points so the report remains lightweight.
    indices = np.linspace(0, len(fpr) - 1, min(250, len(fpr))).astype(int)
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": {
            "fpr": fpr[indices].tolist(),
            "tpr": tpr[indices].tolist(),
            "thresholds": thresholds[indices].tolist(),
        },
    }


def regression_diagnostics(y_true, y_pred, sample_size: int = 1500) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(sample_size, len(y_true))
    indices = np.linspace(0, len(y_true) - 1, n).astype(int)
    residuals = y_true - y_pred
    return {
        "sample": {
            "actual": y_true[indices].tolist(),
            "predicted": y_pred[indices].tolist(),
            "residual": residuals[indices].tolist(),
        }
    }
