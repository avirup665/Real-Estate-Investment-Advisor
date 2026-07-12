from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.config import PROJECT_ROOT


def setup_mlflow(experiment_name: str = "RealEstateInvestmentAdvisor"):
    """Configure local MLflow tracking with a SQLite backend.

    MLflow is imported lazily so the Streamlit application can still run when users
    choose not to install the optional experiment-tracking dependency.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Run: python -m pip install mlflow"
        ) from exc

    tracking_uri = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


@contextmanager
def mlflow_run(run_name: str, enabled: bool = False) -> Iterator[object | None]:
    if not enabled:
        yield None
        return
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield mlflow
