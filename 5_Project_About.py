from __future__ import annotations

import json

import streamlit as st

from src.config import APP_TITLE, MODEL_METADATA_PATH
from src.ui import hero, inject_css, model_disclaimer

st.set_page_config(page_title=f"About | {APP_TITLE}", page_icon="📘", layout="wide")
inject_css()
hero("📘 Project Methodology", "Architecture, target logic, experiment tracking, deliverables, and reproducible execution steps.")

metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")) if MODEL_METADATA_PATH.exists() else {}

st.markdown("""
### Business objective
The application helps investors classify whether a property is a **Good Investment** and estimate its **price after five years**. It combines data cleaning, feature engineering, exploratory analysis, classification, regression, evaluation, MLflow-ready experiment tracking, and Streamlit deployment.

### End-to-end architecture
`Raw CSV → validation and cleaning → domain feature engineering → synthetic target generation → train/test split → 5+ classification models + 5+ regression models → metric-based model selection → serialized pipelines → Streamlit inference and analytics`

### Target construction
- **Future_Price_5Y:** compound growth from current price using a 4.5%–12% annual rate influenced by city price level, transport, readiness, security, parking, amenities, nearby schools/hospitals, property age, property type, and owner type.
- **Good_Investment:** a multi-factor score rewarding below-segment-median price per square foot, above-median growth potential, high transport accessibility, at least three amenities, ready-to-move status, security, parking, and social infrastructure. Score ≥ 5 becomes class 1.
- Target-derived columns are excluded from model inputs to prevent direct leakage.

### Data-quality decisions
- Duplicate IDs are removed.
- Numeric values are coerced and median-imputed when necessary.
- String values are standardized.
- Invalid floor relationships are corrected by ensuring total floors are never below the selected floor.
- The source's rounded lakhs-per-square-foot value is retained as `Price_per_SqFt_Source_Lakhs`; a precise INR-per-square-foot feature is recomputed.
- The reference year is fixed at 2025 because the supplied `Age_of_Property` values exactly match `2025 - Year_Built`, making the workflow reproducible.

### MLflow
Run `python scripts/run_pipeline.py --enable-mlflow` after installing requirements. The project uses a local SQLite tracking URI (`mlflow.db`), logs model parameters/metrics/artifacts, and can be extended to register the selected production models.
""")

st.markdown("### Reproduce in VS Code")
st.code(r"""python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_project.py
streamlit run app.py""", language="powershell")

st.markdown("### Packaged model metadata")
st.json(metadata)
model_disclaimer()
