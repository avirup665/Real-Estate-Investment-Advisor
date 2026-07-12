from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    APP_TITLE, CLASSIFICATION_LEADERBOARD_PATH, DIAGNOSTICS_PATH,
    REGRESSION_LEADERBOARD_PATH, REPORTS_DIR,
)
from src.ui import hero, inject_css, model_disclaimer

st.set_page_config(page_title=f"Models | {APP_TITLE}", page_icon="🧠", layout="wide")
inject_css()
hero("🧠 Model Performance", "Transparent comparison of at least five classification and five regression algorithms.")

if not CLASSIFICATION_LEADERBOARD_PATH.exists() or not REGRESSION_LEADERBOARD_PATH.exists():
    st.error("Model reports are missing. Run: python scripts/run_pipeline.py")
    st.stop()

class_lb = pd.read_csv(CLASSIFICATION_LEADERBOARD_PATH)
reg_lb = pd.read_csv(REGRESSION_LEADERBOARD_PATH)
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))

st.subheader("Classification leaderboard — target: Good_Investment")
st.dataframe(
    class_lb.style.format({"accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "roc_auc": "{:.4f}", "training_seconds": "{:.2f}"}),
    width="stretch", hide_index=True,
)
fig = px.bar(class_lb.sort_values("f1"), x="f1", y="model", orientation="h", color="roc_auc", title="Classification F1 and ROC AUC")
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    cm = diagnostics["classification"]["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion matrix"), width="stretch")
with c2:
    roc = pd.DataFrame(diagnostics["classification"]["roc_curve"])
    fig = px.line(roc, x="fpr", y="tpr", title="ROC curve", labels={"fpr": "False-positive rate", "tpr": "True-positive rate"})
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    st.plotly_chart(fig, width="stretch")

st.subheader("Regression leaderboard — target: Future_Price_5Y")
st.dataframe(
    reg_lb.style.format({"rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.4f}", "training_seconds": "{:.2f}"}),
    width="stretch", hide_index=True,
)
fig = px.bar(reg_lb.sort_values("rmse", ascending=False), x="rmse", y="model", orientation="h", color="r2", title="Regression RMSE and R²")
st.plotly_chart(fig, width="stretch")

sample = pd.DataFrame(diagnostics["regression"]["sample"])
c1, c2 = st.columns(2)
with c1:
    fig = px.scatter(sample, x="actual", y="predicted", opacity=.5, title="Actual versus predicted future price", labels={"actual": "Actual (₹ lakh)", "predicted": "Predicted (₹ lakh)"})
    lo = min(sample["actual"].min(), sample["predicted"].min())
    hi = max(sample["actual"].max(), sample["predicted"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash"))
    st.plotly_chart(fig, width="stretch")
with c2:
    st.plotly_chart(px.scatter(sample, x="predicted", y="residual", opacity=.5, title="Residual plot", labels={"predicted": "Predicted (₹ lakh)", "residual": "Residual (₹ lakh)"}), width="stretch")

st.subheader("Global feature importance")
choice = st.radio("Model task", ["Classifier", "Regressor"], horizontal=True)
path = REPORTS_DIR / ("feature_importance_classifier.csv" if choice == "Classifier" else "feature_importance_regressor.csv")
importance = pd.read_csv(path).head(20)
st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", title=f"Top 20 {choice.lower()} features"), width="stretch")
model_disclaimer()
