from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import APP_TITLE, EDA_SUMMARY_PATH, MODEL_METADATA_PATH
from src.data_loader import load_processed_data
from src.ui import hero, inject_css, model_disclaimer

st.set_page_config(page_title=APP_TITLE, page_icon="🏠", layout="wide")
inject_css()
hero(
    "🏠 Real Estate Investment Advisor",
    "Machine-learning decision support for property profitability, five-year value forecasting, and market exploration.",
)


try:
    df = load_processed_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.code("python scripts/run_pipeline.py\nstreamlit run app.py", language="bash")
    st.stop()

metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")) if MODEL_METADATA_PATH.exists() else {}

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Properties", f"{len(df):,}")
c2.metric("States", f"{df['State'].nunique():,}")
c3.metric("Cities", f"{df['City'].nunique():,}")
c4.metric("Good investments", f"{df['Good_Investment'].mean():.1%}")
c5.metric("Average 5Y ROI", f"{df['Expected_ROI_5Y'].mean():.1f}%")

st.markdown("### Market snapshot")
left, right = st.columns([1.15, 1])
with left:
    city_summary = (
        df.groupby("City", observed=True)
        .agg(Average_Price_Lakhs=("Price_in_Lakhs", "mean"), Listings=("ID", "count"))
        .sort_values("Average_Price_Lakhs", ascending=False)
        .head(12).reset_index()
    )
    fig = px.bar(
        city_summary.sort_values("Average_Price_Lakhs"),
        x="Average_Price_Lakhs", y="City", orientation="h",
        title="Cities with the highest average property prices",
        labels={"Average_Price_Lakhs": "Average price (₹ lakh)"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, width="stretch")
with right:
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="Size_in_SqFt", y="Price_in_Lakhs", color="Good_Investment",
        opacity=0.45, title="Property size versus current price",
        labels={"Size_in_SqFt": "Size (sq ft)", "Price_in_Lakhs": "Price (₹ lakh)"},
        color_continuous_scale="Viridis",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, width="stretch")

st.markdown("### Packaged model summary")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Best classifier", metadata.get("best_classifier", "Run pipeline"))
mc2.metric("Classifier F1", f"{metadata.get('classifier_f1', 0):.3f}")
mc3.metric("Best regressor", metadata.get("best_regressor", "Run pipeline"))
mc4.metric("Regression RMSE", f"{metadata.get('regressor_rmse_lakhs', 0):.2f} lakh")

st.info(
    "Use the pages in the sidebar to predict a property's investment potential, filter listings, "
    "explore all 20 EDA questions, and compare model performance."
)
model_disclaimer()
