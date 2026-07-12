from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import APP_TITLE
from src.data_loader import load_processed_data
from src.ui import hero, inject_css, model_disclaimer

st.set_page_config(page_title=f"Explorer | {APP_TITLE}", page_icon="🔎", layout="wide")
inject_css()
hero("🔎 Property Explorer", "Filter the supplied market and rank listings by model-ready investment indicators.")


df = load_processed_data()
with st.sidebar:
    st.header("Filters")
    states = st.multiselect("State", sorted(df["State"].unique()))
    working = df[df["State"].isin(states)] if states else df
    cities = st.multiselect("City", sorted(working["City"].unique()))
    property_types = st.multiselect("Property type", sorted(df["Property_Type"].unique()))
    bhk_range = st.slider("BHK range", int(df["BHK"].min()), int(df["BHK"].max()), (1, 5))
    price_range = st.slider(
        "Price range (₹ lakh)",
        float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()),
        (float(df["Price_in_Lakhs"].quantile(.05)), float(df["Price_in_Lakhs"].quantile(.95))),
    )
    min_size = st.number_input("Minimum size (sq ft)", min_value=0, value=500, step=100)
    only_good = st.checkbox("Only good investments", value=True)

filtered = df.copy()
if states:
    filtered = filtered[filtered["State"].isin(states)]
if cities:
    filtered = filtered[filtered["City"].isin(cities)]
if property_types:
    filtered = filtered[filtered["Property_Type"].isin(property_types)]
filtered = filtered[
    filtered["BHK"].between(*bhk_range)
    & filtered["Price_in_Lakhs"].between(*price_range)
    & (filtered["Size_in_SqFt"] >= min_size)
]
if only_good:
    filtered = filtered[filtered["Good_Investment"] == 1]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Matching properties", f"{len(filtered):,}")
m2.metric("Median price", f"₹{filtered['Price_in_Lakhs'].median():,.1f} lakh" if len(filtered) else "—")
m3.metric("Median 5Y ROI", f"{filtered['Expected_ROI_5Y'].median():.1f}%" if len(filtered) else "—")
m4.metric("Median price/sq ft", f"₹{filtered['Price_per_SqFt'].median():,.0f}" if len(filtered) else "—")

if filtered.empty:
    st.warning("No properties match the selected filters.")
    st.stop()

chart_data = (
    filtered.groupby("City", observed=True)
    .agg(Listings=("ID", "count"), Median_ROI=("Expected_ROI_5Y", "median"))
    .sort_values("Listings", ascending=False).head(15).reset_index()
)
fig = px.bar(chart_data, x="City", y="Listings", color="Median_ROI", title="Filtered listings by city", labels={"Median_ROI": "Median 5Y ROI (%)"})
st.plotly_chart(fig, width="stretch")

columns = [
    "ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Price_per_SqFt", "Good_Investment", "Investment_Score",
    "Expected_ROI_5Y", "Future_Price_5Y", "Availability_Status",
]
ranked = filtered.sort_values(
    ["Good_Investment", "Investment_Score", "Expected_ROI_5Y"], ascending=False
)[columns].head(5000)
st.dataframe(
    ranked,
    width="stretch",
    hide_index=True,
    column_config={
        "Price_in_Lakhs": st.column_config.NumberColumn("Price (₹ lakh)", format="%.2f"),
        "Price_per_SqFt": st.column_config.NumberColumn("Price/sq ft (₹)", format="%.0f"),
        "Expected_ROI_5Y": st.column_config.NumberColumn("5Y ROI", format="%.2f%%"),
        "Future_Price_5Y": st.column_config.NumberColumn("Future price (₹ lakh)", format="%.2f"),
        "Good_Investment": st.column_config.CheckboxColumn("Good investment"),
    },
)
st.download_button(
    "Download filtered properties",
    data=filtered[columns].to_csv(index=False).encode("utf-8"),
    file_name="filtered_investment_properties.csv",
    mime="text/csv",
)
model_disclaimer()
