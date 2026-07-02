"""
4_Data_Explorer.py
Real Estate Investment Advisor
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Data Explorer", page_icon="📁", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_data.csv"

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()

st.title("📁 Data Explorer")

if df is None:
    st.error("cleaned_data.csv not found in data folder.")
    st.stop()

st.sidebar.header("Filters")

search = st.sidebar.text_input("Search")

filtered = df.copy()

if search:
    mask = filtered.astype(str).apply(
        lambda c: c.str.contains(search, case=False, na=False)
    ).any(axis=1)
    filtered = filtered[mask]

if "State" in filtered.columns:
    states = st.sidebar.multiselect(
        "State",
        sorted(filtered["State"].dropna().unique())
    )
    if states:
        filtered = filtered[filtered["State"].isin(states)]

if "Property_Type" in filtered.columns:
    types = st.sidebar.multiselect(
        "Property Type",
        sorted(filtered["Property_Type"].dropna().unique())
    )
    if types:
        filtered = filtered[filtered["Property_Type"].isin(types)]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", len(filtered))
c2.metric("Columns", len(filtered.columns))
c3.metric("Missing Values", int(filtered.isna().sum().sum()))
c4.metric("Duplicate Rows", int(filtered.duplicated().sum()))

st.subheader("Dataset Preview")
st.dataframe(filtered, use_container_width=True)

st.subheader("Column Information")
info = pd.DataFrame({
    "Column": filtered.columns,
    "Data Type": filtered.dtypes.astype(str).values,
    "Missing": filtered.isna().sum().values,
    "Unique": filtered.nunique().values
})
st.dataframe(info, use_container_width=True, hide_index=True)

st.subheader("Summary Statistics")
st.dataframe(filtered.describe(include="all").transpose(), use_container_width=True)

num = filtered.select_dtypes(include="number")
if not num.empty:
    st.subheader("Correlation Heatmap")
    fig = px.imshow(num.corr(numeric_only=True),
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Numeric Feature Distribution")
    col = st.selectbox("Select Numeric Column", num.columns)
    fig2 = px.histogram(filtered, x=col, nbins=30, title=f"{col} Distribution")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Download Filtered Dataset")
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download CSV",
    csv,
    "filtered_dataset.csv",
    "text/csv",
    use_container_width=True
)

st.caption("Real Estate Investment Advisor • Data Explorer")
