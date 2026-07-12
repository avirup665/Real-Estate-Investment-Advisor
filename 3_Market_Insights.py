from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import APP_TITLE
from src.data_loader import load_processed_data
from src.ui import hero, inject_css

st.set_page_config(page_title=f"Insights | {APP_TITLE}", page_icon="📊", layout="wide")
inject_css()
hero("📊 Market Insights", "Interactive answers to the 20 exploratory-data-analysis questions in the project brief.")


df = load_processed_data()
sample = df.sample(min(7000, len(df)), random_state=42)

tab1, tab2, tab3, tab4 = st.tabs([
    "1–5 Price & size", "6–10 Location", "11–15 Relationships", "16–20 Investment factors"
])

with tab1:
    st.subheader("1. Distribution of property prices")
    st.plotly_chart(px.histogram(df, x="Price_in_Lakhs", nbins=50, marginal="box", labels={"Price_in_Lakhs": "Price (₹ lakh)"}), width="stretch")
    st.subheader("2. Distribution of property sizes")
    st.plotly_chart(px.histogram(df, x="Size_in_SqFt", nbins=50, marginal="box", labels={"Size_in_SqFt": "Size (sq ft)"}), width="stretch")
    st.subheader("3. Price per sq ft by property type")
    st.plotly_chart(px.box(sample, x="Property_Type", y="Price_per_SqFt", points=False, labels={"Price_per_SqFt": "₹ per sq ft"}), width="stretch")
    st.subheader("4. Relationship between size and price")
    st.plotly_chart(px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", opacity=.35), width="stretch")
    st.subheader("5. Outliers in price per sq ft and size")
    outlier_frame = sample.melt(id_vars=["Property_Type"], value_vars=["Price_per_SqFt", "Size_in_SqFt"], var_name="Metric", value_name="Value")
    st.plotly_chart(px.box(outlier_frame, x="Metric", y="Value", color="Property_Type", points="outliers", log_y=True), width="stretch")

with tab2:
    st.subheader("6. Average price per sq ft by state")
    state = df.groupby("State", observed=True)["Price_per_SqFt"].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(state, x="State", y="Price_per_SqFt", labels={"Price_per_SqFt": "Average ₹ per sq ft"}), width="stretch")
    st.subheader("7. Average property price by city")
    city = df.groupby("City", observed=True)["Price_in_Lakhs"].mean().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(city, x="City", y="Price_in_Lakhs", labels={"Price_in_Lakhs": "Average price (₹ lakh)"}), width="stretch")
    st.subheader("8. Median property age by locality")
    locality = df.groupby(["City", "Locality"], observed=True)["Age_of_Property"].median().nlargest(25).reset_index()
    st.dataframe(locality, width="stretch", hide_index=True)
    st.subheader("9. BHK distribution across cities")
    top_cities = df["City"].value_counts().head(12).index
    bhk = df[df["City"].isin(top_cities)].groupby(["City", "BHK"], observed=True).size().reset_index(name="Listings")
    st.plotly_chart(px.bar(bhk, x="City", y="Listings", color="BHK", barmode="stack"), width="stretch")
    st.subheader("10. Price trends for the five most expensive localities")
    loc = df.groupby(["City", "Locality"], observed=True)["Price_in_Lakhs"].mean().nlargest(5).reset_index()
    loc["Location"] = loc["City"] + " / " + loc["Locality"]
    st.plotly_chart(px.bar(loc, x="Location", y="Price_in_Lakhs", labels={"Price_in_Lakhs": "Average price (₹ lakh)"}), width="stretch")

with tab3:
    st.subheader("11. Numeric-feature correlation")
    corr_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Amenity_Count", "Expected_ROI_5Y"]
    corr = df[corr_cols].corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width="stretch")
    st.subheader("12. Nearby schools versus price per sq ft")
    school = df.groupby("Nearby_Schools", observed=True)["Price_per_SqFt"].mean().reset_index()
    st.plotly_chart(px.line(school, x="Nearby_Schools", y="Price_per_SqFt", markers=True), width="stretch")
    st.subheader("13. Nearby hospitals versus price per sq ft")
    hosp = df.groupby("Nearby_Hospitals", observed=True)["Price_per_SqFt"].mean().reset_index()
    st.plotly_chart(px.line(hosp, x="Nearby_Hospitals", y="Price_per_SqFt", markers=True), width="stretch")
    st.subheader("14. Price by furnished status")
    st.plotly_chart(px.box(sample, x="Furnished_Status", y="Price_in_Lakhs", points=False), width="stretch")
    st.subheader("15. Price per sq ft by facing direction")
    st.plotly_chart(px.box(sample, x="Facing", y="Price_per_SqFt", points=False), width="stretch")

with tab4:
    st.subheader("16. Properties by owner type")
    owner = df["Owner_Type"].value_counts().rename_axis("Owner_Type").reset_index(name="Listings")
    st.plotly_chart(px.pie(owner, names="Owner_Type", values="Listings", hole=.4), width="stretch")
    st.subheader("17. Properties by availability status")
    status = df["Availability_Status"].value_counts().rename_axis("Availability_Status").reset_index(name="Listings")
    st.plotly_chart(px.bar(status, x="Availability_Status", y="Listings"), width="stretch")
    st.subheader("18. Parking space and property price")
    st.plotly_chart(px.box(sample, x="Parking_Space", y="Price_in_Lakhs", points=False), width="stretch")
    st.subheader("19. Amenities and price per sq ft")
    amen = df.groupby("Amenity_Count", observed=True)["Price_per_SqFt"].mean().reset_index()
    st.plotly_chart(px.line(amen, x="Amenity_Count", y="Price_per_SqFt", markers=True), width="stretch")
    st.subheader("20. Public transport and investment potential")
    transport = df.groupby("Public_Transport_Accessibility", observed=True).agg(Average_Price_per_SqFt=("Price_per_SqFt", "mean"), Good_Investment_Rate=("Good_Investment", "mean"), Average_ROI=("Expected_ROI_5Y", "mean")).reset_index()
    st.dataframe(transport, width="stretch", hide_index=True, column_config={"Good_Investment_Rate": st.column_config.NumberColumn(format="%.1f%%")})
    st.plotly_chart(px.bar(transport, x="Public_Transport_Accessibility", y="Good_Investment_Rate", labels={"Good_Investment_Rate": "Good-investment rate"}), width="stretch")


st.markdown("### Brief-specific proxy analyses")
left, right = st.columns(2)
with left:
    security_proxy = (
        df.groupby("Security", observed=True)["Good_Investment"]
        .mean().reset_index(name="Good_Investment_Rate")
    )
    st.plotly_chart(
        px.bar(
            security_proxy,
            x="Security",
            y="Good_Investment_Rate",
            title="Security availability as a crime-risk proxy",
            labels={"Good_Investment_Rate": "Good-investment rate"},
        ),
        width="stretch",
    )
with right:
    infra = df.assign(
        Infrastructure_Band=pd.cut(
            df["Infrastructure_Score"], bins=[0, 40, 55, 70, 85, 100],
            include_lowest=True
        )
    ).groupby("Infrastructure_Band", observed=True).agg(
        Average_Future_Price=("Future_Price_5Y", "mean"),
        Average_ROI=("Expected_ROI_5Y", "mean"),
    ).reset_index()
    infra["Infrastructure_Band"] = infra["Infrastructure_Band"].astype(str)
    st.plotly_chart(
        px.line(
            infra,
            x="Infrastructure_Band",
            y="Average_Future_Price",
            markers=True,
            title="Infrastructure score and estimated resale value",
            labels={"Average_Future_Price": "Average 5Y price (₹ lakh)"},
        ),
        width="stretch",
    )

location_heatmap = df.groupby(["State", "Property_Type"], observed=True)["Price_per_SqFt"].mean().reset_index()
st.plotly_chart(
    px.density_heatmap(
        location_heatmap,
        x="State",
        y="Property_Type",
        z="Price_per_SqFt",
        histfunc="avg",
        title="Location-wise price-per-square-foot heatmap",
        labels={"Price_per_SqFt": "Average ₹ per sq ft"},
    ),
    width="stretch",
)
