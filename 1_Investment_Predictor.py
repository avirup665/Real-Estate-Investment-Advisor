from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import AMENITY_NAMES, APP_TITLE, REFERENCE_YEAR
from src.data_loader import load_processed_data
from src.inference import predict_property
from src.ui import hero, inject_css, model_disclaimer

st.set_page_config(page_title=f"Predictor | {APP_TITLE}", page_icon="🎯", layout="wide")
inject_css()
hero("🎯 Investment Predictor", "Enter property details to estimate investment quality and five-year value.")


df = load_processed_data()

state_options = sorted(df["State"].dropna().unique())
with st.form("property_form"):
    st.markdown("#### Location and property")
    c1, c2, c3, c4 = st.columns(4)
    state = c1.selectbox("State", state_options)
    city_options = sorted(df.loc[df["State"] == state, "City"].dropna().unique())
    city = c2.selectbox("City", city_options)
    locality_options = sorted(
        df.loc[(df["State"] == state) & (df["City"] == city), "Locality"].dropna().unique()
    )
    locality = c3.selectbox("Locality", locality_options)
    property_type = c4.selectbox("Property type", sorted(df["Property_Type"].unique()))

    st.markdown("#### Price and structure")
    c1, c2, c3, c4 = st.columns(4)
    bhk = c1.slider("BHK", 1, 5, 3)
    size = c2.number_input("Size (sq ft)", min_value=300, max_value=10000, value=1800, step=50)
    price = c3.number_input("Current price (₹ lakh)", min_value=5.0, max_value=2500.0, value=120.0, step=5.0)
    year_built = c4.slider("Year built", 1990, REFERENCE_YEAR, 2015)

    c1, c2, c3, c4 = st.columns(4)
    total_floors = c1.number_input("Total floors", min_value=1, max_value=100, value=10)
    floor_no = c2.number_input("Floor number", min_value=0, max_value=int(total_floors), value=min(4, int(total_floors)))
    furnished = c3.selectbox("Furnishing", sorted(df["Furnished_Status"].unique()))
    facing = c4.selectbox("Facing", sorted(df["Facing"].unique()))

    st.markdown("#### Connectivity and facilities")
    c1, c2, c3, c4 = st.columns(4)
    schools = c1.slider("Nearby schools", 0, 10, 5)
    hospitals = c2.slider("Nearby hospitals", 0, 10, 4)
    transport = c3.selectbox("Public transport", ["High", "Medium", "Low"])
    availability = c4.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])

    c1, c2, c3, c4 = st.columns(4)
    parking = c1.selectbox("Parking space", ["Yes", "No"])
    security = c2.selectbox("Security", ["Yes", "No"])
    owner_type = c3.selectbox("Owner type", sorted(df["Owner_Type"].unique()))
    amenities = c4.multiselect("Amenities", AMENITY_NAMES, default=["Gym", "Garden"])

    submitted = st.form_submit_button("Analyze investment", type="primary", width="stretch")

if submitted:
    record = {
        "ID": 0,
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Price_in_Lakhs": price,
        "Price_per_SqFt": price * 100000 / size,
        "Year_Built": year_built,
        "Furnished_Status": furnished,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Age_of_Property": REFERENCE_YEAR - year_built,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Amenities": ", ".join(amenities),
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
    }
    with st.spinner("Running classification and regression models..."):
        result = predict_property(record)

    if result["good_investment"]:
        st.markdown(
            f'<div class="result-good"><b>Recommended: Good Investment</b><br>'
            f'The model assigns a {result["good_investment_probability"]:.1%} probability of a positive investment classification.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-watch"><b>Proceed with caution</b><br>'
            f'The model assigns a {result["good_investment_probability"]:.1%} probability of a positive investment classification.</div>',
            unsafe_allow_html=True,
        )

    st.progress(int(result["good_investment_probability"] * 100), text="Good-investment probability")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Estimated price after 5 years", f"₹{result['future_price_lakhs']:,.2f} lakh")
    m2.metric("Estimated capital gain", f"₹{result['estimated_gain_lakhs']:,.2f} lakh")
    m3.metric("Estimated 5Y ROI", f"{result['roi_5y_percent']:.1f}%")
    m4.metric("Annualized return", f"{result['annualized_return_percent']:.1f}%")

    segment = df[(df["City"] == city) & (df["Locality"] == locality) & (df["Property_Type"] == property_type)]
    city_median = df.loc[df["City"] == city, "Price_per_SqFt"].median()
    segment_median = segment["Price_per_SqFt"].median() if not segment.empty else city_median
    offered_ppsf = price * 100000 / size
    st.markdown("#### Data-backed context")
    x1, x2, x3 = st.columns(3)
    x1.metric("Offered price per sq ft", f"₹{offered_ppsf:,.0f}")
    x2.metric("Local segment median", f"₹{segment_median:,.0f}", delta=f"{(offered_ppsf/segment_median-1):.1%}" if segment_median else None)
    x3.metric("Model confidence", f"{result['confidence']:.1%}")
    st.caption(f"Models used: {result['classifier_name']} and {result['regressor_name']}.")

model_disclaimer()
