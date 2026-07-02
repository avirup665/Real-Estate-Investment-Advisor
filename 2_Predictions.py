"""
2_Predictions.py

Real Estate Investment Advisor
Prediction Module
"""

# ==========================================================
# Imports
# ==========================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

import streamlit as st

warnings.filterwarnings("ignore")

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Predictions",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown(
    """
<style>

.main{
    background-color:#F8F9FA;
}

h1{
    color:#0F62FE;
}

h2{
    color:#2563EB;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:10px;
    padding:15px;
    border:1px solid #E5E7EB;
}

[data-testid="stSidebar"]{
    background:#111827;
}

[data-testid="stSidebar"] *{
    color:white;
}

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# Project Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "cleaned_data.csv"

CLASSIFIER_PATH = ROOT / "models" / "classifier.pkl"

REGRESSOR_PATH = ROOT / "models" / "regressor.pkl"

# ==========================================================
# Cached Loaders
# ==========================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        st.error("Dataset not found.")

        return None

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_classifier():

    if CLASSIFIER_PATH.exists():

        return joblib.load(CLASSIFIER_PATH)

    return None


@st.cache_resource
def load_regressor():

    if REGRESSOR_PATH.exists():

        return joblib.load(REGRESSOR_PATH)

    return None


# ==========================================================
# Load Resources
# ==========================================================

df = load_dataset()

classifier = load_classifier()

regressor = load_regressor()

# ==========================================================
# Page Header
# ==========================================================

st.title("🏡 Real Estate Prediction Center")

st.markdown(
"""
Predict whether a property is a **Good Investment**
and estimate its **Future Price (5 Years)** using the
trained Machine Learning models.
"""
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("Prediction Center")

st.sidebar.markdown("---")

st.sidebar.success("Resources")

if df is not None:
    st.sidebar.write("✅ Dataset Loaded")
else:
    st.sidebar.error("Dataset Missing")

if classifier is not None:
    st.sidebar.write("✅ Classifier Loaded")
else:
    st.sidebar.error("Classifier Missing")

if regressor is not None:
    st.sidebar.write("✅ Regressor Loaded")
else:
    st.sidebar.error("Regressor Missing")

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Prediction Workflow

1. Enter Property Details

2. Predict Investment Quality

3. Predict Future Price

4. Review Recommendation
"""
)

# ==========================================================
# Dataset Overview
# ==========================================================

if df is not None:

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Properties",
            f"{len(df):,}"
        )

    with c2:

        if "Price_in_Lakhs" in df.columns:

            st.metric(
                "Average Price",
                f"{df['Price_in_Lakhs'].mean():.2f}"
            )

    with c3:

        if "Future_Price_5Y" in df.columns:

            st.metric(
                "Future Avg Price",
                f"{df['Future_Price_5Y'].mean():.2f}"
            )

    with c4:

        if "State" in df.columns:

            st.metric(
                "States",
                df["State"].nunique()
            )

st.divider()

st.header("🏠 Property Information")

st.info(
"""
The property input form will appear below.

Enter the property details and click **Predict**
to generate:

- Investment Classification
- Future Price Prediction
- Confidence Score
- Investment Recommendation
"""
)

# ==========================================================
# Dynamic Property Input Form
# ==========================================================

if df is None:

    st.stop()

st.subheader("Enter Property Details")

with st.form("prediction_form"):

    # ------------------------------------------------------
    # Row 1
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        states = sorted(df["State"].dropna().unique())

        selected_state = st.selectbox(
            "🏠 State",
            states
        )

    with col2:

        cities = sorted(
            df.loc[
                df["State"] == selected_state,
                "City"
            ].dropna().unique()
        )

        selected_city = st.selectbox(
            "🏙 City",
            cities
        )

    # ------------------------------------------------------
    # Row 2
    # ------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        locality_options = sorted(
            df.loc[
                (df["State"] == selected_state) &
                (df["City"] == selected_city),
                "Locality"
            ].dropna().unique()
        )

        if not locality_options:
            locality_options = sorted(df["Locality"].dropna().unique())

        selected_locality = st.selectbox(
            "📍 Locality",
            locality_options
        )

    with col2:

        property_types = sorted(
            df["Property_Type"].dropna().unique()
        )

        selected_property_type = st.selectbox(
            "🏢 Property Type",
            property_types
        )

    # ------------------------------------------------------
    # Row 3
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        bhk_options = sorted(df["BHK"].dropna().unique())

        selected_bhk = st.selectbox(
            "🛏 BHK",
            bhk_options
        )

    with col2:

        size = st.number_input(
            "📐 Size (SqFt)",
            min_value=int(df["Size_in_SqFt"].min()),
            max_value=int(df["Size_in_SqFt"].max()),
            value=int(df["Size_in_SqFt"].median()),
            step=50
        )

    with col3:

        price = st.number_input(
            "💰 Price (Lakhs)",
            min_value=float(df["Price_in_Lakhs"].min()),
            max_value=float(df["Price_in_Lakhs"].max()),
            value=float(df["Price_in_Lakhs"].median()),
            step=1.0
        )

    # ------------------------------------------------------
    # Row 4
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        furnishing = st.selectbox(
            "🪑 Furnishing",
            sorted(df["Furnished_Status"].dropna().unique())
        )

    with col2:

        parking = st.selectbox(
            "🚗 Parking",
            sorted(df["Parking_Space"].dropna().drop_duplicates())
        )

    with col3:

        security = st.selectbox(
            "🛡 Security",
            sorted(df["Security"].dropna().unique())
        )

    # ------------------------------------------------------
    # Row 5
    # ------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        schools = st.number_input(
            "🏫 Nearby Schools",
            min_value=int(df["Nearby_Schools"].min()),
            max_value=int(df["Nearby_Schools"].max()),
            value=int(df["Nearby_Schools"].median())
        )

    with col2:

        hospitals = st.number_input(
            "🏥 Nearby Hospitals",
            min_value=int(df["Nearby_Hospitals"].min()),
            max_value=int(df["Nearby_Hospitals"].max()),
            value=int(df["Nearby_Hospitals"].median())
        )

    with col3:

        transport = st.selectbox(
            "🚇 Public Transport",
            sorted(
                df["Public_Transport_Accessibility"]
                .dropna()
                .unique()
            )
        )

    # ------------------------------------------------------
    # Submit Button
    # ------------------------------------------------------

    predict_button = st.form_submit_button(
        "🔮 Predict Property"
    )
# ==========================================================
# Build Prediction Data
# ==========================================================

if predict_button:

    # ------------------------------------------------------
    # Build Single Input Record
    # ------------------------------------------------------

    current_year = pd.Timestamp.now().year

    # Estimate defaults for fields not entered by the user
    median_year_built = int(df["Year_Built"].median()) \
        if "Year_Built" in df.columns else current_year - 10

    median_floor = int(df["Floor_No"].median()) \
        if "Floor_No" in df.columns else 1

    median_total_floors = int(df["Total_Floors"].median()) \
        if "Total_Floors" in df.columns else 5

    owner_type = (
        df["Owner_Type"].mode()[0]
        if "Owner_Type" in df.columns else "Owner"
    )

    availability = (
        df["Availability_Status"].mode()[0]
        if "Availability_Status" in df.columns else "Ready to Move"
    )

    facing = (
        df["Facing"].mode()[0]
        if "Facing" in df.columns else "East"
    )

    amenities = (
        df["Amenities"].mode()[0]
        if "Amenities" in df.columns else "Basic"
    )

    # ------------------------------------------------------
    # Derived Features
    # ------------------------------------------------------

    price_per_sqft = (
        price * 100000
    ) / size

    age = current_year - median_year_built

    # ------------------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------------------

    input_df = pd.DataFrame([{

        "State": selected_state,

        "City": selected_city,

        "Locality": selected_locality,

        "Property_Type": selected_property_type,

        "BHK": selected_bhk,

        "Size_in_SqFt": size,

        "Price_in_Lakhs": price,

        "Price_per_SqFt": price_per_sqft,

        "Year_Built": median_year_built,

        "Furnished_Status": furnishing,

        "Floor_No": median_floor,

        "Total_Floors": median_total_floors,

        "Age_of_Property": age,

        "Nearby_Schools": schools,

        "Nearby_Hospitals": hospitals,

        "Public_Transport_Accessibility": transport,

        "Parking_Space": parking,

        "Security": security,

        "Amenities": amenities,

        "Facing": facing,

        "Owner_Type": owner_type,

        "Availability_Status": availability

    }])

    # ------------------------------------------------------
    # Match Training Schema
    # ------------------------------------------------------

    if df is not None:

        training_columns = list(df.columns)

        # Remove target columns if present
        for target in [
            "Future_Price_5Y",
            "Good_Investment",
            "Investment_Score",
            "ID"
        ]:
            if target in training_columns:
                training_columns.remove(target)

        # Add any missing columns with defaults
        for col in training_columns:

            if col not in input_df.columns:

                if col in df.columns:

                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_df[col] = df[col].median()
                    else:
                        input_df[col] = df[col].mode()[0]

        # Reorder columns to match training
        input_df = input_df[training_columns]

    # ------------------------------------------------------
    # Preview Input
    # ------------------------------------------------------

    st.divider()

    st.subheader("📋 Prediction Input")

    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )
    
# ==========================================================
# Run Predictions
# ==========================================================

classification = None
future_price = None
confidence = None
roi = None

prediction_success = False

try:

    # ------------------------------------------------------
    # Classification
    # ------------------------------------------------------

    if classifier is not None:

        classification = classifier.predict(input_df)[0]

        # Probability (if supported)
        if hasattr(classifier, "predict_proba"):

            try:

                confidence = (
                    classifier
                    .predict_proba(input_df)[0]
                    .max()
                    * 100
                )

            except Exception:

                confidence = None

    # ------------------------------------------------------
    # Regression
    # ------------------------------------------------------

    if regressor is not None:

        future_price = float(
            regressor.predict(input_df)[0]
        )

    # ------------------------------------------------------
    # ROI Calculation
    # ------------------------------------------------------

    if future_price is not None:

        roi = (
            (
                future_price - price
            ) / price
        ) * 100

    prediction_success = True

except Exception as e:

    st.error(
        f"Prediction failed.\n\n{e}"
    )

    prediction_success = False
# ==========================================================
# Store Results
# ==========================================================

if prediction_success:

    prediction_result = {

        "Good_Investment": classification,

        "Future_Price_5Y": future_price,

        "Confidence": confidence,

        "ROI": roi

    }
# ==========================================================
# Convert Prediction
# ==========================================================

if prediction_success:

    if classification == 1:

        investment_label = "✅ Good Investment"

        recommendation = "Highly Recommended"

        recommendation_color = "green"

    else:

        investment_label = "❌ Not Recommended"

        recommendation = "Consider Alternatives"

        recommendation_color = "red"
        
# ==========================================================
# Prediction Summary
# ==========================================================

if prediction_success:

    st.divider()

    st.subheader("Prediction Completed")

    st.success(
        "Machine Learning models executed successfully."
    )
# ==========================================================
# Prediction Dashboard
# ==========================================================

if prediction_success:

    st.divider()

    st.header("🏆 Prediction Results")

    # ------------------------------------------------------
    # Result Cards
    # ------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if classification == 1:

            st.success("✅ GOOD INVESTMENT")

        else:

            st.error("❌ NOT RECOMMENDED")

    with col2:

        st.metric(

            "Future Price",

            f"₹ {future_price:.2f} Lakhs"

        )

    with col3:

        st.metric(

            "Estimated ROI",

            f"{roi:.2f}%"

        )

    with col4:

        if confidence is not None:

            st.metric(

                "Confidence",

                f"{confidence:.2f}%"

            )

        else:

            st.metric(

                "Confidence",

                "N/A"

            )

    # ------------------------------------------------------
    # Recommendation Panel
    # ------------------------------------------------------

    st.divider()

    if classification == 1:

        st.success(
            """
### 🎯 AI Recommendation

This property is predicted to be a **Good Investment**.

✔ Strong appreciation potential

✔ Positive future price growth

✔ Suitable for long-term investment
"""
        )

    else:

        st.warning(
            """
### ⚠ AI Recommendation

This property is **Not Recommended**.

• Expected appreciation is limited.

• Compare with higher-ranked properties.

• Consider alternative investment options.
"""
        )

    # ------------------------------------------------------
    # Price Comparison
    # ------------------------------------------------------

    st.divider()

    st.subheader("📉 Current vs Predicted Price")

    comparison = pd.DataFrame({

        "Stage": [

            "Current",

            "Predicted (5 Years)"

        ],

        "Price": [

            price,

            future_price

        ]

    })

    fig = px.bar(

        comparison,

        x="Stage",

        y="Price",

        color="Stage",

        text="Price",

        title="Price Comparison"

    )

    fig.update_traces(

        texttemplate="₹ %{text:.2f} L",

        textposition="outside"

    )

    fig.update_layout(

        template="plotly_white",

        height=500,

        showlegend=False,

        title_x=0.5,

        xaxis_title="",

        yaxis_title="Price (Lakhs)"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # ------------------------------------------------------
    # AI Insights
    # ------------------------------------------------------

    st.divider()

    st.subheader("💡 AI Investment Insights")

    appreciation = future_price - price

    if roi >= 30:

        roi_message = "Excellent return potential."

    elif roi >= 15:

        roi_message = "Good appreciation potential."

    elif roi >= 5:

        roi_message = "Moderate appreciation expected."

    else:

        roi_message = "Limited appreciation expected."

    st.info(
        f"""
### Investment Summary

- **Current Price:** ₹ {price:.2f} Lakhs
- **Predicted Future Price:** ₹ {future_price:.2f} Lakhs
- **Estimated Appreciation:** ₹ {appreciation:.2f} Lakhs
- **Estimated ROI:** {roi:.2f}%

**Assessment:** {roi_message}

The prediction is generated using the trained classification and regression models based on the property characteristics you provided.
"""
    )
# ==========================================================
# Prediction History
# ==========================================================

st.divider()

st.header("📝 Prediction History")

# ----------------------------------------------------------
# Initialize Session State
# ----------------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ----------------------------------------------------------
# Save Current Prediction
# ----------------------------------------------------------

if prediction_success:

    history_record = {
        "State": selected_state,
        "City": selected_city,
        "Locality": selected_locality,
        "Property Type": selected_property_type,
        "BHK": selected_bhk,
        "Size (SqFt)": size,
        "Current Price (Lakhs)": round(price, 2),
        "Predicted Price (Lakhs)": round(future_price, 2),
        "ROI (%)": round(roi, 2),
        "Investment": investment_label,
        "Confidence (%)": (
            round(confidence, 2)
            if confidence is not None
            else None
        )
    }

    # Avoid duplicate consecutive entries
    if (
        len(st.session_state.prediction_history) == 0
        or st.session_state.prediction_history[-1] != history_record
    ):
        st.session_state.prediction_history.append(history_record)

# ----------------------------------------------------------
# Display History
# ----------------------------------------------------------

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # Download CSV
    # ------------------------------------------------------

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ------------------------------------------------------
    # Clear History
    # ------------------------------------------------------

    if st.button(
        "🗑 Clear Prediction History",
        type="secondary"
    ):
        st.session_state.prediction_history.clear()
        st.rerun()

else:

    st.info(
        "No predictions have been made yet. "
        "Complete a prediction to build your history."
    )                                