"""
app.py

Real Estate Investment Advisor
Main Streamlit Application
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================

st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #F8F9FA;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0E1117;
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Metric Cards */
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}

/* Headings */
h1 {
    color: #1E3A8A;
}

h2 {
    color: #2563EB;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"

CLASSIFIER_PATH = BASE_DIR / "models" / "classifier.pkl"

REGRESSOR_PATH = BASE_DIR / "models" / "regressor.pkl"

# ==========================================================
# Cached Loaders
# ==========================================================

@st.cache_data
def load_dataset():

    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    return None


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
# Sidebar
# ==========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/609/609803.png",
    width=100
)

st.sidebar.title("🏡 Real Estate Advisor")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Analytics",
        "🤖 Prediction",
        "📈 Model Performance",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Backend Status")

if df is not None:
    st.sidebar.write("✅ Dataset Loaded")
else:
    st.sidebar.error("Dataset Missing")

if classifier is not None:
    st.sidebar.write("✅ Classifier Loaded")
else:
    st.sidebar.warning("Classifier Missing")

if regressor is not None:
    st.sidebar.write("✅ Regressor Loaded")
else:
    st.sidebar.warning("Regressor Missing")

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Real Estate Investment Advisor**

    Machine Learning Project

    • Classification

    • Regression

    • Analytics

    • Streamlit Dashboard
    """
)

# ==========================================================
# Header
# ==========================================================

st.title("🏡 Real Estate Investment Advisor")

st.markdown(
    """
Welcome to the **Real Estate Investment Advisor Dashboard**.

This application helps users:

- Predict future property prices.
- Identify good investment opportunities.
- Explore real estate trends.
- Compare machine learning models.
- Analyze the housing dataset.
"""
)
# ==========================================================
# Dashboard Home
# ==========================================================

if menu == "🏠 Dashboard":

    st.header("📊 Executive Dashboard")

    if df is None:

        st.error("Dataset not found.")

        st.stop()

    # ------------------------------------------
    # Dataset Statistics
    # ------------------------------------------

    total_properties = len(df)

    avg_price = 0

    avg_size = 0

    avg_future_price = 0

    total_states = 0

    if "Price_in_Lakhs" in df.columns:

        avg_price = df["Price_in_Lakhs"].mean()

    if "Size_in_SqFt" in df.columns:

        avg_size = df["Size_in_SqFt"].mean()

    if "Future_Price_5Y" in df.columns:

        avg_future_price = df["Future_Price_5Y"].mean()

    if "State" in df.columns:

        total_states = df["State"].nunique()

    # ------------------------------------------
    # KPI Cards
    # ------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(

            label="🏠 Total Properties",

            value=f"{total_properties:,}"

        )

    with col2:

        st.metric(

            label="💰 Avg Price (Lakhs)",

            value=f"{avg_price:.2f}"

        )

    with col3:

        st.metric(

            label="📐 Avg Size (SqFt)",

            value=f"{avg_size:.0f}"

        )

    with col4:

        st.metric(

            label="🌍 States",

            value=total_states

        )

    st.write("")

    col5, col6 = st.columns(2)

    with col5:

        st.metric(

            label="📈 Future Avg Price",

            value=f"{avg_future_price:.2f}"

        )

    with col6:

        if classifier is not None and regressor is not None:

            st.success("✅ ML Models Loaded")

        else:

            st.warning("⚠ Some models are missing.")

    st.divider()

    # ------------------------------------------
    # Dataset Preview
    # ------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(

        df.head(10),

        use_container_width=True

    )

    st.divider()

    # ------------------------------------------
    # Dataset Information
    # ------------------------------------------

    info1, info2 = st.columns(2)

    with info1:

        st.subheader("Rows & Columns")

        st.write(f"Rows : {df.shape[0]}")

        st.write(f"Columns : {df.shape[1]}")

    with info2:

        st.subheader("Missing Values")

        missing = df.isnull().sum().sum()

        st.write(f"Total Missing Values : {missing}")

    st.divider()

    st.success("Dashboard Loaded Successfully.")

# ==========================================================
# Interactive Filters
# ==========================================================

if menu == "🏠 Dashboard":

    st.divider()

    st.subheader("🔍 Interactive Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    # -----------------------------
    # State Filter
    # -----------------------------

    with filter_col1:

        if "State" in df.columns:

            states = ["All"] + sorted(df["State"].dropna().unique().tolist())

            selected_state = st.selectbox(
                "Select State",
                states
            )

        else:

            selected_state = "All"

    # -----------------------------
    # Property Type Filter
    # -----------------------------

    with filter_col2:

        if "Property_Type" in df.columns:

            property_types = ["All"] + sorted(
                df["Property_Type"]
                .dropna()
                .unique()
                .tolist()
            )

            selected_property = st.selectbox(
                "Property Type",
                property_types
            )

        else:

            selected_property = "All"

    # -----------------------------
    # BHK Filter
    # -----------------------------

    with filter_col3:

        if "BHK" in df.columns:

            bhk_values = sorted(df["BHK"].dropna().unique())

            selected_bhk = st.selectbox(
                "BHK",
                ["All"] + list(bhk_values)
            )

        else:

            selected_bhk = "All"

# ==========================================================
# Apply Filters
# ==========================================================

filtered_df = df.copy()

# -----------------------------
# State
# -----------------------------

if selected_state != "All":

    filtered_df = filtered_df[
        filtered_df["State"] == selected_state
    ]

# -----------------------------
# Property Type
# -----------------------------

if selected_property != "All":

    filtered_df = filtered_df[
        filtered_df["Property_Type"] == selected_property
    ]

# -----------------------------
# BHK
# -----------------------------

if selected_bhk != "All":

    filtered_df = filtered_df[
        filtered_df["BHK"] == selected_bhk
    ]

st.success(
    f"Showing {len(filtered_df):,} properties"
)

st.divider()

summary1, summary2, summary3 = st.columns(3)

with summary1:

    st.metric(
        "Filtered Properties",
        len(filtered_df)
    )

with summary2:

    if "Price_in_Lakhs" in filtered_df.columns:

        st.metric(
            "Average Price",
            f"{filtered_df['Price_in_Lakhs'].mean():.2f} Lakhs"
        )

with summary3:

    if "Size_in_SqFt" in filtered_df.columns:

        st.metric(
            "Average Size",
            f"{filtered_df['Size_in_SqFt'].mean():.0f} SqFt"
        )
        
# ==========================================================
# Property Price Distribution
# ==========================================================

st.divider()

st.subheader("📈 Property Price Distribution")

if "Price_in_Lakhs" in filtered_df.columns:

    fig_hist = px.histogram(

        filtered_df,

        x="Price_in_Lakhs",

        nbins=40,

        marginal="box",

        opacity=0.85,

        title="Distribution of Property Prices"

    )

    fig_hist.update_layout(

        template="plotly_white",

        height=500,

        title_x=0.5,

        xaxis_title="Price (Lakhs)",

        yaxis_title="Number of Properties",

        legend_title="",

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=20

        )

    )

    fig_hist.update_traces(

        hovertemplate=(
            "<b>Price:</b> %{x:.2f} Lakhs<br>"
            "<b>Count:</b> %{y}<extra></extra>"
        )

    )

    st.plotly_chart(

        fig_hist,

        use_container_width=True

    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(

            "Minimum",

            f"{filtered_df['Price_in_Lakhs'].min():.2f}"

        )

    with c2:

        st.metric(

            "Average",

            f"{filtered_df['Price_in_Lakhs'].mean():.2f}"

        )

    with c3:

        st.metric(

            "Median",

            f"{filtered_df['Price_in_Lakhs'].median():.2f}"

        )

    with c4:

        st.metric(

            "Maximum",

            f"{filtered_df['Price_in_Lakhs'].max():.2f}"

        )

else:

    st.warning(
        "Price_in_Lakhs column not found."
    )
    
      # ==========================================================
# State-wise Average Property Price
# ==========================================================

st.divider()

st.subheader("🏙 State-wise Average Property Price")

if (
    "State" in filtered_df.columns
    and "Price_in_Lakhs" in filtered_df.columns
):

    state_price = (
        filtered_df
        .groupby("State", as_index=False)["Price_in_Lakhs"]
        .mean()
        .sort_values(
            by="Price_in_Lakhs",
            ascending=False
        )
    )

    # Allow user to limit number of states displayed
    top_n = st.slider(
        "Number of States to Display",
        min_value=5,
        max_value=min(30, len(state_price)),
        value=min(10, len(state_price)),
        step=1
    )

    state_price = state_price.head(top_n)

    fig_state = px.bar(
        state_price,
        x="State",
        y="Price_in_Lakhs",
        color="Price_in_Lakhs",
        text="Price_in_Lakhs",
        title="Average Property Price by State"
    )

    fig_state.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>State:</b> %{x}<br>"
            "<b>Average Price:</b> %{y:.2f} Lakhs"
            "<extra></extra>"
        )
    )

    fig_state.update_layout(
        template="plotly_white",
        height=550,
        title_x=0.5,
        xaxis_title="State",
        yaxis_title="Average Price (Lakhs)",
        coloraxis_showscale=False,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=40
        )
    )

    st.plotly_chart(
        fig_state,
        use_container_width=True
    )

    st.dataframe(
        state_price.rename(
            columns={
                "Price_in_Lakhs": "Average Price (Lakhs)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Required columns ('State' and 'Price_in_Lakhs') "
        "are not available in the dataset."
    )
    
# ==========================================================
# Property Type Distribution
# ==========================================================

st.divider()

st.subheader("🏠 Property Type Distribution")

if "Property_Type" in filtered_df.columns:

    property_counts = (
        filtered_df["Property_Type"]
        .value_counts()
        .reset_index()
    )

    property_counts.columns = [
        "Property_Type",
        "Count"
    ]

    fig_pie = px.pie(

        property_counts,

        names="Property_Type",

        values="Count",

        hole=0.45,

        title="Distribution of Property Types"

    )

    fig_pie.update_traces(

        textposition="inside",

        textinfo="percent+label",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Properties: %{value}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )

    )

    fig_pie.update_layout(

        template="plotly_white",

        height=550,

        title_x=0.5,

        legend_title="Property Type",

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )

    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    st.subheader("Property Type Summary")

    summary = property_counts.copy()

    summary["Percentage"] = (
        summary["Count"]
        / summary["Count"].sum()
        * 100
    ).round(2)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Property_Type column not found."
    )                          
    
# ==========================================================
# BHK Distribution
# ==========================================================

st.divider()

st.subheader("🛏 BHK Distribution")

if "BHK" in filtered_df.columns:

    # ----------------------------------------
    # Count properties by BHK
    # ----------------------------------------

    bhk_df = (
        filtered_df
        .groupby("BHK")
        .size()
        .reset_index(name="Count")
        .sort_values("BHK")
    )

    fig_bhk = px.bar(

        bhk_df,

        x="BHK",

        y="Count",

        color="Count",

        text="Count",

        title="Distribution of Properties by BHK"

    )

    fig_bhk.update_traces(

        textposition="outside",

        hovertemplate=(
            "<b>%{x} BHK</b><br>"
            "Properties : %{y}"
            "<extra></extra>"
        )

    )

    fig_bhk.update_layout(

        template="plotly_white",

        height=550,

        title_x=0.5,

        xaxis_title="BHK",

        yaxis_title="Number of Properties",

        coloraxis_showscale=False,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        )

    )

    st.plotly_chart(
        fig_bhk,
        use_container_width=True
    )

    # ----------------------------------------
    # Summary Table
    # ----------------------------------------

    st.subheader("BHK Summary")

    summary = bhk_df.copy()

    summary["Percentage"] = (
        summary["Count"]
        / summary["Count"].sum()
        * 100
    ).round(2)

    summary.columns = [

        "BHK",

        "Properties",

        "Percentage (%)"

    ]

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    # ----------------------------------------
    # Key Statistics
    # ----------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Most Common BHK",

            int(
                bhk_df.loc[
                    bhk_df["Count"].idxmax(),
                    "BHK"
                ]
            )

        )

    with col2:

        st.metric(

            "Highest Count",

            int(
                bhk_df["Count"].max()
            )

        )

    with col3:

        st.metric(

            "Available BHK Types",

            bhk_df.shape[0]

        )

else:

    st.warning(
        "BHK column not found."
    )
    
# ==========================================================
# Price vs Size Scatter Plot
# ==========================================================

st.divider()

st.subheader("📐 Price vs Size Analysis")

required_columns = ["Size_in_SqFt", "Price_in_Lakhs"]

if all(col in filtered_df.columns for col in required_columns):

    # ------------------------------------------
    # Color By Option
    # ------------------------------------------

    color_options = ["None"]

    if "Property_Type" in filtered_df.columns:
        color_options.append("Property_Type")

    if "Good_Investment" in filtered_df.columns:
        color_options.append("Good_Investment")

    if "State" in filtered_df.columns:
        color_options.append("State")

    color_by = st.selectbox(
        "Color Points By",
        color_options,
        index=0
    )

    # ------------------------------------------
    # Prepare Hover Data
    # ------------------------------------------

    hover_columns = []

    for col in [
        "State",
        "City",
        "Property_Type",
        "BHK",
        "Future_Price_5Y",
        "Investment_Score"
    ]:
        if col in filtered_df.columns:
            hover_columns.append(col)

    # ------------------------------------------
    # Create Scatter Plot
    # ------------------------------------------

    if color_by == "None":

        fig_scatter = px.scatter(

            filtered_df,

            x="Size_in_SqFt",

            y="Price_in_Lakhs",

            hover_data=hover_columns,

            opacity=0.75,

            title="Property Size vs Property Price"

        )

    else:

        fig_scatter = px.scatter(

            filtered_df,

            x="Size_in_SqFt",

            y="Price_in_Lakhs",

            color=color_by,

            hover_data=hover_columns,

            opacity=0.75,

            title=f"Property Size vs Price (Colored by {color_by})"

        )

    # ------------------------------------------
    # Layout
    # ------------------------------------------

    fig_scatter.update_layout(

        template="plotly_white",

        height=600,

        title_x=0.5,

        xaxis_title="Property Size (SqFt)",

        yaxis_title="Price (Lakhs)",

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=30
        )

    )

    fig_scatter.update_traces(

        marker=dict(
            size=8,
            line=dict(width=0.5)
        ),

        hovertemplate=
        "<b>Size:</b> %{x:.0f} SqFt<br>"
        "<b>Price:</b> %{y:.2f} Lakhs"
        "<extra></extra>"

    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    # ------------------------------------------
    # Statistics
    # ------------------------------------------

    stats1, stats2, stats3 = st.columns(3)

    with stats1:

        st.metric(
            "Average Size",
            f"{filtered_df['Size_in_SqFt'].mean():.0f} SqFt"
        )

    with stats2:

        st.metric(
            "Average Price",
            f"{filtered_df['Price_in_Lakhs'].mean():.2f} Lakhs"
        )

    with stats3:

        correlation = filtered_df[
            ["Size_in_SqFt", "Price_in_Lakhs"]
        ].corr().iloc[0, 1]

        st.metric(
            "Correlation",
            f"{correlation:.2f}"
        )

else:

    st.warning(
        "Required columns "
        "'Size_in_SqFt' and "
        "'Price_in_Lakhs' are missing."
    )
    
# ==========================================================
# Correlation Heatmap
# ==========================================================

st.divider()

st.subheader("🔥 Correlation Heatmap")

# ------------------------------------------
# Select Numeric Columns
# ------------------------------------------

numeric_df = filtered_df.select_dtypes(include=["number"])

if numeric_df.shape[1] >= 2:

    # --------------------------------------
    # Correlation Matrix
    # --------------------------------------

    corr_matrix = numeric_df.corr(numeric_only=True).round(2)

    # --------------------------------------
    # Plotly Heatmap
    # --------------------------------------

    fig_heatmap = px.imshow(

        corr_matrix,

        text_auto=True,

        aspect="auto",

        color_continuous_scale="RdBu_r",

        origin="lower",

        title="Correlation Between Numerical Features"

    )

    fig_heatmap.update_layout(

        template="plotly_white",

        height=750,

        title_x=0.5,

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=40
        ),

        coloraxis_colorbar=dict(
            title="Correlation"
        )

    )

    fig_heatmap.update_xaxes(

        tickangle=45,

        side="bottom"

    )

    fig_heatmap.update_yaxes(

        autorange="reversed"

    )

    st.plotly_chart(

        fig_heatmap,

        use_container_width=True

    )

    # --------------------------------------
    # Strongest Correlations
    # --------------------------------------

    st.subheader("Top Feature Correlations")

    corr_pairs = (
        corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1
            ).astype(bool)
        )
        .stack()
        .reset_index()
    )

    corr_pairs.columns = [

        "Feature 1",

        "Feature 2",

        "Correlation"

    ]

    corr_pairs["Absolute"] = (
        corr_pairs["Correlation"].abs()
    )

    corr_pairs = (
        corr_pairs
        .sort_values(
            "Absolute",
            ascending=False
        )
        .drop(columns="Absolute")
    )

    st.dataframe(

        corr_pairs.head(15),

        use_container_width=True,

        hide_index=True

    )

else:

    st.warning(

        "Not enough numerical columns to compute a correlation matrix."

    )
    
# ==========================================================
# Feature Importance
# Model Detection
# ==========================================================

st.divider()

st.subheader("📊 Feature Importance")

# ----------------------------------------
# Choose Available Model
# ----------------------------------------

selected_pipeline = None
selected_name = None

if classifier is not None:

    selected_pipeline = classifier
    selected_name = "Classifier"

elif regressor is not None:

    selected_pipeline = regressor
    selected_name = "Regressor"

else:

    st.info("No trained model available.")

# ----------------------------------------
# Continue only if a model exists
# ----------------------------------------

if selected_pipeline is not None:

    estimator = None

    preprocessor = None

    # ------------------------------------
    # Detect sklearn Pipeline
    # ------------------------------------

    if hasattr(selected_pipeline, "named_steps"):

        if "classifier" in selected_pipeline.named_steps:

            estimator = selected_pipeline.named_steps["classifier"]

        elif "regressor" in selected_pipeline.named_steps:

            estimator = selected_pipeline.named_steps["regressor"]

        if "preprocessor" in selected_pipeline.named_steps:

            preprocessor = selected_pipeline.named_steps["preprocessor"]

    else:

        estimator = selected_pipeline

    # ------------------------------------
    # Recover Feature Names
    # ------------------------------------

    feature_names = []

    if preprocessor is not None:

        try:

            feature_names = (
                preprocessor.get_feature_names_out()
                .tolist()
            )

        except Exception:

            feature_names = []

    # ------------------------------------
    # Extract Feature Importance
    # ------------------------------------

    importance = None

    if hasattr(estimator, "feature_importances_"):

        importance = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):

        coef = estimator.coef_

        if coef.ndim > 1:

            importance = np.abs(coef).mean(axis=0)

        else:

            importance = np.abs(coef)

    # ------------------------------------
    # Validate
    # ------------------------------------

    if importance is None:

        st.info(
            f"{type(estimator).__name__} "
            "does not expose feature importance."
        )

    elif len(feature_names) != len(importance):

        st.warning(
            "Feature names could not be matched "
            "with model importance."
        )

    else:

        feature_importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importance

        })

        feature_importance_df = (
            feature_importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )
# ==========================================================
# Prepare Top Feature Importance Data
# ==========================================================

if "feature_importance_df" in locals():

    # ------------------------------------------
    # Remove Invalid Values
    # ------------------------------------------

    feature_importance_df = (
        feature_importance_df
        .dropna(subset=["Importance"])
        .copy()
    )

    feature_importance_df = feature_importance_df[
        feature_importance_df["Importance"] >= 0
    ]

    # ------------------------------------------
    # Sort Descending
    # ------------------------------------------

    feature_importance_df.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )

    feature_importance_df.reset_index(
        drop=True,
        inplace=True
    )

    # ------------------------------------------
    # Select Top Features
    # ------------------------------------------

    max_features = min(
        20,
        len(feature_importance_df)
    )

    top_n = st.slider(

        "Number of Top Features",

        min_value=5,

        max_value=max_features,

        value=max_features,

        step=1,

        key="feature_slider"

    )

    top_features = (
        feature_importance_df
        .head(top_n)
        .copy()
    )

    # ------------------------------------------
    # Round Importance
    # ------------------------------------------

    top_features["Importance"] = (
        top_features["Importance"]
        .round(6)
    )

    # ------------------------------------------
    # Ranking
    # ------------------------------------------

    top_features.insert(

        0,

        "Rank",

        range(
            1,
            len(top_features) + 1
        )

    )

    # ------------------------------------------
    # Short Display Labels
    # ------------------------------------------

    top_features["Display Feature"] = (
        top_features["Feature"]
        .astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    # ------------------------------------------
    # Normalize Importance
    # ------------------------------------------

    max_imp = top_features["Importance"].max()

    if max_imp > 0:

        top_features["Relative Importance"] = (
            top_features["Importance"]
            / max_imp
            * 100
        ).round(2)

    else:

        top_features["Relative Importance"] = 0

    st.success(
        f"Displaying Top {len(top_features)} Important Features"
    )
    
# ==========================================================
# Feature Importance Chart
# ==========================================================

st.subheader("📊 Top Feature Importance")

if len(top_features) > 0:

    # ------------------------------------------
    # Highest importance at top
    # ------------------------------------------

    plot_df = (
        top_features
        .sort_values(
            "Importance",
            ascending=True
        )
        .copy()
    )

    # ------------------------------------------
    # Horizontal Bar Chart
    # ------------------------------------------

    fig_importance = px.bar(

        plot_df,

        x="Importance",

        y="Display Feature",

        orientation="h",

        color="Relative Importance",

        color_continuous_scale="Viridis",

        text="Importance",

        title="Most Important Features"

    )

    # ------------------------------------------
    # Hover Information
    # ------------------------------------------

    fig_importance.update_traces(

        texttemplate="%{text:.4f}",

        textposition="outside",

        hovertemplate=
        "<b>%{y}</b><br>"
        "Importance : %{x:.6f}<br>"
        "Relative : %{marker.color:.2f}%"
        "<extra></extra>"

    )

    # ------------------------------------------
    # Layout
    # ------------------------------------------

    fig_importance.update_layout(

        template="plotly_white",

        height=max(
            500,
            len(plot_df) * 30
        ),

        title_x=0.5,

        xaxis_title="Feature Importance",

        yaxis_title="",

        coloraxis_colorbar=dict(
            title="Relative (%)"
        ),

        margin=dict(
            l=40,
            r=30,
            t=60,
            b=30
        ),

        yaxis=dict(
            automargin=True
        )

    )

    st.plotly_chart(

        fig_importance,

        use_container_width=True

    )

else:

    st.warning(
        "No feature importance data available."
    )
    
# ==========================================================
# Feature Importance Summary
# ==========================================================

if len(top_features) > 0:

    st.divider()

    st.subheader("🏆 Feature Importance Summary")

    # -------------------------------------------------------
    # KPI Cards
    # -------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Most Important Feature",
            top_features.iloc[0]["Display Feature"]
        )

    with col2:

        st.metric(
            "Highest Importance",
            f"{top_features.iloc[0]['Importance']:.4f}"
        )

    with col3:

        st.metric(
            "Average Importance",
            f"{top_features['Importance'].mean():.4f}"
        )

    with col4:

        st.metric(
            "Features Displayed",
            len(top_features)
        )

    st.write("")

    # -------------------------------------------------------
    # Detailed Table
    # -------------------------------------------------------

    st.subheader("📋 Top Features Table")

    display_df = top_features[
        [
            "Rank",
            "Display Feature",
            "Importance",
            "Relative Importance"
        ]
    ].copy()

    display_df.columns = [

        "Rank",

        "Feature",

        "Importance",

        "Relative (%)"

    ]

    st.dataframe(

        display_df,

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------------
    # Download Button
    # -------------------------------------------------------

    csv = display_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Feature Importance CSV",

        data=csv,

        file_name="feature_importance.csv",

        mime="text/csv"

    )

    # -------------------------------------------------------
    # Insights Panel
    # -------------------------------------------------------

    st.info(
        f"""
### 📈 Model Insights

• **Top Feature:** {top_features.iloc[0]['Display Feature']}

• **Importance Score:** {top_features.iloc[0]['Importance']:.4f}

• **Average Feature Importance:** {top_features['Importance'].mean():.4f}

• Showing **Top {len(top_features)}** most influential features used by the trained model.

Higher importance values indicate that the model relied more heavily on these features when making predictions.
"""
    )

else:

    st.warning(
        "Feature importance information is unavailable for the selected model."
    )
# ==========================================================
# Investment Highlights
# ==========================================================

st.divider()

st.header("🏆 Investment Highlights")

if filtered_df is not None and len(filtered_df) > 0:

    # -------------------------------------------------------
    # Highest Investment Score
    # -------------------------------------------------------

    if "Investment_Score" in filtered_df.columns:

        best_score_row = filtered_df.loc[
            filtered_df["Investment_Score"].idxmax()
        ]

        highest_score = best_score_row["Investment_Score"]

    else:

        best_score_row = None

        highest_score = 0

    # -------------------------------------------------------
    # Highest Future Price
    # -------------------------------------------------------

    if "Future_Price_5Y" in filtered_df.columns:

        best_future_row = filtered_df.loc[
            filtered_df["Future_Price_5Y"].idxmax()
        ]

        highest_future_price = best_future_row["Future_Price_5Y"]

    else:

        best_future_row = None

        highest_future_price = 0

    # -------------------------------------------------------
    # Highest ROI
    # -------------------------------------------------------

    highest_roi = 0

    roi_row = None

    if (
        "Future_Price_5Y" in filtered_df.columns
        and
        "Price_in_Lakhs" in filtered_df.columns
    ):

        roi_df = filtered_df.copy()

        roi_df["ROI"] = (
            (
                roi_df["Future_Price_5Y"]
                -
                roi_df["Price_in_Lakhs"]
            )
            /
            roi_df["Price_in_Lakhs"]
        ) * 100

        roi_df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True
        )

        roi_df.dropna(
            subset=["ROI"],
            inplace=True
        )

        if len(roi_df):

            roi_row = roi_df.loc[
                roi_df["ROI"].idxmax()
            ]

            highest_roi = roi_row["ROI"]

    # -------------------------------------------------------
    # Good Investments
    # -------------------------------------------------------

    total_good = 0

    if "Good_Investment" in filtered_df.columns:

        total_good = int(
            filtered_df["Good_Investment"].sum()
        )

    # -------------------------------------------------------
    # KPI Cards
    # -------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "⭐ Highest Investment Score",
            f"{highest_score:.2f}"
        )

        if best_score_row is not None:

            st.caption(
                f"{best_score_row.get('City','')} | "
                f"{best_score_row.get('Property_Type','')}"
            )

    with c2:

        st.metric(
            "📈 Highest Future Price",
            f"{highest_future_price:.2f} Lakhs"
        )

        if best_future_row is not None:

            st.caption(
                f"{best_future_row.get('City','')} | "
                f"{best_future_row.get('Property_Type','')}"
            )

    with c3:

        st.metric(
            "💰 Highest ROI",
            f"{highest_roi:.2f}%"
        )

        if roi_row is not None:

            st.caption(
                f"{roi_row.get('City','')} | "
                f"{roi_row.get('Property_Type','')}"
            )

    with c4:

        st.metric(
            "✅ Good Investments",
            f"{total_good:,}"
        )

        if len(filtered_df):

            pct = (
                total_good
                /
                len(filtered_df)
            ) * 100

            st.caption(
                f"{pct:.1f}% of properties"
            )

else:

    st.warning(
        "No data available."
    )
# ==========================================================
# Top Investment Opportunities
# ==========================================================

st.divider()

st.header("🏆 Top Investment Opportunities")

required_columns = [
    "Investment_Score",
    "Price_in_Lakhs",
    "Future_Price_5Y"
]

if all(col in filtered_df.columns for col in required_columns):

    investment_df = filtered_df.copy()

    # -------------------------------------------------------
    # ROI Calculation
    # -------------------------------------------------------

    investment_df["ROI (%)"] = (
        (
            investment_df["Future_Price_5Y"]
            -
            investment_df["Price_in_Lakhs"]
        )
        /
        investment_df["Price_in_Lakhs"]
    ) * 100

    investment_df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    investment_df.dropna(
        subset=["ROI (%)"],
        inplace=True
    )

    # -------------------------------------------------------
    # Sort by Investment Score
    # -------------------------------------------------------

    investment_df = (
        investment_df
        .sort_values(
            by="Investment_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------
    # Rank
    # -------------------------------------------------------

    investment_df.insert(
        0,
        "Rank",
        range(1, len(investment_df) + 1)
    )

    # -------------------------------------------------------
    # Top N Selector
    # -------------------------------------------------------

    max_rows = min(50, len(investment_df))

    top_n = st.slider(
        "Top Investment Opportunities",
        min_value=5,
        max_value=max_rows,
        value=min(25, max_rows),
        step=5,
        key="leaderboard_slider"
    )

    leaderboard = investment_df.head(top_n).copy()

    # -------------------------------------------------------
    # Columns to Display
    # -------------------------------------------------------

    display_columns = []

    preferred_columns = [

        "Rank",

        "State",

        "City",

        "Locality",

        "Property_Type",

        "BHK",

        "Price_in_Lakhs",

        "Future_Price_5Y",

        "ROI (%)",

        "Investment_Score"

    ]

    for col in preferred_columns:

        if col in leaderboard.columns:

            display_columns.append(col)

    leaderboard = leaderboard[display_columns]

    # -------------------------------------------------------
    # Formatting
    # -------------------------------------------------------

    formatter = {}

    if "Price_in_Lakhs" in leaderboard.columns:
        formatter["Price_in_Lakhs"] = "{:.2f}"

    if "Future_Price_5Y" in leaderboard.columns:
        formatter["Future_Price_5Y"] = "{:.2f}"

    if "Investment_Score" in leaderboard.columns:
        formatter["Investment_Score"] = "{:.2f}"

    if "ROI (%)" in leaderboard.columns:
        formatter["ROI (%)"] = "{:.2f}%"

    st.dataframe(

        leaderboard.style
        .format(formatter)
        .background_gradient(
            subset=["Investment_Score"],
            cmap="Greens"
        )
        .background_gradient(
            subset=["ROI (%)"],
            cmap="Blues"
        ),

        use_container_width=True,

        hide_index=True

    )

    # -------------------------------------------------------
    # Quick Summary
    # -------------------------------------------------------

    st.success(
        f"Showing Top {len(leaderboard)} Investment Opportunities "
        f"ranked by Investment Score."
    )

else:

    st.warning(
        "Investment Score or Future Price columns are missing."
    )
    
# ==========================================================
# Investment Score Distribution
# ==========================================================

st.divider()

st.header("📊 Investment Score Distribution")

if "Investment_Score" in filtered_df.columns:

    fig_score = px.histogram(

        filtered_df,

        x="Investment_Score",

        nbins=30,

        marginal="box",

        color_discrete_sequence=["#2E8B57"],

        title="Distribution of Investment Scores"

    )

    fig_score.update_layout(

        template="plotly_white",

        height=550,

        title_x=0.5,

        xaxis_title="Investment Score",

        yaxis_title="Number of Properties",

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=30

        )

    )

    fig_score.update_traces(

        hovertemplate=

        "<b>Investment Score</b>: %{x:.2f}<br>"

        "<b>Properties</b>: %{y}"

        "<extra></extra>"

    )

    st.plotly_chart(

        fig_score,

        use_container_width=True

    )

else:

    st.warning(

        "Investment Score column not found."

    )

# ==========================================================
# Summary Insights
# ==========================================================

st.divider()

st.header("🎯 Investment Insights")

if len(filtered_df):

    col1, col2, col3 = st.columns(3)

    # -------------------------------------------------------
    # Average Investment Score
    # -------------------------------------------------------

    if "Investment_Score" in filtered_df.columns:

        avg_score = filtered_df["Investment_Score"].mean()

        with col1:

            st.metric(

                "Average Score",

                f"{avg_score:.2f}"

            )

    # -------------------------------------------------------
    # Median ROI
    # -------------------------------------------------------

    if (

        "Future_Price_5Y" in filtered_df.columns

        and

        "Price_in_Lakhs" in filtered_df.columns

    ):

        roi = (

            (

                filtered_df["Future_Price_5Y"]

                -

                filtered_df["Price_in_Lakhs"]

            )

            /

            filtered_df["Price_in_Lakhs"]

        ) * 100

        median_roi = roi.median()

        with col2:

            st.metric(

                "Median ROI",

                f"{median_roi:.2f}%"

            )

    # -------------------------------------------------------
    # Best Performing State
    # -------------------------------------------------------

    if (

        "State" in filtered_df.columns

        and

        "Investment_Score" in filtered_df.columns

    ):

        best_state = (

            filtered_df

            .groupby("State")["Investment_Score"]

            .mean()

            .sort_values(ascending=False)

        )

        if len(best_state):

            with col3:

                st.metric(

                    "Best State",

                    best_state.index[0]

                )

    st.info(

f"""

### 📈 Portfolio Summary

• Properties Analysed : **{len(filtered_df):,}**

• Average Investment Score : **{filtered_df['Investment_Score'].mean():.2f}**

• Average Future Price : **{filtered_df['Future_Price_5Y'].mean():.2f} Lakhs**

• Average Current Price : **{filtered_df['Price_in_Lakhs'].mean():.2f} Lakhs**

• Potential Appreciation :

**{((filtered_df['Future_Price_5Y'].mean()-filtered_df['Price_in_Lakhs'].mean())/filtered_df['Price_in_Lakhs'].mean()*100):.2f}%**

This dashboard ranks properties based on the trained machine learning model's Investment Score and projected 5-year appreciation.

"""

    )

else:

    st.warning(

        "No properties available."

    )
# ==========================================================
# Downloads
# ==========================================================

st.divider()

st.header("📥 Export Dashboard Data")

download_col1, download_col2, download_col3 = st.columns(3)

# ----------------------------------------------------------
# Filtered Dataset
# ----------------------------------------------------------

with download_col1:

    csv_dataset = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="📥 Download Filtered Dataset",

        data=csv_dataset,

        file_name="filtered_real_estate_data.csv",

        mime="text/csv",

        use_container_width=True

    )

# ----------------------------------------------------------
# Leaderboard
# ----------------------------------------------------------

with download_col2:

    if "leaderboard" in locals():

        csv_leaderboard = leaderboard.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="🏆 Download Leaderboard",

            data=csv_leaderboard,

            file_name="investment_leaderboard.csv",

            mime="text/csv",

            use_container_width=True

        )

# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------

with download_col3:

    if "top_features" in locals():

        csv_features = top_features.to_csv(index=False).encode("utf-8")

        st.download_button(

            label="📊 Download Feature Importance",

            data=csv_features,

            file_name="feature_importance.csv",

            mime="text/csv",

            use_container_width=True

        )

# ==========================================================
# Model Information
# ==========================================================

st.divider()

st.header("🤖 Machine Learning Information")

info1, info2 = st.columns(2)

with info1:

    st.markdown("""
### Classification Model

- Target : **Good_Investment**
- Purpose : Investment Classification
- Pipeline : ColumnTransformer
- Missing Value Handling : SimpleImputer
- Feature Scaling : StandardScaler
- Encoding : OneHotEncoder
""")

with info2:

    st.markdown("""
### Regression Model

- Target : **Future_Price_5Y**
- Purpose : Future Price Prediction
- Pipeline : ColumnTransformer
- Missing Value Handling : SimpleImputer
- Feature Scaling : StandardScaler
- Encoding : OneHotEncoder
""")

# ==========================================================
# Dataset Information
# ==========================================================

st.divider()

st.header("📅 Dataset Information")

dataset_info = pd.DataFrame({

    "Metric": [

        "Total Records",

        "Total Features",

        "Numeric Features",

        "Categorical Features",

        "Filtered Records"

    ],

    "Value": [

        len(df),

        len(df.columns),

        len(df.select_dtypes(include="number").columns),

        len(df.select_dtypes(exclude="number").columns),

        len(filtered_df)

    ]

})

st.dataframe(

    dataset_info,

    use_container_width=True,

    hide_index=True

)

# ==========================================================
# Developer Information
# ==========================================================

st.divider()

st.header("👨‍💻 Project Information")

st.markdown("""

## 🏡 Real Estate Investment Advisor

### Machine Learning Modules

✅ Property Investment Classification

✅ Future Price Prediction

✅ Interactive Dashboard

✅ Investment Analytics

✅ Feature Importance

✅ Correlation Analysis

---

### Technologies Used

- Python

- Streamlit

- Scikit-Learn

- Pandas

- NumPy

- Plotly

- Joblib

---

### Machine Learning

- Logistic Regression

- Decision Tree

- Random Forest

- Gradient Boosting

- Linear Regression

- Random Forest Regressor

- Gradient Boosting Regressor

""")

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
"""
<center>

### 🏡 Real Estate Investment Advisor

Built with ❤️ using

**Python • Streamlit • Scikit-Learn • Plotly**

© 2026

</center>
""",
unsafe_allow_html=True
)                                                