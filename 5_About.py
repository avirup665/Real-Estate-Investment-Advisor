"""
5_About.py
Real Estate Investment Advisor
"""

import streamlit as st

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Project")

st.markdown("""
## 🏡 Real Estate Investment Advisor

The **Real Estate Investment Advisor** is an end-to-end machine learning application
that helps users evaluate residential properties by predicting:

- ✅ Whether a property is a good investment
- 📈 Estimated future property price (5-year forecast)
- 💹 Expected Return on Investment (ROI)

The application combines interactive analytics with machine learning models to
support data-driven investment decisions.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Objectives")
    st.markdown("""
- Analyze real estate data
- Identify promising investment opportunities
- Predict future property prices
- Classify properties as good or poor investments
- Provide interactive dashboards for analysis
""")

with col2:
    st.subheader("🧠 Machine Learning")
    st.markdown("""
**Classification**
- Predicts investment suitability

**Regression**
- Predicts future property price

**Techniques Used**
- Data preprocessing
- Feature engineering
- Pipeline-based preprocessing
- Model evaluation
""")

st.divider()

st.subheader("⚙️ Technology Stack")

tech = {
    "Frontend": "Streamlit",
    "Backend": "Python",
    "Machine Learning": "Scikit-learn",
    "Data Analysis": "Pandas, NumPy",
    "Visualization": "Plotly",
    "Model Storage": "Joblib"
}

st.table(tech.items())

st.divider()

st.subheader("🧩 Project Workflow")

st.markdown("""
1. Load dataset
2. Clean and preprocess data
3. Engineer features
4. Train classification model
5. Train regression model
6. Save trained models
7. Predict investment quality
8. Forecast future price
9. Visualize insights in Streamlit
""")

st.divider()

st.subheader("📂 Application Pages")

pages = [
    ("🏠 Dashboard", "KPIs, charts and investment analytics"),
    ("🔮 Predictions", "Predict investment quality and future price"),
    ("📊 Model Performance", "Evaluate ML models"),
    ("📁 Data Explorer", "Explore and filter the dataset"),
    ("ℹ️ About", "Project information")
]

st.table({"Page":[p[0] for p in pages],
          "Description":[p[1] for p in pages]})

st.divider()

st.subheader("🚀 Future Enhancements")

st.markdown("""
- Deep learning price prediction
- Live property listings integration
- Interactive maps
- Time-series forecasting
- User authentication
- Cloud database integration
- AI-powered investment recommendations
""")

st.divider()

st.subheader("👨‍💻 Developer")

st.info("""
**Project:** Real Estate Investment Advisor

**Framework:** Streamlit

**Machine Learning:** Scikit-learn

**Visualization:** Plotly
""")

st.divider()

st.success("Thank you for using the Real Estate Investment Advisor!")

st.caption("© 2026 Real Estate Investment Advisor • Built with Streamlit and Scikit-learn")
