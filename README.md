# Real-Estate-Investment-Advisor
A machine learning application to assist potential investors in making real estate decisions. 
RealEstateInvestmentAdvisor/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE                     (optional)
│
├── data/
│   └── cleaned_data.csv
│
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── pages/
│   ├── 2_Predictions.py
│   ├── 3_Model_Performance.py
│   ├── 4_Data_Explorer.py
│   └── 5_About.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── classifier_utils.py
│   ├── regressor_utils.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── utils.py
│
├── assets/
│   ├── logo.png                (optional)
│   ├── banner.png              (optional)
│   └── screenshots/            (optional)
│
└── outputs/                    (optional)
    ├── classification/
    └── regression/

# 🏡 Real Estate Investment Advisor

A **Streamlit-based Machine Learning application** that helps users
analyze residential properties, predict future prices, and identify
promising real estate investments.

------------------------------------------------------------------------

## 🚀 Features

-   📊 Interactive Dashboard
-   🔮 Property Investment Prediction
-   💰 Future Price Prediction (Regression)
-   ✅ Investment Classification
-   📈 ROI Estimation
-   📉 Model Performance Evaluation
-   📁 Data Explorer
-   ℹ️ About Page
-   📥 CSV Downloads
-   📊 Interactive Plotly Visualizations

------------------------------------------------------------------------

## 🛠️ Technology Stack

  Category              Technology
  --------------------- --------------------
  Language              Python
  Frontend              Streamlit
  Machine Learning      Scikit-learn
  Data Processing       Pandas, NumPy
  Visualization         Plotly, Matplotlib
  Model Serialization   Joblib

------------------------------------------------------------------------

## 📂 Project Structure

``` text
RealEstateInvestmentAdvisor/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── cleaned_data.csv
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
├── pages/
│   ├── 2_Predictions.py
│   ├── 3_Model_Performance.py
│   ├── 4_Data_Explorer.py
│   └── 5_About.py
├── src/
│   ├── preprocessing.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   ├── classifier_utils.py
│   ├── regressor_utils.py
│   └── utils.py
└── assets/
```

------------------------------------------------------------------------

## ⚙️ Installation

### Clone the repository

``` bash
git clone https://github.com/<your-username>/RealEstateInvestmentAdvisor.git
cd RealEstateInvestmentAdvisor
```

### Create a virtual environment (recommended)

``` bash
python -m venv venv
```

Activate it:

-   Windows:

``` bash
venv\Scripts\activate
```

-   macOS/Linux:

``` bash
source venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Application

``` bash
streamlit run app.py
```

The application will open in your default browser.

------------------------------------------------------------------------

## 🤖 Machine Learning Workflow

1.  Load and preprocess the dataset.
2.  Perform feature engineering.
3.  Train a classification model to identify good investments.
4.  Train a regression model to predict future property prices.
5.  Save trained models using Joblib.
6.  Serve predictions through a Streamlit interface.

------------------------------------------------------------------------

## 📊 Pages

### 🏠 Dashboard

-   KPIs
-   Interactive charts
-   Investment leaderboard
-   Feature importance

### 🔮 Predictions

-   Dynamic property input form
-   Investment recommendation
-   Future price prediction
-   ROI calculation
-   Prediction history

### 📈 Model Performance

-   Classification metrics
-   Regression metrics
-   Confusion matrix
-   ROC curve
-   Actual vs Predicted analysis
-   Residual plots

### 📁 Data Explorer

-   Dataset preview
-   Filters
-   Search
-   Correlation heatmap
-   Summary statistics
-   CSV download

### ℹ️ About

-   Project overview
-   Objectives
-   Technology stack
-   Workflow
-   Future enhancements

------------------------------------------------------------------------

## 📦 Dependencies

-   streamlit
-   pandas
-   numpy
-   scikit-learn
-   plotly
-   matplotlib
-   joblib
-   statsmodels

------------------------------------------------------------------------

## 🚀 Deployment

This project can be deployed using **Streamlit Community Cloud**.

1.  Push the project to GitHub.
2.  Connect the repository in Streamlit Community Cloud.
3.  Set **app.py** as the entry point.
4.  Deploy.

------------------------------------------------------------------------

## 🔮 Future Enhancements

-   Live property listing integration
-   Interactive maps
-   Deep learning price prediction
-   User authentication
-   Cloud database integration
-   Explainable AI (XAI)

------------------------------------------------------------------------

## 👨‍💻 Author

**Avirup Ghosh**

------------------------------------------------------------------------

## 📄 License

This project is intended for educational and academic purposes.

    
