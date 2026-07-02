"""
3_Model_Performance.py

NOTE:
This template is a corrected scaffold for the Model Performance page.
It expects classifier.pkl, regressor.pkl and cleaned_data.csv.
Replace/extend evaluation sections as needed for your project.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Model Performance",
                   page_icon="📊",
                   layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cleaned_data.csv"
CLASSIFIER_PATH = ROOT / "models" / "classifier.pkl"
REGRESSOR_PATH = ROOT / "models" / "regressor.pkl"

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None

@st.cache_resource
def load_model(path):
    return joblib.load(path) if path.exists() else None

df = load_dataset()
classifier = load_model(CLASSIFIER_PATH)
regressor = load_model(REGRESSOR_PATH)

accuracy = precision = recall = f1 = None
regression_results = None

st.title("📊 Model Performance Dashboard")

if df is None:
    st.error("Dataset not found.")
    st.stop()

# ---------------- Classification ----------------

if classifier is not None and "Good_Investment" in df.columns:

    X = df.drop(columns=[c for c in [
        "Good_Investment","Future_Price_5Y","Investment_Score","ID"
    ] if c in df.columns])

    y = df["Good_Investment"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42,stratify=y
    )

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,zero_division=0)
    recall = recall_score(y_test,y_pred,zero_division=0)
    f1 = f1_score(y_test,y_pred,zero_division=0)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",f"{accuracy:.2%}")
    c2.metric("Precision",f"{precision:.2%}")
    c3.metric("Recall",f"{recall:.2%}")
    c4.metric("F1",f"{f1:.2%}")

    st.dataframe(pd.DataFrame(
        classification_report(y_test,y_pred,output_dict=True,zero_division=0)
    ).T)

    cm = confusion_matrix(y_test,y_pred)
    st.plotly_chart(px.imshow(cm,text_auto=True,
                              x=["Pred 0","Pred 1"],
                              y=["Actual 0","Actual 1"]),
                    use_container_width=True)

# ---------------- Regression ----------------

if regressor is not None and "Future_Price_5Y" in df.columns:

    X = df.drop(columns=[c for c in [
        "Future_Price_5Y","Good_Investment","Investment_Score","ID"
    ] if c in df.columns])

    y = df["Future_Price_5Y"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    y_pred = regressor.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)

    regression_results = {"RMSE":rmse,"MAE":mae,"R2":r2}

    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE",f"{rmse:.3f}")
    c2.metric("MAE",f"{mae:.3f}")
    c3.metric("R²",f"{r2:.4f}")

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={"x":"Actual","y":"Predicted"},
        title="Actual vs Predicted"
    )
    st.plotly_chart(fig,use_container_width=True)

# ---------------- Summary ----------------

st.divider()
st.header("🏆 Overall Summary")

if (
    classifier is not None
    and regressor is not None
    and accuracy is not None
    and regression_results is not None
):

    summary = pd.DataFrame({
        "Model":["Classification","Regression"],
        "Metric 1":[accuracy,regression_results["RMSE"]],
        "Metric 2":[precision,regression_results["MAE"]],
        "Metric 3":[recall,regression_results["R2"]],
        "Metric 4":[f1,np.nan]
    })

    st.dataframe(summary,use_container_width=True,hide_index=True)

else:
    st.warning("Performance metrics are not available.")

st.caption("Real Estate Investment Advisor • Model Performance")
