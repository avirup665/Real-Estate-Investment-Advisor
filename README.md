# 🏠 Real Estate Investment Advisor

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59.1-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-3.14.0-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A professional end-to-end data science application that helps users evaluate residential properties by:

1. **Classifying whether a property is a good investment**
2. **Predicting its estimated price after five years**

The project combines data preprocessing, exploratory data analysis, feature engineering, classification, regression, experiment tracking, model evaluation, and a multi-page Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [System Workflow](#system-workflow)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Target Variables](#target-variables)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Testing and Validation](#testing-and-validation)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Important Modeling Note](#important-modeling-note)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [License](#license)

---

## Project Overview

The **Real Estate Investment Advisor** is a machine-learning decision-support system designed for property investors, home buyers, real-estate companies, and listing platforms.

The application uses property location, price, size, type, age, amenities, transport accessibility, security, parking, ownership, and availability information to generate data-backed investment insights.

### Main outputs

- Good Investment prediction
- Investment probability or confidence score
- Estimated property price after five years
- Estimated capital appreciation
- Expected five-year ROI
- Annualized return estimate
- Market comparison and property exploration
- Model-performance reports and feature importance

---

## Business Problem

Real-estate investment decisions are often based on incomplete information and subjective judgment. This project creates a consistent analytical workflow that can:

- Assist investors in identifying potentially profitable properties
- Help buyers compare properties using measurable indicators
- Support real-estate businesses in automating listing evaluation
- Improve platform transparency through model-backed recommendations
- Estimate future value using property and location characteristics

---

## Key Features

### Data processing

- Required-column validation
- Duplicate removal
- Missing-value handling
- Text standardization
- Numerical type conversion
- Outlier and range checks
- Floor and total-floor consistency correction
- Recalculation of price per square foot

### Exploratory data analysis

The project includes the 20 EDA requirements specified in the project documentation, covering:

- Property price and size distributions
- Price per square foot by property type
- Size-versus-price relationships
- Outlier detection
- State, city, and locality analysis
- BHK distribution across cities
- Numeric-feature correlations
- Schools, hospitals, furnishing, facing, parking, and amenities
- Ownership and availability analysis
- Transport accessibility and investment potential

### Machine learning

- Five classification models
- Five regression models
- Automated model comparison
- Best-model selection
- Serialized prediction pipelines
- Classification and regression leaderboards
- Feature-importance reports
- Prediction diagnostics

### Application

- Professional multi-page Streamlit interface
- Interactive property prediction form
- Property filtering and CSV download
- Market-level visual analysis
- Model-confidence display
- ROI and future-value calculations
- Model-performance dashboard
- Methodology and limitations page

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.13 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Plotly |
| Web application | Streamlit |
| Model serialization | Joblib |
| Experiment tracking | MLflow |
| Testing | Pytest, Streamlit AppTest |
| Development environment | VS Code |
| Version control | Git and GitHub |

---

## System Workflow

```text
Raw housing dataset
        │
        ▼
Data validation and cleaning
        │
        ▼
Feature engineering
        │
        ├──────────────► Exploratory data analysis
        │
        ▼
Target generation
        │
        ├──────────────► Classification models
        │                    Target: Good_Investment
        │
        └──────────────► Regression models
                             Target: Future_Price_5Y
        │
        ▼
Model evaluation and comparison
        │
        ▼
Best model serialization
        │
        ├──────────────► MLflow experiment tracking
        │
        ▼
Streamlit prediction and analytics dashboard
```

---

## Dataset

The source dataset is stored at:

```text
data/raw/india_housing_prices.csv
```

### Dataset dimensions

| Dataset | Rows | Columns |
|---|---:|---:|
| Raw dataset | 250,000 | 23 |
| Processed dataset | 250,000 | 39 |

The processed dataset contains **zero missing cells** after the complete preprocessing pipeline.

### Important source features

- `State`
- `City`
- `Locality`
- `Property_Type`
- `BHK`
- `Size_in_SqFt`
- `Price_in_Lakhs`
- `Price_per_SqFt`
- `Year_Built`
- `Furnished_Status`
- `Floor_No`
- `Total_Floors`
- `Age_of_Property`
- `Nearby_Schools`
- `Nearby_Hospitals`
- `Public_Transport_Accessibility`
- `Parking_Space`
- `Security`
- `Amenities`
- `Facing`
- `Owner_Type`
- `Availability_Status`

A complete explanation of the fields is available in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

---

## Feature Engineering

The preprocessing pipeline creates additional features that improve both interpretability and predictive performance.

| Engineered feature | Description |
|---|---|
| `Amenity_Count` | Number of amenities available |
| `Amenity_Density_Score` | Normalized amenity availability score |
| `Has_Gym` | Gym indicator |
| `Has_Pool` | Swimming-pool indicator |
| `Has_Clubhouse` | Clubhouse indicator |
| `Has_Garden` | Garden indicator |
| `Has_Playground` | Playground indicator |
| `Floor_Ratio` | Floor number divided by total floors |
| `Infrastructure_Score` | Combined transport, school, hospital, and amenity score |
| `Growth_Rate_Annual` | Estimated annual appreciation rate |
| `Expected_ROI_5Y` | Estimated percentage return over five years |
| `Below_Segment_Median` | Whether price per square foot is below the local segment median |
| `Investment_Score` | Domain-based multi-factor investment score |

---

## Target Variables

### Classification target: `Good_Investment`

The classification task determines whether a property is likely to be a good investment.

The label is generated through a transparent multi-factor scoring rule based on:

- Price per square foot relative to the local segment median
- Expected appreciation rate
- Public transport accessibility
- Amenity availability
- Ready-to-move status
- Security
- Parking
- Nearby schools and hospitals

### Regression target: `Future_Price_5Y`

The regression task predicts the estimated property value after five years.

Future value is generated using compound appreciation:

```text
Future Price = Current Price × (1 + Annual Growth Rate)⁵
```

The annual growth rate is adjusted using property and location characteristics and is constrained to a realistic project range.

---

## Machine Learning Models

### Classification models

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Extra Trees Classifier
5. Gaussian Naive Bayes

Model selection criterion: **F1 score**

### Regression models

1. Linear Regression
2. Ridge Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Extra Trees Regressor

Model selection criterion: **RMSE**

### Evaluation metrics

#### Classification

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Confusion matrix
- ROC curve

#### Regression

- RMSE
- MAE
- R² score
- Actual-versus-predicted analysis
- Residual analysis

---

## Model Performance

Models were evaluated using a 30,000-row stratified modeling sample:

- Training rows: 24,000
- Test rows: 6,000
- Random state: 42

### Best classification model

| Metric | Result |
|---|---:|
| Model | Random Forest |
| Accuracy | 0.9122 |
| Precision | 0.9036 |
| Recall | 0.9020 |
| F1 score | 0.9028 |
| ROC AUC | 0.9753 |

### Best regression model

| Metric | Result |
|---|---:|
| Model | Extra Trees |
| RMSE | ₹5.7497 lakh |
| MAE | ₹4.1263 lakh |
| R² | 0.9993 |

Detailed model comparisons are available in:

```text
reports/classification_leaderboard.csv
reports/regression_leaderboard.csv
reports/prediction_diagnostics.json
```

---

## Streamlit Application

The dashboard contains six application views, including the home page.

### Home dashboard

- Dataset overview
- Number of properties, states, and cities
- Good-investment percentage
- Average five-year ROI
- City price comparison
- Property size versus current price
- Best-model summary

### Investment Predictor

- Property-detail input form
- Good-investment prediction
- Prediction confidence
- Estimated five-year price
- Estimated capital gain
- Expected ROI
- Annualized return
- Investor recommendation card

### Property Explorer

- Filters for state, city, locality, property type, BHK, area, and price
- Interactive listing exploration
- Market-level summaries
- Filtered-data CSV download

### Market Insights

- Interactive responses to all 20 required EDA questions
- Price, size, location, feature, ownership, amenity, and investment analysis

### Model Performance

- Classification leaderboard
- Regression leaderboard
- Confusion matrix
- ROC curve
- Actual-versus-predicted plot
- Residual analysis
- Feature importance

### Project About

- Problem statement
- Methodology
- System architecture
- Model assumptions
- Limitations
- Business use cases

---

## Project Structure

```text
RealEstateInvestmentAdvisor_Professional/
│
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── runtime.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── pages/
│   ├── 1_Investment_Predictor.py
│   ├── 2_Property_Explorer.py
│   ├── 3_Market_Insights.py
│   ├── 4_Model_Performance.py
│   └── 5_Project_About.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── inference.py
│   ├── evaluation.py
│   ├── eda.py
│   ├── mlflow_tracking.py
│   └── ui.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── train_models.py
│   └── validate_project.py
│
├── data/
│   ├── raw/
│   │   └── india_housing_prices.csv
│   └── processed/
│       └── cleaned_housing_data.csv
│
├── models/
│   ├── best_classifier.joblib
│   ├── best_regressor.joblib
│   └── metadata.json
│
├── reports/
│   ├── eda_summary.json
│   ├── classification_leaderboard.csv
│   ├── regression_leaderboard.csv
│   ├── experiment_runs.csv
│   ├── feature_importance_classifier.csv
│   ├── feature_importance_regressor.csv
│   ├── prediction_diagnostics.json
│   ├── target_distribution.csv
│   ├── city_market_summary.csv
│   └── data_quality_report.json
│
├── tests/
│   ├── conftest.py
│   ├── test_feature_engineering.py
│   └── test_inference.py
│
├── mlruns/
│   └── README.md
│
├── PROJECT_DOCUMENTATION.md
├── DATA_DICTIONARY.md
├── MLFLOW_SETUP.md
├── PROJECT_MANIFEST.md
├── VALIDATION_REPORT.txt
├── setup_windows.bat
└── run_app.bat
```

---

## Installation and Setup

### Prerequisites

- Python 3.13
- VS Code
- Git
- A terminal such as PowerShell or Command Prompt

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Real-Estate-Investment-Advisor.git
cd Real-Estate-Investment-Advisor
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\Activate.ps1
```

When PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Validate the project

```powershell
python scripts\validate_project.py
```

---

## Running the Project

### Start Streamlit

```powershell
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

### One-click Windows setup

Run the following file once:

```text
setup_windows.bat
```

Then launch the application using:

```text
run_app.bat
```

### Rebuild all generated artifacts

The repository already contains processed data, reports, and trained models. To regenerate them:

```powershell
python scripts\run_pipeline.py --training-rows 30000
```

This command:

- Loads the raw dataset
- Cleans and preprocesses the data
- Engineers model features
- Creates target variables
- Generates EDA reports
- Trains classification models
- Trains regression models
- Evaluates all models
- Saves the best models
- Writes reports and metadata

---

## MLflow Experiment Tracking

MLflow integration is optional and is included for professional experiment management.

### Install development dependencies

```powershell
python -m pip install -r requirements-dev.txt
```

### Run the training pipeline with MLflow

```powershell
python scripts\run_pipeline.py --training-rows 30000 --enable-mlflow
```

### Start the MLflow interface

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the address displayed in the terminal, normally:

```text
http://127.0.0.1:5000
```

MLflow records:

- Model names
- Hyperparameters
- Accuracy, precision, recall, F1, and ROC AUC
- RMSE, MAE, and R²
- Training time
- Model artifacts
- Registered best models

---

## Testing and Validation

### Run automated tests

```powershell
python -m pytest -q
```

### Run project validation

```powershell
python scripts\validate_project.py
```

The packaged project was validated for:

- Complete preprocessing and training pipeline execution
- Python syntax compilation
- Model artifact loading
- End-to-end sample inference
- Three passing Pytest tests
- All six Streamlit scripts passing AppTest
- Successful prediction-form submission
- Streamlit health endpoint response
- MLflow experiment and model-registry smoke testing

See [`VALIDATION_REPORT.txt`](VALIDATION_REPORT.txt) for the full validation summary.

---

## Deployment on Streamlit Cloud

1. Push all required files to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Select **Create app**.
4. Choose the GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Deploy the application.

The repository already contains the processed dataset and trained model artifacts, so the deployed application does not need to retrain models during startup.

### GitHub upload note

The processed dataset is larger than GitHub's browser-upload limit. Use Git from the VS Code terminal or GitHub Desktop instead of uploading the files individually through the website.

```powershell
git init
git add .
git commit -m "Initial commit: Real Estate Investment Advisor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Real-Estate-Investment-Advisor.git
git push -u origin main
```

---

## Important Modeling Note

The original dataset does not contain observed historical five-year sale prices or verified investment outcomes. Therefore:

- `Future_Price_5Y` is generated using a transparent compound-growth rule.
- `Good_Investment` is generated using a documented multi-factor investment score.

The project demonstrates a complete machine-learning workflow and should be treated as an educational decision-support application, not as a guarantee of investment returns.

The very high regression R² must be interpreted in this context because the regression target is algorithmically derived from the available property features.

---

## Future Enhancements

- Integrate verified historical transaction data
- Add latitude and longitude for map-based analysis
- Include crime rate and neighborhood safety data
- Add interest rates, inflation, and macroeconomic indicators
- Include builder reputation and project-completion history
- Add explainability using SHAP or LIME
- Introduce time-series forecasting
- Add model-drift monitoring
- Connect to a live property-listing API
- Add user authentication and saved investment portfolios
- Deploy models through a REST API

---

## Author

**Avirup Ghosh**  
Data Science Project — Real Estate and Financial Analytics

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

---

> **Disclaimer:** This application is intended for educational and analytical purposes. Real-estate investment decisions should also consider legal verification, financing costs, taxes, market risk, location-specific research. 
