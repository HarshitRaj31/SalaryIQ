# SalaryIQ — Intelligent Job Salary Prediction System

<p align="center">
  <strong>Predict salaries. Understand the factors. Explore the data.</strong>
</p>

<p align="center">
  A machine learning-powered salary prediction platform with interactive analytics, prediction history, and explainable AI.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/React-Vite-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple" />
</p>

---

## Overview

**SalaryIQ** is an intelligent job salary prediction system that estimates salaries in USD using machine learning.

The application analyzes employment and job-related attributes, generates salary predictions, explains model outputs using SHAP, and stores previous predictions for further analysis.

Rather than presenting a salary estimate as a black-box result, SalaryIQ aims to make predictions more interpretable through feature explanations, analytics, and historical comparisons.

## Key Features

| Feature | Description |
|---|---|
| Salary Prediction | Predict estimated salary using job-related attributes |
| Multiple ML Models | Compare Linear Regression, Random Forest, and Gradient Boosting |
| Explainable AI | Understand individual predictions using SHAP |
| Feature Importance | Explore model feature importance |
| Prediction History | Store, view, search, and manage previous predictions |
| Analytics Dashboard | Explore model performance and salary statistics |
| Prediction Comparison | Compare multiple saved predictions |
| PDF Reports | Generate downloadable prediction reports |
| CSV Export | Export prediction history |
| Batch Prediction API | Submit multiple prediction records |
| REST API | Access prediction and analytics functionality through FastAPI |

## Technology Stack

**Frontend**
- React
- Vite
- JavaScript
- CSS
- Recharts
- Lucide React

**Backend**
- Python
- FastAPI
- Pydantic
- SQLite

**Machine Learning**
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Joblib

## System Architecture

```text
                  USER
                   |
                   v
          REACT + VITE FRONTEND
                   |
                   v
              FASTAPI
                   |
          +--------+--------+
          |                 |
          v                 v
   PREDICTION SERVICE   HISTORY SERVICE
          |                 |
          v                 v
    TRAINED ML MODEL      SQLITE DB
          |
          v
     SHAP EXPLAINER
          |
          v
   EXPLANATION RESPONSE
          |
          v
     FRONTEND DASHBOARD
```

## Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Dataset collection and inspection.
2. Exploratory Data Analysis.
3. Missing-value handling and duplicate investigation.
4. Feature preprocessing.
5. Model training and hyperparameter tuning.
6. Cross-validation and model evaluation.
7. Best-model selection.
8. Model serialization.
9. Prediction through the FastAPI backend.
10. SHAP-based prediction explanation.

### Dataset

The project uses the `ds_salaries.csv` dataset.

- Records before cleaning: **3,755**
- Records after exact duplicate removal: **2,584**
- Prediction target: `salary_in_usd`

**Input features**

- Work year
- Experience level
- Employment type
- Job title
- Employee residence
- Remote ratio
- Company location
- Company size

**Important:** This is an international salary dataset. Predictions should not be interpreted as India-specific salary benchmarks.

### Models Evaluated

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The models are evaluated using MAE, RMSE, and R². Model selection uses cross-validation RMSE.

### Evaluation Results

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | $40,002.86 | $53,230.96 | 0.3522 |
| Random Forest | $39,038.61 | $51,414.47 | 0.3957 |
| Gradient Boosting | $38,199.49 | $50,683.36 | 0.4127 |

Random Forest was selected based on the lowest reported cross-validation RMSE:

**$50,578.10 ± $1,855.82**

Gradient Boosting achieved slightly better test-set metrics in this evaluation. The model-selection criterion and test metrics should therefore be considered separately.

## Explainable AI — SHAP

SalaryIQ integrates SHAP to help explain individual model predictions.

The explanation dashboard can present:

- Baseline model output
- Feature contributions
- Positive and negative influences
- Contribution summaries
- Global feature importance, where supported by the SHAP implementation

SHAP explanations help users understand how model features influence an individual estimate. They do not establish causation.

## Backend API

Base URL:

```text
http://127.0.0.1:8000
```

Main endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/prediction/predict` | Predict one salary |
| POST | `/api/prediction/batch` | Batch predictions |
| GET | `/api/prediction/metadata` | Retrieve input metadata |
| GET | `/api/prediction/health` | Health check |
| GET | `/api/prediction/feature-importance` | Retrieve model feature importance |
| GET | `/api/prediction/history` | Retrieve prediction history |
| GET | `/api/prediction/history/summary` | Retrieve history summary |
| GET | `/api/prediction/history/{prediction_id}` | Retrieve a saved prediction |
| DELETE | `/api/prediction/history/{prediction_id}` | Delete one prediction |
| DELETE | `/api/prediction/history` | Clear prediction history |
| POST | `/api/prediction/explain` | Explain a prediction |
| GET | `/api/prediction/history/{prediction_id}/explain` | Explain a historical prediction |
| GET | `/api/prediction/explain/global` | Retrieve global explanations |

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
Job-Salary-Prediction/
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── Analytics.jsx
│       ├── FeatureImportance.jsx
│       ├── PredictionHistory.jsx
│       ├── ShapDashboard.jsx
│       ├── utils/
│       │   └── pdfReportGenerator.js
│       └── styles and CSS files
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   ├── routes/
│   │   ├── schemas/
│   │   └── services/
│   └── data/
│       └── prediction_history.db
│
└── ml/
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── notebooks/
    ├── src/
    ├── models/
    └── reports/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Job-Salary-Prediction
```

### 2. Set up the Python environment

From the repository root:

```powershell
python -m venv ml/venv
.\ml\venv\Scripts\Activate.ps1
```

Install backend dependencies using your project's dependency file:

```powershell
pip install -r backend/requirements.txt
```

If your requirements file is stored elsewhere, use its actual path.

### 3. Start the backend

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

### 4. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## Important Notes

- The trained model files must be available at the configured model paths.
- The dataset represents international salary records, not a live salary database.
- Model performance depends on the dataset and its coverage.
- Salary estimates are not guaranteed actual compensation.
- An MAE-based estimated range is an error-based indication, not a calibrated confidence interval.
- SQLite is suitable for the current local application; production deployment should consider persistent database hosting and backups.

## Future Improvements

- Add India-specific salary datasets.
- Integrate updated salary market data.
- Improve model performance for high-salary outliers.
- Add user authentication and personalized dashboards.
- Introduce cloud database support.
- Add automated model retraining.
- Deploy the frontend and backend.
- Add monitoring for prediction drift and model performance.

## Project Goals

SalaryIQ was developed to combine machine learning, full-stack development, data analysis, and explainable AI into one practical application.

The goal is not only to predict a salary, but also to help users understand the factors behind the estimate.

---

<p align="center">
  <strong>SalaryIQ</strong><br/>
  Turning job data into understandable salary insights.
</p>
#   S a l a r y I Q  
 