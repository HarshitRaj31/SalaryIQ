# 🇮🇳🇺🇸 SalaryIQ — Intelligent Job Salary Prediction System

<p align="center">
  <strong>Predict salaries. Understand the factors. Explore the data.</strong>
</p>

<p align="center">
  An intelligent, machine learning-powered salary prediction platform supporting India and the United States, with explainable AI, interactive analytics, and prediction history.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/React-Vite-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple" />
  <img src="https://img.shields.io/badge/Status-Development-orange" />
</p>

---

## 🌟 Overview

**SalaryIQ** is a machine learning-powered job salary prediction platform designed to estimate salaries based on professional experience, job roles, employment details, location, and company attributes.

The platform supports two independent salary prediction systems:

- 🇮🇳 **India:** Estimates salaries in Indian Rupees (INR), represented in Lakhs Per Annum (LPA).
- 🇺🇸 **United States:** Estimates salaries in US Dollars (USD), represented as annual compensation.

SalaryIQ combines machine learning, data analytics, and Explainable AI (XAI) to make salary predictions more understandable.

Users can generate salary estimates, explore influential job attributes, view prediction history, and analyze salary trends through an interactive dashboard.

## ✨ Key Features

| Feature | Description |
|---|---|
| Dual-Country Prediction | Independent salary prediction for India and the United States |
| India Salary Prediction | Estimates salary in INR/LPA |
| US Salary Prediction | Estimates salary in USD/year |
| Multiple ML Models | Linear Regression, Random Forest, Gradient Boosting |
| Explainable AI | Understand individual predictions using SHAP |
| Feature Importance | Identify influential model features |
| Prediction History | Store and retrieve previous predictions |
| Analytics Dashboard | Explore salary statistics and model performance |
| Prediction Comparison | Compare saved prediction profiles |
| PDF Reports | Generate downloadable prediction reports |
| CSV Export | Export prediction history |
| Batch Prediction API | Submit multiple prediction records |
| REST API | Access prediction functionality through FastAPI |

## 🛠️ Technology Stack

### Frontend
- React.js
- Vite
- JavaScript
- CSS
- Recharts
- Lucide React

### Backend
- Python
- FastAPI
- Pydantic
- SQLite

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Joblib

### Deployment
- Vercel — Frontend
- Render — Backend

## 🏗️ System Architecture

```text
                         USER
                          |
                          v
                  REACT + VITE FRONTEND
                          |
                          v
                     FASTAPI BACKEND
                          |
                  COUNTRY SELECTION
                          |
                +---------+---------+
                |                   |
                v                   v
          INDIA MODEL            US MODEL
                |                   |
                v                   v
        INR / LPA OUTPUT       USD / YEAR OUTPUT
                |                   |
                +---------+---------+
                          |
                          v
                 PREDICTION SERVICE
                          |
                +---------+---------+
                |                   |
                v                   v
           SHAP SERVICE        HISTORY SERVICE
                |                   |
                v                   v
          EXPLANATIONS           SQLITE DB
                |                   |
                +---------+---------+
                          |
                          v
                  FRONTEND DASHBOARD
```

Each country uses its own trained model and preprocessing pipeline. This prevents country-specific input features and salary units from being mixed.

## 🤖 Machine Learning System

SalaryIQ uses separate datasets and trained models for India and the United States.

### 🇮🇳 India Salary Prediction

**Dataset:** `indian_tech_jobs_2026.csv`

The India-specific dataset was cleaned and processed to prepare training data for salary prediction.

Dataset preparation:

| Metric | Records |
|---|---:|
| Original records | 23,201 |
| Records with disclosed salary | 2,768 |
| Cleaned salary records | 2,348 |
| Records after deduplication | 2,269 |

**Prediction target:** `salary_in_lpa`

**Model input features:**
- Work year
- Experience level
- Employment type
- Standardized job role
- Primary city
- Remote-work ratio
- Company size
- Average years of experience
- Company rating
- Number of skills

### India Model Evaluation

Three machine learning models were evaluated:

| Model | CV RMSE (LPA) | Test MAE (LPA) | Test RMSE (LPA) | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 8.466 | 5.686 | 7.167 | 0.4462 |
| Random Forest | 7.997 | 5.281 | 7.090 | 0.4581 |
| Gradient Boosting | 7.802 | 5.273 | 6.887 | 0.4887 |

**Selected model:** Gradient Boosting Regressor

Test-set performance:
- MAE: 5.273 LPA
- RMSE: 6.887 LPA
- R²: 0.4887

The test MAE indicates an average absolute prediction error of approximately ₹5.27 lakh per year on the evaluation dataset.

### 🇺🇸 United States Salary Prediction

**Dataset:** `ds_salaries.csv`

The US model uses an international technology salary dataset.

Dataset preparation:
- Original records: 3,755
- Records after exact duplicate removal: 2,584
- Prediction target: `salary_in_usd`

**Input features:**
- Work year
- Experience level
- Employment type
- Job title
- Employee residence
- Remote ratio
- Company location
- Company size

### US Model Evaluation

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | $40,002.86 | $53,230.96 | 0.3522 |
| Random Forest | $39,038.61 | $51,414.47 | 0.3957 |
| Gradient Boosting | $38,199.49 | $50,683.36 | 0.4127 |

The existing US model selection used cross-validation RMSE.

**Selected model:** Random Forest Regressor

Reported cross-validation RMSE:

`$50,578.10 ± $1,855.82`

Gradient Boosting achieved slightly better test-set metrics in the reported evaluation. Cross-validation model selection and held-out test performance are separate evaluation criteria.

**Note:** The US dataset is international and should not be treated as an India-specific salary benchmark.

## 🧠 Explainable AI — SHAP

SalaryIQ integrates SHAP (SHapley Additive exPlanations) to make model predictions more interpretable.

The explainability system aims to provide:

- Baseline model output
- Individual feature contributions
- Positive and negative feature influences
- Contribution summaries
- Global feature importance, where supported

Each country must use explanations generated from its corresponding model.

SHAP explains how input features contribute to a model's output. It does not establish causal relationships between job attributes and salary.

## 📊 Analytics and Prediction History

SalaryIQ provides tools to explore and understand previous predictions.

### Analytics Dashboard
- Salary prediction statistics
- Model performance metrics
- Feature importance
- Salary distribution
- Prediction trends

### Prediction History
- View previous predictions
- Retrieve individual prediction records
- Review job profile details
- Compare saved predictions
- Filter records by country, where supported

### Export and Reporting
- Download PDF prediction reports
- Export prediction history to CSV

## 🔌 Backend REST API

**Production backend:**

https://salaryiq-api-fec0.onrender.com

**Local backend:**

```text
http://127.0.0.1:8000
```

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | API health |
| POST | `/api/prediction/predict` | Predict one salary |
| POST | `/api/prediction/batch` | Batch predictions |
| GET | `/api/prediction/metadata` | Retrieve input metadata |
| GET | `/api/prediction/health` | Prediction service health |
| GET | `/api/prediction/feature-importance` | Model feature importance |
| GET | `/api/prediction/history` | Retrieve prediction history |
| GET | `/api/prediction/history/summary` | History summary |
| GET | `/api/prediction/history/{prediction_id}` | Retrieve one prediction |
| DELETE | `/api/prediction/history/{prediction_id}` | Delete one prediction |
| DELETE | `/api/prediction/history` | Clear prediction history |
| POST | `/api/prediction/explain` | Explain a prediction |
| GET | `/api/prediction/history/{prediction_id}/explain` | Explain a saved prediction |
| GET | `/api/prediction/explain/global` | Global model explanations |

Check the live API documentation for the exact supported routes and request schemas.

**Interactive API documentation:**

```text
http://127.0.0.1:8000/docs
```

## 📁 Project Structure

```text
Job-Salary-Prediction/
│
├── frontend/
│   └── job/
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   ├── services/
│       │   ├── utils/
│       │   ├── App.jsx
│       │   └── main.jsx
│       ├── package.json
│       └── vite.config.js
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── routes/
│   │   │   └── prediction.py
│   │   ├── schemas/
│   │   │   └── salary.py
│   │   └── services/
│   │       ├── prediction_service.py
│   │       ├── history_service.py
│   │       └── shap_service.py
│   │
│   ├── data/
│   │   └── prediction_history.db
│   └── requirements.txt
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   ├── src/
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── best_india_model.pkl
│   │   ├── feature_metadata.json
│   │   ├── india_feature_metadata.json
│   │   ├── model_comparison.json
│   │   └── india_model_comparison.json
│   └── reports/
│
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.x
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Job-Salary-Prediction
```

### 2. Create a Python virtual environment

From the repository root:

```powershell
python -m venv ml/venv
.\ml\venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

### 4. Start the backend

From the repository root:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Install frontend dependencies

Open another terminal:

```powershell
cd frontend/job
npm install
```

### 6. Configure frontend environment

Create a `.env` file inside `frontend/job/`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Use your deployed backend URL for production.

### 7. Start the frontend

```powershell
npm run dev
```

Frontend:

```text
http://localhost:5173
```

## ☁️ Deployment

SalaryIQ uses Vercel for frontend hosting and Render for backend hosting.

### Frontend — Vercel

<https://job-salary-prediction-zeta.vercel.app/>

Configuration:

| Setting | Value |
|---|---|
| Framework | Vite |
| Root directory | `frontend/job` |
| Build command | `npm run build` |
| Output directory | `dist` |

Environment variable:

```env
VITE_API_BASE_URL=https://salaryiq-api-fec0.onrender.com
```

### Backend — Render

<https://salaryiq-api-fec0.onrender.com>

Build command:

```bash
pip install -r backend/requirements.txt
```

Start command:

```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT
```

Ensure that both trained model artifacts and their required dependencies are included in the backend deployment environment.

**Production consideration:** SQLite data stored on an ephemeral Render filesystem may not persist across restarts or redeployments. Use a persistent disk or a managed database for production history storage.

## 🧪 Testing

The application should be tested across both prediction systems.

| Test Area | Expected Behavior |
|---|---|
| India Prediction | Returns INR/LPA salary estimate |
| US Prediction | Returns USD/year salary estimate |
| Country Selection | Selects the correct model |
| Input Validation | Rejects invalid or missing inputs |
| Model Loading | Loads the correct country model |
| Prediction History | Saves and retrieves predictions |
| Feature Importance | Uses the correct model |
| SHAP | Explains the selected model |
| Frontend Integration | Displays results correctly |
| API Health | Returns backend status |

Testing status should be updated based on actual executed test results.

## ⚠️ Important Notes and Limitations

- SalaryIQ provides estimated salaries, not guaranteed compensation.
- India salary data is based on disclosed advertised salary records and may not represent the entire job market.
- The US model uses an international dataset.
- Model performance depends on dataset coverage, feature quality, and distribution.
- MAE represents average absolute prediction error; it is not a calibrated confidence interval.
- SHAP contributions describe model behavior, not causation.
- Model performance should be monitored as new salary data becomes available.
- Currency values and salary units must remain country-specific.

## 🚀 Future Improvements

- Expand India-specific salary datasets.
- Integrate regularly updated salary market information.
- Improve prediction performance for high-salary outliers.
- Add user authentication and personalized dashboards.
- Introduce managed cloud database support.
- Automate model retraining and evaluation.
- Implement model drift monitoring.
- Add more country-specific salary models.
- Improve model calibration and prediction uncertainty estimates.

## 🎯 Project Goals

SalaryIQ brings together:

- Full-stack web development
- Machine learning
- Data preprocessing and exploratory analysis
- Model evaluation
- Explainable Artificial Intelligence
- REST API development
- Cloud deployment

The goal is not simply to predict a salary, but to help people understand the factors behind salary estimates across different job markets.

---

<p align="center">
  <strong>SalaryIQ</strong><br/>
  Turning job data into understandable salary insights.
</p>

<p align="center">
  Built with Python, React, FastAPI and Machine Learning.
</p>
