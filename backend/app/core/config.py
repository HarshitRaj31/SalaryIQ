"""
Application configuration settings for the Job Salary Prediction API.
Supports dual-country models (India and United States).
"""

import os
from typing import List

# Paths
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(APP_DIR)
REPO_ROOT = os.path.dirname(BACKEND_DIR)
ML_DIR = os.path.join(REPO_ROOT, "ml")

# ML Model Paths
MODEL_DIR = os.path.join(ML_DIR, "models")

# India model
INDIA_MODEL_PATH = os.path.join(MODEL_DIR, "best_india_model.pkl")
INDIA_METADATA_PATH = os.path.join(MODEL_DIR, "india_feature_metadata.json")
INDIA_COMPARISON_PATH = os.path.join(MODEL_DIR, "india_model_comparison.json")

# US model
US_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
US_METADATA_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")
US_COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.json")

# Backward-compatible default model aliases (pointing to US model)
MODEL_PATH = US_MODEL_PATH
METADATA_PATH = US_METADATA_PATH
COMPARISON_PATH = US_COMPARISON_PATH

# Database Paths
DATA_DIR = os.path.join(BACKEND_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "prediction_history.db")

# API Metadata
PROJECT_NAME: str = "Job Salary Prediction API"
VERSION: str = "2.0.0"
DESCRIPTION: str = (
    "Production dual-country machine learning API to predict tech job salaries "
    "for India (INR / LPA) and the United States (USD / year)."
)
API_V1_STR: str = "/api/v1"

# CORS Configuration
CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://job-salary-prediction-zeta.vercel.app",
]
