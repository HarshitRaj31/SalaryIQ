"""
Application configuration settings for the Job Salary Prediction API.
"""

import os
from typing import List

# Paths
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(APP_DIR)
REPO_ROOT = os.path.dirname(BACKEND_DIR)
ML_DIR = os.path.join(REPO_ROOT, "ml")

MODEL_PATH = os.path.join(ML_DIR, "models", "best_model.pkl")
METADATA_PATH = os.path.join(ML_DIR, "models", "feature_metadata.json")
COMPARISON_PATH = os.path.join(ML_DIR, "models", "model_comparison.json")

# Database Paths
DATA_DIR = os.path.join(BACKEND_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "prediction_history.db")

# API Metadata
PROJECT_NAME: str = "Job Salary Prediction API"
VERSION: str = "1.0.0"
DESCRIPTION: str = (
    "Production machine learning API to predict tech job salaries "
    "based on experience, role, location, remote work status, and company size."
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

