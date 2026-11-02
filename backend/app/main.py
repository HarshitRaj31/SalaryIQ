"""
Main FastAPI Application Entrypoint for Job Salary Prediction Backend.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure app package is discoverable when running main.py directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core import config
from app.routes.prediction import router as prediction_router

# Initialize FastAPI application
app = FastAPI(
    title=config.PROJECT_NAME,
    version=config.VERSION,
    description=config.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure Cross-Origin Resource Sharing (CORS) for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://job-salary-prediction-zeta.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(prediction_router, prefix=config.API_V1_STR)
app.include_router(prediction_router, prefix="/api")  # Convenience alias
app.include_router(prediction_router, prefix="")      # Root alias (/prediction/predict)

# Direct endpoint aliases for frontend convenience
from typing import Optional
from app.routes.prediction import (
    clear_prediction_history,
    delete_prediction_history,
    explain_global_endpoint,
    explain_historical_prediction_endpoint,
    explain_prediction_endpoint,
    get_feature_importance_endpoint,
    get_history_summary,
    get_prediction_details,
    get_prediction_history,
    predict_salary_endpoint,
)
from app.schemas.salary import (
    FeatureImportanceResponse,
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)

@app.post("/predict", response_model=SalaryPredictionResponse, tags=["Salary Prediction"])
def root_predict_alias(request: SalaryPredictionRequest):
    return predict_salary_endpoint(request)


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Salary Prediction"])
def root_feature_importance_alias(top_n: int = 10):
    """Exposes the actual trained Random Forest model's Top influential features."""
    return get_feature_importance_endpoint(top_n=top_n)


@app.get("/history", tags=["Prediction History"])
def root_history_alias(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    experience_level: Optional[str] = None,
):
    """Direct alias to fetch paginated prediction history from SQLite."""
    return get_prediction_history(
        limit=limit,
        offset=offset,
        search=search,
        experience_level=experience_level,
    )


@app.get("/history/summary", tags=["Prediction History"])
def root_history_summary_alias():
    """Direct alias to fetch aggregated history statistics from SQLite."""
    return get_history_summary()


@app.get("/history/{prediction_id}", tags=["Prediction History"])
def root_prediction_detail_alias(prediction_id: int):
    """Direct alias to fetch a single prediction detail record."""
    return get_prediction_details(prediction_id=prediction_id)


@app.delete("/history/{prediction_id}", tags=["Prediction History"])
def root_delete_prediction_alias(prediction_id: int):
    """Direct alias to delete a specific prediction from SQLite."""
    return delete_prediction_history(prediction_id=prediction_id)


@app.delete("/history", tags=["Prediction History"])
def root_clear_history_alias():
    """Direct alias to clear all historical prediction records."""
    return clear_prediction_history()


@app.post("/explain", tags=["Explainable AI (SHAP)"])
def root_explain_alias(request: SalaryPredictionRequest):
    """Direct alias to compute local TreeSHAP attribution and waterfall data."""
    return explain_prediction_endpoint(request)


@app.get("/history/{prediction_id}/explain", tags=["Explainable AI (SHAP)"])
def root_explain_history_alias(prediction_id: int):
    """Direct alias to explain a historical SQLite prediction with SHAP."""
    return explain_historical_prediction_endpoint(prediction_id)


@app.get("/explain/global", tags=["Explainable AI (SHAP)"])
def root_explain_global_alias(top_n: int = 10):
    """Direct alias to fetch global SHAP feature importance overview."""
    return explain_global_endpoint(top_n=top_n)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint welcoming users and providing documentation links."""
    return JSONResponse(
        content={
            "service": config.PROJECT_NAME,
            "version": config.VERSION,
            "status": "online",
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_spec": "/openapi.json",
            },
            "endpoints": {
                "predict": f"{config.API_V1_STR}/prediction/predict",
                "batch_predict": f"{config.API_V1_STR}/prediction/predict/batch",
                "metadata": f"{config.API_V1_STR}/prediction/metadata",
                "feature_importance": f"{config.API_V1_STR}/prediction/feature-importance",
                "history": f"{config.API_V1_STR}/prediction/history",
                "history_summary": f"{config.API_V1_STR}/prediction/history/summary",
                "health": f"{config.API_V1_STR}/prediction/health",
            },
        }
    )


@app.get("/health", tags=["General"])
async def global_health():
    """Global API health check."""
    return {"status": "ok", "service": config.PROJECT_NAME, "version": config.VERSION}


if __name__ == "__main__":
    # Allows running directly via `python backend/app/main.py`
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )

