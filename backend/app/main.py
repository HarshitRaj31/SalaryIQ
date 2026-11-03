"""
Main FastAPI Application Entrypoint for Job Salary Prediction Backend.
Supports dual-country models (India and United States).
"""

import os
import sys
import uvicorn
from typing import Optional
from fastapi import FastAPI, Query
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
    allow_origins=[origin.rstrip("/") for origin in config.CORS_ORIGINS],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(prediction_router, prefix=config.API_V1_STR)
app.include_router(prediction_router, prefix="/api")  # Convenience alias
app.include_router(prediction_router, prefix="")      # Root alias (/prediction/predict)

# Direct endpoint aliases for frontend convenience
from app.routes.prediction import (
    clear_prediction_history,
    delete_prediction_history,
    explain_global_endpoint,
    explain_historical_prediction_endpoint,
    explain_prediction_endpoint,
    get_feature_importance_endpoint,
    get_history_summary,
    get_metadata_endpoint,
    get_prediction_details,
    get_prediction_history,
    predict_salary_endpoint,
)
from app.schemas.salary import (
    DropdownOptionsResponse,
    FeatureImportanceResponse,
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)


@app.post("/predict", response_model=SalaryPredictionResponse, tags=["Salary Prediction"])
def root_predict_alias(request: SalaryPredictionRequest):
    """Direct alias to calculate predicted salary (US or India)."""
    return predict_salary_endpoint(request)


@app.get("/metadata", response_model=DropdownOptionsResponse, tags=["Salary Prediction"])
def root_metadata_alias():
    """Direct alias to fetch UI form dropdown options for India and US."""
    return get_metadata_endpoint()


@app.get("/feature-importance", response_model=FeatureImportanceResponse, tags=["Salary Prediction"])
def root_feature_importance_alias(
    top_n: int = Query(default=10, ge=1, le=50),
    country: str = Query(default="US", description="Country code: 'US' or 'IN'"),
):
    """Direct alias to fetch model feature importance for US or India."""
    return get_feature_importance_endpoint(top_n=top_n, country=country)


@app.get("/history", tags=["Prediction History"])
def root_history_alias(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    experience_level: Optional[str] = None,
    country: Optional[str] = Query(default=None, description="Filter history by country: 'US' or 'IN'"),
):
    """Direct alias to fetch paginated prediction history from SQLite."""
    return get_prediction_history(
        limit=limit,
        offset=offset,
        search=search,
        experience_level=experience_level,
        country=country,
    )


@app.get("/history/summary", tags=["Prediction History"])
def root_history_summary_alias(
    country: Optional[str] = Query(default=None, description="Optional filter by country: 'US' or 'IN'"),
):
    """Direct alias to fetch aggregated history statistics from SQLite."""
    return get_history_summary(country=country)


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
def root_explain_global_alias(
    top_n: int = Query(default=10, ge=1, le=50),
    country: str = Query(default="US", description="Country code: 'US' or 'IN'"),
):
    """Direct alias to fetch global SHAP feature importance overview."""
    return explain_global_endpoint(top_n=top_n, country=country)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint welcoming users and providing documentation links."""
    return JSONResponse(
        content={
            "service": config.PROJECT_NAME,
            "version": config.VERSION,
            "status": "online",
            "supported_countries": ["US", "IN"],
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
    """Global API health check reporting both India and US models."""
    return {
        "status": "ok",
        "service": config.PROJECT_NAME,
        "version": config.VERSION,
        "details": prediction_service.health_check(),
    }


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
    )
