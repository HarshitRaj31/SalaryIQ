"""
API router endpoints for Dual-Country Salary Prediction (India & United States).
"""

import logging
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, Query, status

logger = logging.getLogger(__name__)

from app.schemas.salary import (
    BatchSalaryPredictionRequest,
    BatchSalaryPredictionResponse,
    DropdownOptionsResponse,
    FeatureImportanceResponse,
    SalaryPredictionRequest,
    SalaryPredictionResponse,
)
from app.services.prediction_service import prediction_service
from app.services.history_service import history_service
from app.services.shap_service import shap_service

router = APIRouter(prefix="/prediction", tags=["Salary Prediction"])


# ==============================================================================
# 1. CORE PREDICTION ENDPOINTS
# ==============================================================================

@router.post(
    "/predict",
    response_model=SalaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Job Salary (India or US)",
    description="Predicts estimated salary and confidence range based on selected country (US or IN). Optimized for <5ms inference.",
)
def predict_salary_endpoint(
    request: SalaryPredictionRequest,
    include_shap: bool = Query(
        default=False,
        description="Whether to synchronously compute SHAP explanation (defaults to False for fast inference).",
    ),
):
    try:
        data_dict = request.model_dump(exclude_unset=True)

        # Validate country code if explicitly supplied
        if request.country:
            c = request.country.strip().upper()
            if c not in ["US", "USA", "UNITED STATES", "IN", "IND", "INDIA"]:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Unsupported country code '{request.country}'. Supported countries are 'US' (United States) and 'IN' (India).",
                )

        # Validate numerical ranges
        if request.work_year is not None and (request.work_year < 2000 or request.work_year > 2030):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid work_year '{request.work_year}'. Must be between 2000 and 2030.",
            )

        if request.remote_ratio is not None and request.remote_ratio not in [0, 50, 100]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid remote_ratio '{request.remote_ratio}'. Must be 0, 50, or 100.",
            )

        if request.experience_avg_yrs is not None and (request.experience_avg_yrs < 0 or request.experience_avg_yrs > 40):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid experience_avg_yrs '{request.experience_avg_yrs}'. Must be between 0.0 and 40.0.",
            )

        if request.company_rating is not None and (request.company_rating < 1.0 or request.company_rating > 5.0):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid company_rating '{request.company_rating}'. Must be between 1.0 and 5.0.",
            )

        # Perform country-routed prediction (<5ms pure ML inference) with isolated timing
        t_pred_start = time.perf_counter()
        result = prediction_service.predict(data_dict)
        pred_time_ms = round((time.perf_counter() - t_pred_start) * 1000, 3)
        result["prediction_time_ms"] = pred_time_ms

        # Compute SHAP explanation only if requested or explicit flag is True
        compute_shap = include_shap or getattr(request, "include_shap", False)
        if compute_shap:
            try:
                t_shap_start = time.perf_counter()
                result["shap_explanation"] = shap_service.explain_prediction(
                    data_dict, country=result.get("country")
                )
                shap_time_ms = round((time.perf_counter() - t_shap_start) * 1000, 3)
                result["shap_calculation_time_ms"] = shap_time_ms
                logger.info(
                    f"[PredictionRouter] Prediction time: {pred_time_ms:.2f}ms | SHAP calculation time: {shap_time_ms:.2f}ms"
                )
            except Exception as ex:
                logger.warning(f"[PredictionRouter] SHAP explanation warning: {ex}")
                result["shap_explanation"] = None
                result["shap_calculation_time_ms"] = None
        else:
            result["shap_explanation"] = None
            result["shap_calculation_time_ms"] = None
            logger.info(
                f"[PredictionRouter] Pure ML prediction completed in {pred_time_ms:.2f}ms (SHAP calculation omitted)."
            )

        # Automatically save successful predictions
        history_service.save_prediction(
            request_data=result.get("input_features", data_dict),
            prediction_result=result,
        )

        return result

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf),
        )
    except Exception as e:
        logger.error(f"[PredictionRouter] Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchSalaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Predict Job Salaries",
    description="Vectorized prediction for multiple job profiles simultaneously across countries.",
)
def predict_batch_endpoint(batch_request: BatchSalaryPredictionRequest):
    try:
        profiles_data = [p.model_dump(exclude_unset=True) for p in batch_request.profiles]
        predictions = prediction_service.predict_batch(
            profiles_data, country=batch_request.country
        )
        return {
            "total_count": len(predictions),
            "predictions": predictions,
        }
    except Exception as e:
        logger.error(f"[PredictionRouter] Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )


@router.get(
    "/metadata",
    response_model=DropdownOptionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get UI Form Dropdown Options (India & US)",
    description="Returns valid categories for roles, experience levels, company sizes, and cities for both India and US.",
)
def get_metadata_endpoint():
    try:
        return prediction_service.get_metadata()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metadata: {str(e)}",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Service Health Check (Both Models)",
)
def health_check_endpoint():
    return prediction_service.health_check()


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Feature Importance (Country-Aware)",
    description="Returns influential features calculated directly from the active trained model (India GB or US RF).",
)
def get_feature_importance_endpoint(
    top_n: int = Query(default=10, ge=1, le=50),
    country: str = Query(default="US", description="Country code: 'US' or 'IN'"),
):
    try:
        return prediction_service.get_feature_importance(top_n=top_n, country=country)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate feature importance: {str(e)}",
        )


# ==============================================================================
# 2. PREDICTION HISTORY ENDPOINTS
# ==============================================================================

@router.get("/history", summary="Get Prediction History")
def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    experience_level: Optional[str] = None,
    country: Optional[str] = Query(default=None, description="Filter history by country: 'US' or 'IN'"),
):
    return {
        "total": history_service.get_total_count(
            search=search,
            experience_level=experience_level,
            country=country,
        ),
        "history": history_service.get_history(
            limit=limit,
            offset=offset,
            search=search,
            experience_level=experience_level,
            country=country,
        ),
    }


@router.get("/history/summary", summary="Get History Statistics")
def get_history_summary(
    country: Optional[str] = Query(default=None, description="Optional filter by country: 'US' or 'IN'"),
):
    return history_service.get_summary_stats(country=country)


@router.get("/history/{prediction_id}", summary="Get Prediction Details")
def get_prediction_details(prediction_id: int):
    prediction = history_service.get_prediction_by_id(prediction_id)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found",
        )

    # Attach SHAP explanation using record's country if not present
    if not prediction.get("shap_explanation"):
        try:
            features = prediction.get("input_features") or prediction
            prediction["shap_explanation"] = shap_service.explain_prediction(
                features, country=prediction.get("country", "US")
            )
        except Exception as e:
            logger.warning(f"Could not compute SHAP for historical record {prediction_id}: {e}")

    return prediction


@router.delete("/history/{prediction_id}", summary="Delete Prediction")
def delete_prediction_history(prediction_id: int):
    deleted = history_service.delete_prediction(prediction_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found",
        )

    return {"message": "Prediction deleted successfully"}


@router.delete("/history", summary="Clear Prediction History")
def clear_prediction_history():
    deleted_count = history_service.clear_history()

    return {
        "message": "Prediction history cleared successfully",
        "deleted_count": deleted_count,
    }


# ==============================================================================
# 3. EXPLAINABLE AI (SHAP) ENDPOINTS
# ==============================================================================

@router.post(
    "/explain",
    status_code=status.HTTP_200_OK,
    summary="Explain Prediction with SHAP (Country-Aware)",
    description="Computes local TreeSHAP attribution and waterfall breakdown for an individual prediction.",
)
def explain_prediction_endpoint(request: SalaryPredictionRequest):
    try:
        t_start = time.perf_counter()
        data_dict = request.model_dump(exclude_unset=True)
        explanation = shap_service.explain_prediction(data_dict, country=request.country)
        shap_time_ms = round((time.perf_counter() - t_start) * 1000, 3)
        explanation["shap_calculation_time_ms"] = shap_time_ms
        logger.info(f"[SHAP] Dedicated explanation calculated in {shap_time_ms:.2f}ms.")
        return explanation
    except Exception as e:
        logger.error(f"[SHAP] Explanation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation error: {str(e)}",
        )


@router.get(
    "/history/{prediction_id}/explain",
    status_code=status.HTTP_200_OK,
    summary="Explain Historical Prediction with SHAP",
    description="Loads a historical prediction from SQLite and computes its exact TreeSHAP attribution.",
)
def explain_historical_prediction_endpoint(prediction_id: int):
    try:
        record = history_service.get_prediction_by_id(prediction_id)
        if not record:
            raise HTTPException(status_code=404, detail="Prediction record not found")
        features = record.get("input_features") or record
        return shap_service.explain_prediction(features, country=record.get("country", "US"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP explanation error: {str(e)}",
        )


@router.get(
    "/explain/global",
    status_code=status.HTTP_200_OK,
    summary="Get Global SHAP Feature Importance (Country-Aware)",
    description="Returns global TreeSHAP and feature importance overview for specified country (US or IN).",
)
def explain_global_endpoint(
    top_n: int = Query(default=10, ge=1, le=50),
    country: str = Query(default="US", description="Country code: 'US' or 'IN'"),
):
    try:
        return shap_service.get_global_shap_importance(top_n=top_n, country=country)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global explanation error: {str(e)}",
        )
