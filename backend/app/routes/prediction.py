"""
API router endpoints for Salary Prediction.
"""

from typing import Any, Dict,Optional
from fastapi import APIRouter, HTTPException, status

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



@router.post(
    "/predict",
    response_model=SalaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Job Salary",
    description="Calculates estimated salary and expected confidence range.",
)
def predict_salary_endpoint(request: SalaryPredictionRequest):
    try:
        data_dict = request.model_dump(exclude_unset=True)
        result = prediction_service.predict(data_dict)

        # Attach real-time TreeSHAP explanation
        try:
            result["shap_explanation"] = shap_service.explain_prediction(data_dict)
        except Exception as ex:
            print(f"[PredictionRouter] SHAP explanation warning: {ex}")
            result["shap_explanation"] = None

        # Automatically save successful predictions
        history_service.save_prediction(
            request_data=result.get("input_features", data_dict),
            prediction_result=result,
        )

        return result

    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(fnf),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchSalaryPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Predict Job Salaries",
    description="Vectorized prediction for multiple job profiles simultaneously.",
)
def predict_batch_endpoint(batch_request: BatchSalaryPredictionRequest):
    try:
        profiles_data = [p.model_dump(exclude_unset=True) for p in batch_request.profiles]
        predictions = prediction_service.predict_batch(profiles_data)
        return {
            "total_count": len(predictions),
            "predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}",
        )


@router.get(
    "/metadata",
    response_model=DropdownOptionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get UI Form Dropdown Options",
    description="Returns valid categories for job titles, experience levels, company sizes, and countries.",
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
    summary="Service Health Check",
)
def health_check_endpoint():
    return prediction_service.health_check()


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Model Feature Importance",
    description="Returns the Top influential features calculated directly from the trained Random Forest model with percentage contributions.",
)
def get_feature_importance_endpoint(top_n: int = 10):
    try:
        return prediction_service.get_feature_importance(top_n=top_n)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate feature importance: {str(e)}",
        )

# GET all prediction history
@router.get("/history", summary="Get Prediction History")
def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    experience_level: Optional[str] = None,
):
    return {
        "total": history_service.get_total_count(
            search=search,
            experience_level=experience_level,
        ),
        "history": history_service.get_history(
            limit=limit,
            offset=offset,
            search=search,
            experience_level=experience_level,
        ),
    }


# GET prediction statistics
@router.get("/history/summary", summary="Get History Statistics")
def get_history_summary():
    return history_service.get_summary_stats()


# GET one prediction by ID
@router.get("/history/{prediction_id}", summary="Get Prediction Details")
def get_prediction_details(prediction_id: int):
    prediction = history_service.get_prediction_by_id(prediction_id)

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found",
        )

    # Attach SHAP explanation if not already present
    if not prediction.get("shap_explanation"):
        try:
            features = prediction.get("input_features") or prediction
            prediction["shap_explanation"] = shap_service.explain_prediction(features)
        except Exception as e:
            logger.warning(f"Could not compute SHAP for historical record {prediction_id}: {e}")

    return prediction


# DELETE one prediction
@router.delete("/history/{prediction_id}", summary="Delete Prediction")
def delete_prediction_history(prediction_id: int):
    deleted = history_service.delete_prediction(prediction_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Prediction record not found",
        )

    return {"message": "Prediction deleted successfully"}


# DELETE all history
@router.delete("/history", summary="Clear Prediction History")
def clear_prediction_history():
    deleted_count = history_service.clear_history()

    return {
        "message": "Prediction history cleared successfully",
        "deleted_count": deleted_count,
    }


# ==============================================================================
# Explainable AI (SHAP) Endpoints
# ==============================================================================


@router.post(
    "/explain",
    status_code=status.HTTP_200_OK,
    summary="Explain Prediction with SHAP",
    description="Computes exact local TreeSHAP attribution and waterfall breakdown for an individual prediction.",
)
def explain_prediction_endpoint(request: SalaryPredictionRequest):
    try:
        data_dict = request.model_dump(exclude_unset=True)
        return shap_service.explain_prediction(data_dict)
    except Exception as e:
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
        return shap_service.explain_prediction(features)
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
    summary="Get Global SHAP Feature Importance",
    description="Returns global TreeSHAP and feature importance overview.",
)
def explain_global_endpoint(top_n: int = 10):
    try:
        return shap_service.get_global_shap_importance(top_n=top_n)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global explanation error: {str(e)}",
        )
