"""
Pydantic schemas for request validation and response serialization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SalaryPredictionRequest(BaseModel):
    """Input features required to predict annual salary."""

    job_title: str = Field(
        default="Data Scientist",
        description="Target job title (e.g. Data Scientist, Data Engineer, Machine Learning Engineer)",
    )
    experience_level: str = Field(
        default="SE",
        description="Experience level tier: EN (Entry), MI (Mid), SE (Senior), EX (Executive)",
    )
    employment_type: Optional[str] = Field(
        default="FT",
        description="Employment arrangement: FT (Full-Time), PT (Part-Time), CT (Contract), FL (Freelance)",
    )
    company_location: Optional[str] = Field(
        default="US",
        description="2-letter ISO country code of employer (e.g. US, GB, CA, DE, IN)",
    )
    employee_residence: Optional[str] = Field(
        default=None,
        description="2-letter ISO country code of employee residence (defaults to company_location)",
    )
    company_size: Optional[str] = Field(
        default="M",
        description="Company size: S (<50 employees), M (50-250 employees), L (>250 employees)",
    )
    remote_ratio: Optional[int] = Field(
        default=100,
        description="Remote work percentage: 0 (On-site), 50 (Hybrid), 100 (Fully Remote)",
    )
    work_year: Optional[int] = Field(
        default=2023,
        description="Work year for trend adjustment (e.g. 2023)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_title": "Data Scientist",
                "experience_level": "SE",
                "employment_type": "FT",
                "company_location": "US",
                "employee_residence": "US",
                "company_size": "M",
                "remote_ratio": 100,
                "work_year": 2023,
            }
        }
    }


class SalaryRange(BaseModel):
    """Estimated confidence interval range for prediction."""

    min: float
    max: float
    formatted_min: str
    formatted_max: str


class SalaryPredictionResponse(BaseModel):
    """Output schema returned from salary prediction."""

    predicted_salary_usd: float
    formatted_salary: str
    salary_range: SalaryRange
    confidence_margin_mae: float
    input_features: Dict[str, Any]
    model_version: str
    shap_explanation: Optional[Dict[str, Any]] = None


class BatchSalaryPredictionRequest(BaseModel):
    """Batch input schema for predicting multiple job profiles simultaneously."""

    profiles: List[SalaryPredictionRequest]


class BatchSalaryPredictionResponse(BaseModel):
    """Batch output response schema."""

    total_count: int
    predictions: List[SalaryPredictionResponse]


class DropdownOptionsResponse(BaseModel):
    """Schema containing valid categories for populating UI dropdown selectors."""

    job_titles: List[str]
    experience_levels: Dict[str, str]
    employment_types: Dict[str, str]
    company_sizes: Dict[str, str]
    remote_ratios: Dict[int, str]
    top_countries: List[Dict[str, str]]



class FeatureImportanceItem(BaseModel):
    """Individual feature importance metric item."""

    feature: str
    raw_feature: str
    importance: float
    percentage: float
    category: str


class FeatureImportanceResponse(BaseModel):
    """Feature importance analysis response schema."""

    model_name: str
    top_features: List[FeatureImportanceItem]
    summary_explanation: str
    total_features_evaluated: int


class PredictionHistoryItem(BaseModel):
    """Individual prediction history entry from SQLite storage."""

    id: int
    timestamp: str
    job_title: str
    experience_level: str
    employment_type: str
    company_location: str
    employee_residence: str
    company_size: str
    remote_ratio: int
    work_year: int
    predicted_salary_usd: float
    formatted_salary: str
    salary_range: SalaryRange
    confidence_margin_mae: float
    model_version: str
    input_features: Dict[str, Any]


class PredictionHistoryListResponse(BaseModel):
    """Paginated list of prediction history entries."""

    total_count: int
    limit: int
    offset: int
    history: List[PredictionHistoryItem]


class PredictionHistoryStatsResponse(BaseModel):
    """Aggregated analytical statistics across all stored predictions."""

    total_predictions: int
    average_predicted_salary: float
    formatted_avg_salary: str
    min_predicted_salary: float
    max_predicted_salary: float
    most_frequent_role: Optional[str] = None
    top_experience_level: Optional[str] = None
    top_locations: List[Dict[str, Any]] = []
    remote_distribution: Dict[str, int] = {}
    last_prediction_at: Optional[str] = None
