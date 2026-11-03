"""
Pydantic schemas for dual-country request validation and response serialization.
Supports India (INR / LPA) and United States (USD / year) prediction pipelines.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SalaryPredictionRequest(BaseModel):
    """
    Input features required to predict salary.
    Supports country selection: 'US' (United States) or 'IN' (India).
    """

    country: Optional[str] = Field(
        default="US",
        description="Target country code: 'US' (United States) or 'IN' (India)",
    )

    # Common & US Features
    job_title: Optional[str] = Field(
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
        description="Work year for trend adjustment (e.g. 2023 or 2026)",
    )

    # India-Specific Features (Optional with smart defaults if country='IN')
    role_standard: Optional[str] = Field(
        default=None,
        description="Standardized tech role in India (e.g. Data Scientist, Data Analyst, Machine Learning Engineer)",
    )
    primary_city_clean: Optional[str] = Field(
        default=None,
        description="Primary Indian tech hub (e.g. Bangalore, Hyderabad, Pune, Mumbai, Delhi / NCR, Remote)",
    )
    experience_avg_yrs: Optional[float] = Field(
        default=None,
        description="Total average years of professional experience (e.g. 5.0)",
    )
    company_rating: Optional[float] = Field(
        default=None,
        description="Company glassdoor/ambitionbox rating (1.0 - 5.0, e.g. 4.2)",
    )
    skills_count: Optional[int] = Field(
        default=None,
        description="Count of key technical skills possessed (e.g. 6)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "country": "US",
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
    """
    Output schema returned from salary prediction.
    Maintains backward compatibility with older fields while supplying dual-country metadata.
    """

    country: str = Field(default="US", description="Predicted country code: 'US' or 'IN'")
    predicted_salary: float = Field(description="Numerical predicted salary in primary currency unit")
    currency: str = Field(default="USD", description="Currency identifier: 'USD' or 'INR'")
    salary_unit: str = Field(default="YEAR", description="Salary unit: 'YEAR' (USD/year) or 'LPA' (Lakhs Per Annum)")
    formatted_salary: str = Field(description="Human-readable formatted salary string (e.g. '$120,000 / year' or '₹15.5 LPA')")
    model_used: str = Field(description="Name of machine learning model used for prediction")

    # Backward compatibility fields for existing frontend and reports
    predicted_salary_usd: float = Field(description="Equivalent or direct salary in USD (backward compatible)")
    salary_range: SalaryRange = Field(description="Estimated prediction interval based on model MAE")
    confidence_margin_mae: float = Field(description="Mean Absolute Error of the active model")
    input_features: Dict[str, Any] = Field(description="Echo of validated input features used by the model")
    model_version: str = Field(description="Filename/identifier of the active model")
    shap_explanation: Optional[Dict[str, Any]] = Field(default=None, description="SHAP feature attribution breakdown")


class BatchSalaryPredictionRequest(BaseModel):
    """Batch input schema for predicting multiple job profiles simultaneously."""

    country: Optional[str] = "US"
    profiles: List[SalaryPredictionRequest]


class BatchSalaryPredictionResponse(BaseModel):
    """Batch output response schema."""

    total_count: int
    predictions: List[SalaryPredictionResponse]


class DropdownOptionsResponse(BaseModel):
    """Schema containing valid categories for populating UI dropdown selectors for both countries."""

    # US / Global Options
    job_titles: List[str]
    experience_levels: Dict[str, str]
    employment_types: Dict[str, str]
    company_sizes: Dict[str, str]
    remote_ratios: Dict[int, str]
    top_countries: List[Dict[str, str]]

    # India Options
    india_roles: Optional[List[str]] = None
    india_cities: Optional[List[str]] = None
    india_experience_levels: Optional[Dict[str, str]] = None
    india_company_sizes: Optional[Dict[str, str]] = None
    supported_countries: Optional[List[Dict[str, str]]] = None


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
    country: Optional[str] = "US"
    currency: Optional[str] = "USD"
    salary_unit: Optional[str] = "YEAR"
    top_features: List[FeatureImportanceItem]
    summary_explanation: str
    total_features_evaluated: int


class PredictionHistoryItem(BaseModel):
    """Individual prediction history entry from SQLite storage."""

    id: int
    timestamp: str
    country: Optional[str] = "US"
    currency: Optional[str] = "USD"
    salary_unit: Optional[str] = "YEAR"
    model_used: Optional[str] = None
    predicted_salary: Optional[float] = None
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
    country_breakdown: Optional[Dict[str, int]] = None
