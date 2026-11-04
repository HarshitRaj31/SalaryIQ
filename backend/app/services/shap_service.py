"""
SalaryIQ Explainable AI Service
Dual-country SHAP-based local and global salary prediction explanations.
Supports India (Gradient Boosting in INR / LPA) and United States (Random Forest in USD / year).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir.parents[2] / "ml" / "venv" / "Lib" / "site-packages",
        Path(r"c:\Users\iamha\OneDrive\Desktop\Job\Job-Salary-Prediction\ml\venv\Lib\site-packages"),
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
            break
    import shap

from app.core import config

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# HUMAN-READABLE METADATA & LABELS
# -----------------------------------------------------------------------------
EXP_LABELS = {
    "EN": "Entry-level (EN)",
    "MI": "Mid-level (MI)",
    "SE": "Senior-level (SE)",
    "EX": "Executive (EX)",
}

EMP_LABELS = {
    "FT": "Full-Time (FT)",
    "PT": "Part-Time (PT)",
    "CT": "Contract (CT)",
    "FL": "Freelance (FL)",
}

SIZE_LABELS = {
    "S": "Small (< 50)",
    "M": "Medium (50 - 250)",
    "L": "Large (> 250)",
}

COUNTRY_NAMES = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "DE": "Germany",
    "IN": "India",
    "FR": "France",
    "ES": "Spain",
    "AU": "Australia",
    "NL": "Netherlands",
    "BR": "Brazil",
}

US_FEATURE_METADATA = {
    "experience_level": {"title": "Experience Seniority", "category": "Seniority"},
    "employee_residence": {"title": "Employee Residence", "category": "Geography"},
    "company_location": {"title": "Company Location", "category": "Geography"},
    "job_title": {"title": "Job Title / Role", "category": "Role"},
    "work_year": {"title": "Work Year & Market Cycle", "category": "Temporal"},
    "remote_ratio": {"title": "Workplace Setting", "category": "Workplace"},
    "company_size": {"title": "Company Size / Scale", "category": "Company"},
    "employment_type": {"title": "Employment Contract Type", "category": "Contract"},
}

INDIA_FEATURE_METADATA = {
    "experience_avg_yrs": {"title": "Years of Professional Experience", "category": "Experience"},
    "role_standard": {"title": "Job Role / Specialization", "category": "Role"},
    "primary_city_clean": {"title": "Tech Hub City", "category": "Geography"},
    "experience_level": {"title": "Seniority Tier", "category": "Seniority"},
    "company_rating": {"title": "Employer Reputation Rating", "category": "Company"},
    "skills_count": {"title": "Technical Skillset Depth", "category": "Competency"},
    "company_size": {"title": "Company Scale", "category": "Company"},
    "remote_ratio": {"title": "Workplace Flexibility", "category": "Workplace"},
    "work_year": {"title": "Market Cycle Year", "category": "Temporal"},
    "employment_type": {"title": "Employment Arrangement", "category": "Contract"},
}


def format_money_usd(value: float) -> str:
    """Formats value in USD ($)."""
    return f"${abs(round(value)):,.0f}"


def format_money_inr(value: float) -> str:
    """Formats value in INR LPA (₹)."""
    return f"₹{abs(value):.2f} LPA"


# -----------------------------------------------------------------------------
# SHAP SERVICE CLASS
# -----------------------------------------------------------------------------
class ShapExplainabilityService:
    """
    Dual-country SHAP explainability service supporting:
    - US Model (Random Forest in USD)
    - India Model (Gradient Boosting in INR / LPA)
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            # US model state
            cls._instance._us_initialized = False
            cls._instance._us_pipeline = None
            cls._instance._us_preprocessor = None
            cls._instance._us_regressor = None
            cls._instance._us_explainer = None
            cls._instance._us_base_value = None
            cls._instance._us_feature_names = []
            cls._instance._us_global_cache = None

            # India model state
            cls._instance._india_initialized = False
            cls._instance._india_pipeline = None
            cls._instance._india_preprocessor = None
            cls._instance._india_regressor = None
            cls._instance._india_explainer = None
            cls._instance._india_base_value = None
            cls._instance._india_feature_names = []
            cls._instance._india_global_cache = None
            cls._instance._prediction_cache = {}

        return cls._instance

    # -------------------------------------------------------------------------
    # 1. INITIALIZATION & EXPLAINERS
    # -------------------------------------------------------------------------
    def _initialize_us_explainer(self):
        """Initializes TreeExplainer for the US model pipeline."""
        if self._us_initialized:
            return

        from app.services.prediction_service import prediction_service
        if getattr(prediction_service, '_us_pipeline', None) is not None:
            self._us_pipeline = prediction_service._us_pipeline
        else:
            us_path = config.US_MODEL_PATH
            if not os.path.exists(us_path):
                raise FileNotFoundError(f"US model file not found at: {us_path}")
            self._us_pipeline = joblib.load(us_path)
        self._us_preprocessor = self._us_pipeline.named_steps["preprocessor"]
        self._us_regressor = self._us_pipeline.named_steps["regressor"]
        self._us_feature_names = list(self._us_preprocessor.get_feature_names_out())

        try:
            self._us_explainer = shap.TreeExplainer(self._us_regressor)
            raw_ev = self._us_explainer.expected_value
            self._us_base_value = float(np.asarray(raw_ev).reshape(-1)[0])
            self._us_initialized = True
            logger.info(f"US SHAP initialized. Base value: ${self._us_base_value:,.2f}")
        except Exception as exc:
            self._us_explainer = None
            raise RuntimeError(f"Unable to initialize US TreeSHAP: {exc}") from exc

    def _initialize_india_explainer(self):
        """Initializes TreeExplainer for the India Gradient Boosting model pipeline."""
        if self._india_initialized:
            return

        from app.services.prediction_service import prediction_service
        if getattr(prediction_service, '_india_pipeline', None) is not None:
            self._india_pipeline = prediction_service._india_pipeline
        else:
            india_path = config.INDIA_MODEL_PATH
            if not os.path.exists(india_path):
                raise FileNotFoundError(f"India model file not found at: {india_path}")
            self._india_pipeline = joblib.load(india_path)
        self._india_preprocessor = self._india_pipeline.named_steps["preprocessor"]
        self._india_regressor = self._india_pipeline.named_steps["regressor"]
        self._india_feature_names = list(self._india_preprocessor.get_feature_names_out())

        try:
            self._india_explainer = shap.TreeExplainer(self._india_regressor)
            raw_ev = self._india_explainer.expected_value
            self._india_base_value = float(np.asarray(raw_ev).reshape(-1)[0])
            self._india_initialized = True
            logger.info(f"India SHAP initialized. Base value: ₹{self._india_base_value:.2f} LPA")
        except Exception as exc:
            self._india_explainer = None
            raise RuntimeError(f"Unable to initialize India TreeSHAP: {exc}") from exc

    # -------------------------------------------------------------------------
    # 2. FEATURE GROUPING & DISPLAY
    # -------------------------------------------------------------------------
    @staticmethod
    def _group_shap_values(
        feature_names: List[str],
        shap_values: np.ndarray,
        feature_keys: List[str],
    ) -> Dict[str, float]:
        """Aggregates transformed one-hot encoded SHAP values back into logical input features."""
        grouped = {k: 0.0 for k in feature_keys}

        for f_name, s_val in zip(feature_names, shap_values):
            matched = False
            for k in feature_keys:
                if (
                    f_name == k
                    or f_name.endswith(f"__{k}")
                    or f_name.startswith(f"cat__{k}_")
                    or f_name.startswith(f"num__{k}")
                ):
                    grouped[k] += float(s_val)
                    matched = True
                    break

            if not matched:
                logger.warning(f"Could not map transformed feature: {f_name}")

        return grouped

    @staticmethod
    def _display_value_us(key: str, raw_value: Any) -> str:
        """Formatted human-readable string representation for US feature values."""
        if key == "experience_level":
            return EXP_LABELS.get(raw_value, str(raw_value))
        if key == "employment_type":
            return EMP_LABELS.get(raw_value, str(raw_value))
        if key == "company_size":
            return SIZE_LABELS.get(raw_value, str(raw_value))
        if key in ("employee_residence", "company_location"):
            country = COUNTRY_NAMES.get(raw_value, raw_value)
            return f"{country} ({raw_value})"
        if key == "remote_ratio":
            labels = {100: "100% Fully Remote", 50: "50% Hybrid", 0: "On-site (0%)"}
            return labels.get(raw_value, f"{raw_value}% Remote")
        return str(raw_value)

    @staticmethod
    def _display_value_india(key: str, raw_value: Any) -> str:
        """Formatted human-readable string representation for India feature values."""
        if key == "experience_level":
            return EXP_LABELS.get(raw_value, str(raw_value))
        if key == "experience_avg_yrs":
            return f"{float(raw_value):.1f} Years"
        if key == "company_rating":
            return f"{float(raw_value):.1f} / 5.0"
        if key == "skills_count":
            return f"{int(raw_value)} Verified Skills"
        if key == "company_size":
            return SIZE_LABELS.get(raw_value, str(raw_value))
        if key == "remote_ratio":
            labels = {100: "100% Fully Remote", 50: "50% Hybrid", 0: "On-site (0%)"}
            return labels.get(raw_value, f"{raw_value}% Remote")
        return str(raw_value)

    # -------------------------------------------------------------------------
    # 3. LOCAL PREDICTION EXPLANATION (COUNTRY ROUTING)
    # -------------------------------------------------------------------------
    def explain_prediction(
        self,
        profile: Dict[str, Any],
        country: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes exact local TreeSHAP attribution and waterfall breakdown.
        Routes to India or US model based on country.
        """
        c = str(country or profile.get("country", "")).strip().upper()
        if not c:
            loc = str(profile.get("company_location", "")).strip().upper()
            if loc == "IN" or profile.get("primary_city_clean") or profile.get("role_standard"):
                c = "IN"
            else:
                c = "US"

        # Fast cache lookup for identical prediction inputs (<0.1ms)
        cache_key = f"{c}:{tuple(sorted((k, str(v)) for k, v in profile.items() if k not in ('include_shap', 'shap_explanation')))}"
        if hasattr(self, '_prediction_cache') and cache_key in self._prediction_cache:
            return dict(self._prediction_cache[cache_key])

        if c in ["IN", "IND", "INDIA"]:
            res = self._explain_india_prediction(profile)
        else:
            res = self._explain_us_prediction(profile)

        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
        if len(self._prediction_cache) >= 256:
            self._prediction_cache.clear()
        self._prediction_cache[cache_key] = res
        return res

    def _explain_us_prediction(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Computes TreeSHAP for US prediction in USD."""
        self._initialize_us_explainer()

        clean_profile = {
            "work_year": int(profile.get("work_year", 2023)),
            "experience_level": str(profile.get("experience_level", "SE")),
            "employment_type": str(profile.get("employment_type", "FT")),
            "job_title": str(profile.get("job_title", "Data Scientist")),
            "employee_residence": str(profile.get("employee_residence", "US")),
            "remote_ratio": int(profile.get("remote_ratio", 100)),
            "company_location": str(profile.get("company_location", "US")),
            "company_size": str(profile.get("company_size", "M")),
        }

        df = pd.DataFrame([clean_profile])
        transformed = self._us_preprocessor.transform(df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        prediction = float(self._us_regressor.predict(transformed)[0])

        try:
            exp_obj = self._us_explainer(transformed)
            shap_matrix = exp_obj.values
        except Exception:
            shap_matrix = self._us_explainer.shap_values(transformed)

        raw_shap = np.asarray(shap_matrix).reshape(-1)
        base_value = float(self._us_base_value)

        grouped_shap = self._group_shap_values(
            self._us_feature_names, raw_shap, list(US_FEATURE_METADATA.keys())
        )

        feature_contributions = []
        for key, shap_val in grouped_shap.items():
            metadata = US_FEATURE_METADATA[key]
            raw_val = clean_profile[key]
            direction = "positive" if shap_val >= 0 else "negative"
            pct = (shap_val / abs(base_value) * 100) if base_value != 0 else 0.0
            sign = "+" if shap_val >= 0 else "-"

            feature_contributions.append({
                "field": key,
                "feature_title": metadata["title"],
                "category": metadata["category"],
                "value": self._display_value_us(key, raw_val),
                "raw_value": raw_val,
                "shap_value": round(shap_val, 2),
                "formatted_shap": f"{sign}{format_money_usd(shap_val)}",
                "direction": direction,
                "percentage_impact": round(pct, 2),
                "abs_impact": abs(shap_val),
            })

        feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        top_positive = [f for f in feature_contributions if f["direction"] == "positive"]
        top_negative = [f for f in feature_contributions if f["direction"] == "negative"]

        waterfall_steps = []
        running_total = base_value
        waterfall_steps.append({
            "name": "Base Market Average",
            "feature": "Expected Base Value",
            "value": round(base_value),
            "delta": round(base_value),
            "type": "base",
            "formatted_value": format_money_usd(base_value),
            "formatted_delta": format_money_usd(base_value),
        })

        for feat in feature_contributions:
            sv = feat["shap_value"]
            running_total += sv
            waterfall_steps.append({
                "name": feat["feature_title"],
                "feature": feat["field"],
                "value": round(running_total),
                "delta": round(sv),
                "type": feat["direction"],
                "formatted_value": format_money_usd(running_total),
                "formatted_delta": feat["formatted_shap"],
            })

        waterfall_steps.append({
            "name": "Final Predicted Salary",
            "feature": "Prediction Result",
            "value": round(prediction),
            "delta": 0,
            "type": "total",
            "formatted_value": format_money_usd(prediction),
            "formatted_delta": "$0",
        })

        summary = (
            f"The US base market salary for tech roles starts at {format_money_usd(base_value)} / year. "
            f"For this candidate, {feature_contributions[0]['feature_title']} ({feature_contributions[0]['value']}) "
            f"is the most influential factor, contributing {feature_contributions[0]['formatted_shap']}."
        )

        return {
            "country": "US",
            "currency": "USD",
            "salary_unit": "YEAR",
            "base_salary": round(base_value, 2),
            "predicted_salary": round(prediction, 2),
            "formatted_base": format_money_usd(base_value),
            "formatted_predicted": f"{format_money_usd(prediction)} / year",
            "total_shap_impact": round(float(np.sum(raw_shap)), 2),
            "formatted_total_impact": (
                ("+" if np.sum(raw_shap) >= 0 else "-")
                + format_money_usd(np.sum(raw_shap))
            ),
            "feature_contributions": feature_contributions,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "waterfall": waterfall_steps,
            "summary": summary,
        }

    def _explain_india_prediction(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Computes TreeSHAP for India prediction in INR LPA."""
        self._initialize_india_explainer()

        # Sanitize profile
        exp_lvl = str(profile.get("experience_level", "SE")).strip().upper()
        role = profile.get("role_standard") or profile.get("job_title") or "Data Scientist"
        city = profile.get("primary_city_clean") or profile.get("company_location") or "Bangalore"
        if city.upper() in ["IN", "INDIA"]:
            city = "Bangalore"

        exp_yrs = profile.get("experience_avg_yrs")
        if exp_yrs is None or float(exp_yrs) <= 0.0:
            tier_defaults = {"EN": 1.5, "MI": 4.0, "SE": 7.0, "EX": 12.0}
            experience_avg_yrs = tier_defaults.get(exp_lvl, 5.0)
        else:
            experience_avg_yrs = float(np.clip(float(exp_yrs), 0.0, 30.0))

        rating = profile.get("company_rating")
        company_rating = float(np.clip(float(rating if rating is not None else 3.8), 1.0, 5.0))

        skills = profile.get("skills_count")
        skills_count = int(np.clip(int(skills if skills is not None else 5), 1, 25))

        clean_profile = {
            "work_year": int(profile.get("work_year", 2026)),
            "experience_level": exp_lvl,
            "employment_type": "FT",
            "role_standard": str(role).strip(),
            "primary_city_clean": str(city).strip(),
            "remote_ratio": int(profile.get("remote_ratio", 50)),
            "company_size": str(profile.get("company_size", "L")).strip().upper(),
            "experience_avg_yrs": experience_avg_yrs,
            "company_rating": company_rating,
            "skills_count": skills_count,
        }

        df = pd.DataFrame([clean_profile])
        transformed = self._india_preprocessor.transform(df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)

        prediction = float(self._india_regressor.predict(transformed)[0])

        try:
            exp_obj = self._india_explainer(transformed)
            shap_matrix = exp_obj.values
        except Exception:
            shap_matrix = self._india_explainer.shap_values(transformed)

        raw_shap = np.asarray(shap_matrix).reshape(-1)
        base_value = float(self._india_base_value)

        grouped_shap = self._group_shap_values(
            self._india_feature_names, raw_shap, list(INDIA_FEATURE_METADATA.keys())
        )

        feature_contributions = []
        for key, shap_val in grouped_shap.items():
            metadata = INDIA_FEATURE_METADATA[key]
            raw_val = clean_profile[key]
            direction = "positive" if shap_val >= 0 else "negative"
            pct = (shap_val / abs(base_value) * 100) if base_value != 0 else 0.0
            sign = "+" if shap_val >= 0 else "-"

            feature_contributions.append({
                "field": key,
                "feature_title": metadata["title"],
                "category": metadata["category"],
                "value": self._display_value_india(key, raw_val),
                "raw_value": raw_val,
                "shap_value": round(shap_val, 2),
                "formatted_shap": f"{sign}{format_money_inr(shap_val)}",
                "direction": direction,
                "percentage_impact": round(pct, 2),
                "abs_impact": abs(shap_val),
            })

        feature_contributions.sort(key=lambda x: x["abs_impact"], reverse=True)

        top_positive = [f for f in feature_contributions if f["direction"] == "positive"]
        top_negative = [f for f in feature_contributions if f["direction"] == "negative"]

        waterfall_steps = []
        running_total = base_value
        waterfall_steps.append({
            "name": "Base Market Average",
            "feature": "Expected Base Value",
            "value": round(base_value, 2),
            "delta": round(base_value, 2),
            "type": "base",
            "formatted_value": format_money_inr(base_value),
            "formatted_delta": format_money_inr(base_value),
        })

        for feat in feature_contributions:
            sv = feat["shap_value"]
            running_total += sv
            waterfall_steps.append({
                "name": feat["feature_title"],
                "feature": feat["field"],
                "value": round(running_total, 2),
                "delta": round(sv, 2),
                "type": feat["direction"],
                "formatted_value": format_money_inr(running_total),
                "formatted_delta": feat["formatted_shap"],
            })

        waterfall_steps.append({
            "name": "Final Predicted Salary",
            "feature": "Prediction Result",
            "value": round(prediction, 2),
            "delta": 0,
            "type": "total",
            "formatted_value": format_money_inr(prediction),
            "formatted_delta": "₹0 LPA",
        })

        summary = (
            f"The India baseline average salary across tech roles is {format_money_inr(base_value)}. "
            f"For this candidate, {feature_contributions[0]['feature_title']} ({feature_contributions[0]['value']}) "
            f"is the primary factor, contributing {feature_contributions[0]['formatted_shap']}."
        )

        return {
            "country": "IN",
            "currency": "INR",
            "salary_unit": "LPA",
            "base_salary": round(base_value, 2),
            "predicted_salary": round(prediction, 2),
            "formatted_base": format_money_inr(base_value),
            "formatted_predicted": format_money_inr(prediction),
            "total_shap_impact": round(float(np.sum(raw_shap)), 2),
            "formatted_total_impact": (
                ("+" if np.sum(raw_shap) >= 0 else "-")
                + format_money_inr(np.sum(raw_shap))
            ),
            "feature_contributions": feature_contributions,
            "top_positive": top_positive,
            "top_negative": top_negative,
            "waterfall": waterfall_steps,
            "summary": summary,
        }

    # -------------------------------------------------------------------------
    # 4. GLOBAL SHAP IMPORTANCE
    # -------------------------------------------------------------------------
    def get_global_shap_importance(self, top_n: int = 10, country: str = "US") -> Dict[str, Any]:
        """Returns global feature importance derived directly from the active trained model."""
        c = str(country).strip().upper()
        from app.services.prediction_service import prediction_service

        return prediction_service.get_feature_importance(top_n=top_n, country=c)


# Global service instance
shap_service = ShapExplainabilityService()
