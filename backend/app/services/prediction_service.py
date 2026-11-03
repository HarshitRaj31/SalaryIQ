"""
Prediction service business logic layer for dual-country salary prediction.
Handles independent model lifecycle, input normalization, inference, and feature metadata
for India (INR / LPA) and the United States (USD / year).
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core import config

logger = logging.getLogger(__name__)

# Ensure ml/src/ is on sys.path for fallback imports
ML_SRC = os.path.join(config.ML_DIR, "src")
if ML_SRC not in sys.path:
    sys.path.insert(0, ML_SRC)


class PredictionService:
    """
    Singleton service managing dual-country salary prediction pipelines:
    1. India Model: Gradient Boosting Regressor predicting salary in INR / LPA
    2. US Model: Random Forest Regressor predicting salary in USD / year
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._us_pipeline = None
            cls._instance._india_pipeline = None
            cls._instance._us_metadata = None
            cls._instance._india_metadata = None
            cls._instance._us_mae = 39038.61
            cls._instance._india_mae = 5.27
            cls._instance._us_model_name = "Random Forest Regressor"
            cls._instance._india_model_name = "Gradient Boosting Regressor"
            cls._instance._initialize_models()
        return cls._instance

    # -------------------------------------------------------------------------
    # 1. MODEL INITIALIZATION & LOADING
    # -------------------------------------------------------------------------
    def _initialize_models(self):
        """Loads both US and India models independently."""
        self._load_us_model()
        self._load_india_model()

    def _load_us_model(self):
        """Loads and caches the US model pipeline and error margin."""
        if self._us_pipeline is not None:
            return

        us_path = config.US_MODEL_PATH
        if not os.path.exists(us_path):
            logger.warning(f"[PredictionService] US model artifact not found at: {us_path}")
            return

        try:
            self._us_pipeline = joblib.load(us_path)
            logger.info(f"[PredictionService] Loaded US model from: {us_path}")

            if os.path.exists(config.US_METADATA_PATH):
                with open(config.US_METADATA_PATH, "r", encoding="utf-8") as f:
                    self._us_metadata = json.load(f)

            if os.path.exists(config.US_COMPARISON_PATH):
                with open(config.US_COMPARISON_PATH, "r", encoding="utf-8") as f:
                    comp = json.load(f)
                    self._us_model_name = comp.get("best_model", "Random Forest Regressor")
                    self._us_mae = float(comp.get("metrics", {}).get(self._us_model_name, {}).get("MAE", 39038.61))
        except Exception as e:
            logger.error(f"[PredictionService] Failed to load US model: {e}")

    def _load_india_model(self):
        """Loads and caches the India model pipeline and error margin."""
        if self._india_pipeline is not None:
            return

        india_path = config.INDIA_MODEL_PATH
        if not os.path.exists(india_path):
            logger.warning(f"[PredictionService] India model artifact not found at: {india_path}")
            return

        try:
            self._india_pipeline = joblib.load(india_path)
            logger.info(f"[PredictionService] Loaded India model from: {india_path}")

            if os.path.exists(config.INDIA_METADATA_PATH):
                with open(config.INDIA_METADATA_PATH, "r", encoding="utf-8") as f:
                    self._india_metadata = json.load(f)

            if os.path.exists(config.INDIA_COMPARISON_PATH):
                with open(config.INDIA_COMPARISON_PATH, "r", encoding="utf-8") as f:
                    comp = json.load(f)
                    self._india_model_name = comp.get("best_model", "Gradient Boosting Regressor")
                    self._india_mae = float(comp.get("metrics", {}).get(self._india_model_name, {}).get("MAE", 5.27))
        except Exception as e:
            logger.error(f"[PredictionService] Failed to load India model: {e}")

    # -------------------------------------------------------------------------
    # 2. COUNTRY ROUTING & PREDICTION
    # -------------------------------------------------------------------------
    def predict(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified prediction entrypoint.
        Routes to India or US model based on 'country' parameter.
        """
        country = self._resolve_country_code(feature_data)

        if country == "IN":
            return self.predict_india(feature_data)
        elif country == "US":
            return self.predict_us(feature_data)
        else:
            raise ValueError(
                f"Unsupported country '{country}'. Supported country codes are 'US' (United States) and 'IN' (India)."
            )

    def _resolve_country_code(self, data: Dict[str, Any]) -> str:
        """Determines the target country from explicit country field or fallback heuristics."""
        raw_country = data.get("country")

        if raw_country is not None and str(raw_country).strip():
            c = str(raw_country).strip().upper()
            if c in ["IN", "IND", "INDIA"]:
                return "IN"
            if c in ["US", "USA", "UNITED STATES"]:
                return "US"
            return c  # Return unknown country code as-is for validation error

        # Heuristic fallbacks if country is not explicitly provided
        loc = str(data.get("company_location", "")).strip().upper()
        res = str(data.get("employee_residence", "")).strip().upper()
        if loc == "IN" or res == "IN" or data.get("primary_city_clean") or data.get("role_standard"):
            return "IN"

        return "US"

    # -------------------------------------------------------------------------
    # 3. UNITED STATES PREDICTION (USD / YEAR)
    # -------------------------------------------------------------------------
    def sanitize_us_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validates, normalizes, and applies defaults for US prediction features."""
        company_loc = str(raw_input.get("company_location", "US")).strip().upper()
        employee_res = str(raw_input.get("employee_residence", company_loc)).strip().upper()

        return {
            "work_year": int(raw_input.get("work_year", 2023)),
            "experience_level": str(raw_input.get("experience_level", "SE")).strip().upper(),
            "employment_type": str(raw_input.get("employment_type", "FT")).strip().upper(),
            "job_title": str(raw_input.get("job_title", "Data Scientist")).strip(),
            "employee_residence": employee_res,
            "company_location": company_loc,
            "company_size": str(raw_input.get("company_size", "M")).strip().upper(),
            "remote_ratio": int(raw_input.get("remote_ratio", 100)),
        }

    def predict_us(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs salary prediction using the US Random Forest model pipeline."""
        if self._us_pipeline is None:
            self._load_us_model()
            if self._us_pipeline is None:
                raise FileNotFoundError(f"US model artifact not found at: {config.US_MODEL_PATH}")

        sanitized = self.sanitize_us_input(feature_data)
        df_input = pd.DataFrame([sanitized])

        raw_pred = float(self._us_pipeline.predict(df_input)[0])
        pred_salary = max(10000.0, raw_pred)

        lower_bound = max(5000.0, pred_salary - self._us_mae)
        upper_bound = pred_salary + self._us_mae

        formatted_salary = f"${round(pred_salary):,} / year"

        return {
            "country": "US",
            "predicted_salary": round(pred_salary, 2),
            "currency": "USD",
            "salary_unit": "YEAR",
            "formatted_salary": formatted_salary,
            "model_used": self._us_model_name,
            "predicted_salary_usd": round(pred_salary, 2),
            "salary_range": {
                "min": round(lower_bound, 2),
                "max": round(upper_bound, 2),
                "formatted_min": f"${round(lower_bound):,}",
                "formatted_max": f"${round(upper_bound):,}",
            },
            "confidence_margin_mae": round(self._us_mae, 2),
            "input_features": sanitized,
            "model_version": os.path.basename(config.US_MODEL_PATH),
        }

    # -------------------------------------------------------------------------
    # 4. INDIA PREDICTION (INR / LPA)
    # -------------------------------------------------------------------------
    def sanitize_india_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validates, normalizes, and applies defaults for India prediction features."""
        # 1. Experience level
        exp_lvl = str(raw_input.get("experience_level", "SE")).strip().upper()
        if exp_lvl not in ["EN", "MI", "SE", "EX"]:
            exp_lvl = "SE"

        # 2. Standardized Tech Role
        role = raw_input.get("role_standard") or raw_input.get("job_title") or "Data Scientist"
        role_map = {
            "data scientist": "Data Scientist",
            "data analyst": "Data Analyst",
            "data engineer": "Data Engineer",
            "machine learning engineer": "Machine Learning Engineer",
            "ml engineer": "Machine Learning Engineer",
            "business analyst": "Business Analyst",
            "python developer": "Python Developer",
        }
        role_standard = role_map.get(str(role).strip().lower(), str(role).strip())

        # 3. Primary Indian City
        city = raw_input.get("primary_city_clean") or raw_input.get("company_location") or "Bangalore"
        if city.upper() in ["IN", "INDIA"]:
            city = "Bangalore"
        primary_city = str(city).strip()

        # 4. Experience Years (intelligent default based on tier)
        exp_yrs = raw_input.get("experience_avg_yrs")
        if exp_yrs is None or float(exp_yrs) <= 0.0:
            tier_defaults = {"EN": 1.5, "MI": 4.0, "SE": 7.0, "EX": 12.0}
            experience_avg_yrs = tier_defaults.get(exp_lvl, 5.0)
        else:
            experience_avg_yrs = float(np.clip(float(exp_yrs), 0.0, 30.0))

        # 5. Company Rating (1.0 - 5.0)
        rating = raw_input.get("company_rating")
        company_rating = float(np.clip(float(rating if rating is not None else 3.8), 1.0, 5.0))

        # 6. Skills Count (1 - 25)
        skills = raw_input.get("skills_count")
        skills_count = int(np.clip(int(skills if skills is not None else 5), 1, 25))

        # 7. Other features
        work_year = int(raw_input.get("work_year", 2026))
        remote_ratio = int(raw_input.get("remote_ratio", 50))
        company_size = str(raw_input.get("company_size", "L")).strip().upper()
        if company_size not in ["S", "M", "L"]:
            company_size = "L"

        return {
            "work_year": work_year,
            "experience_level": exp_lvl,
            "employment_type": "FT",
            "role_standard": role_standard,
            "primary_city_clean": primary_city,
            "remote_ratio": remote_ratio,
            "company_size": company_size,
            "experience_avg_yrs": experience_avg_yrs,
            "company_rating": company_rating,
            "skills_count": skills_count,
        }

    def predict_india(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """Performs salary prediction using the India Gradient Boosting model pipeline."""
        if self._india_pipeline is None:
            self._load_india_model()
            if self._india_pipeline is None:
                raise FileNotFoundError(f"India model artifact not found at: {config.INDIA_MODEL_PATH}")

        sanitized = self.sanitize_india_input(feature_data)
        df_input = pd.DataFrame([sanitized])

        raw_pred = float(self._india_pipeline.predict(df_input)[0])
        pred_salary_lpa = max(1.5, raw_pred)

        lower_bound = max(1.0, pred_salary_lpa - self._india_mae)
        upper_bound = pred_salary_lpa + self._india_mae

        formatted_salary = f"₹{pred_salary_lpa:.2f} LPA"

        # Equivalent USD for backward compatibility (using standard ~84.0 INR/USD exchange rate)
        # 1 LPA = 100,000 INR
        salary_in_inr = pred_salary_lpa * 100_000.0
        predicted_salary_usd = round(salary_in_inr / 84.0, 2)

        return {
            "country": "IN",
            "predicted_salary": round(pred_salary_lpa, 2),
            "currency": "INR",
            "salary_unit": "LPA",
            "formatted_salary": formatted_salary,
            "model_used": self._india_model_name,
            "predicted_salary_usd": predicted_salary_usd,
            "salary_range": {
                "min": round(lower_bound, 2),
                "max": round(upper_bound, 2),
                "formatted_min": f"₹{lower_bound:.2f} LPA",
                "formatted_max": f"₹{upper_bound:.2f} LPA",
            },
            "confidence_margin_mae": round(self._india_mae, 2),
            "input_features": sanitized,
            "model_version": os.path.basename(config.INDIA_MODEL_PATH),
        }

    # -------------------------------------------------------------------------
    # 5. BATCH PREDICTION
    # -------------------------------------------------------------------------
    def predict_batch(self, profiles: List[Dict[str, Any]], country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Runs batch prediction for a collection of job profiles."""
        results = []
        for profile in profiles:
            p = dict(profile)
            if country and "country" not in p:
                p["country"] = country
            results.append(self.predict(p))
        return results

    # -------------------------------------------------------------------------
    # 6. METADATA DISCOVERY
    # -------------------------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        """Provides sanitized choices for UI dropdown controls for both India and US."""
        experience_levels = {
            "EN": "Entry-level / Junior",
            "MI": "Mid-level / Intermediate",
            "SE": "Senior-level / Specialist",
            "EX": "Executive / Director / VP",
        }

        employment_types = {
            "FT": "Full-Time (FT)",
            "PT": "Part-Time (PT)",
            "CT": "Contract (CT)",
            "FL": "Freelance (FL)",
        }

        company_sizes = {
            "S": "Small (< 50 employees)",
            "M": "Medium (50 - 250 employees)",
            "L": "Large (> 250 employees)",
        }

        remote_ratios = {
            0: "On-site (0% remote)",
            50: "Hybrid (50% remote)",
            100: "Fully Remote (100% remote)",
        }

        top_countries = [
            {"code": "US", "name": "United States"},
            {"code": "IN", "name": "India"},
            {"code": "GB", "name": "United Kingdom"},
            {"code": "CA", "name": "Canada"},
            {"code": "DE", "name": "Germany"},
        ]

        # Extract US Job titles
        us_titles = []
        if self._us_metadata and "categories" in self._us_metadata:
            us_titles = self._us_metadata["categories"].get("job_title", [])
        if not us_titles:
            us_titles = [
                "Data Scientist",
                "Data Engineer",
                "Data Analyst",
                "Machine Learning Engineer",
                "Analytics Engineer",
                "Data Architect",
            ]

        # Extract India Roles and Cities
        india_roles = [
            "Data Scientist",
            "Data Analyst",
            "Data Engineer",
            "Machine Learning Engineer",
            "Business Analyst",
            "Python Developer",
        ]
        india_cities = [
            "Bangalore",
            "Hyderabad",
            "Pune",
            "Mumbai",
            "Delhi / NCR",
            "Chennai",
            "Kolkata",
            "Ahmedabad",
            "Remote",
        ]
        if self._india_metadata and "categories" in self._india_metadata:
            india_roles = self._india_metadata["categories"].get("role_standard", india_roles)
            india_cities = self._india_metadata["categories"].get("primary_city_clean", india_cities)

        supported_countries = [
            {"code": "US", "name": "United States", "currency": "USD", "unit": "YEAR", "symbol": "$"},
            {"code": "IN", "name": "India", "currency": "INR", "unit": "LPA", "symbol": "₹"},
        ]

        return {
            "job_titles": sorted(us_titles),
            "experience_levels": experience_levels,
            "employment_types": employment_types,
            "company_sizes": company_sizes,
            "remote_ratios": remote_ratios,
            "top_countries": top_countries,
            "supported_countries": supported_countries,
            "india_roles": sorted(india_roles),
            "india_cities": sorted(india_cities),
            "india_experience_levels": experience_levels,
            "india_company_sizes": company_sizes,
        }

    # -------------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE (COUNTRY-AWARE)
    # -------------------------------------------------------------------------
    def get_feature_importance(self, top_n: int = 10, country: str = "US") -> Dict[str, Any]:
        """
        Extracts mathematical feature importances directly from the trained country model.
        Returns exact percentage contributions and non-fabricated empirical summaries.
        """
        c = str(country).strip().upper()
        if c in ["IN", "IND", "INDIA"]:
            return self._get_india_feature_importance(top_n=top_n)
        return self._get_us_feature_importance(top_n=top_n)

    def _get_us_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """Calculates exact feature importance from the US Random Forest pipeline."""
        if self._us_pipeline is None:
            self._load_us_model()
            if self._us_pipeline is None:
                raise FileNotFoundError("US model pipeline not loaded.")

        preprocessor = self._us_pipeline.named_steps["preprocessor"]
        regressor = self._us_pipeline.named_steps["regressor"]

        if not hasattr(regressor, "feature_importances_"):
            raise ValueError("US regressor does not expose feature_importances_.")

        raw_feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
        total_importance = float(np.sum(importances)) or 1.0

        country_labels = {"US": "United States", "IN": "India", "GB": "United Kingdom", "CA": "Canada"}
        feature_records = []

        for raw_name, imp in zip(raw_feature_names, importances):
            pct = (float(imp) / total_importance) * 100
            readable_name = raw_name
            category = "Other"

            if raw_name.startswith("cat__employee_residence_"):
                code = raw_name.replace("cat__employee_residence_", "")
                readable_name = f"Employee Residence: {country_labels.get(code, code)}"
                category = "Geography"
            elif raw_name.startswith("cat__company_location_"):
                code = raw_name.replace("cat__company_location_", "")
                readable_name = f"Company Location: {country_labels.get(code, code)}"
                category = "Geography"
            elif raw_name.startswith("cat__experience_level_"):
                code = raw_name.replace("cat__experience_level_", "")
                readable_name = f"Experience Tier: {code}"
                category = "Experience"
            elif raw_name.startswith("cat__job_title_"):
                title = raw_name.replace("cat__job_title_", "")
                readable_name = f"Role: {title}"
                category = "Role"
            elif raw_name.startswith("cat__company_size_"):
                code = raw_name.replace("cat__company_size_", "")
                readable_name = f"Company Scale: {code}"
                category = "Company"
            elif raw_name.startswith("num__work_year"):
                readable_name = "Work Year (Historical Market Cycle)"
                category = "Temporal"
            elif raw_name.startswith("num__remote_ratio"):
                readable_name = "Remote Work Ratio"
                category = "Workplace"
            elif raw_name.startswith("cat__employment_type_"):
                code = raw_name.replace("cat__employment_type_", "")
                readable_name = f"Employment: {code}"
                category = "Contract"

            feature_records.append({
                "feature": readable_name,
                "raw_feature": str(raw_name),
                "importance": round(float(imp), 6),
                "percentage": round(float(pct), 2),
                "category": category,
            })

        sorted_features = sorted(feature_records, key=lambda x: x["importance"], reverse=True)
        top_features = sorted_features[:top_n]

        top_driver = top_features[0]["feature"]
        top_pct = top_features[0]["percentage"]
        second_driver = top_features[1]["feature"]
        second_pct = top_features[1]["percentage"]

        explanation = (
            f"According to the trained US Random Forest regressor, {top_driver} is the single most decisive factor, "
            f"accounting for {top_pct}% of the model's decision splits, followed by {second_driver} ({second_pct}%)."
        )

        return {
            "model_name": self._us_model_name,
            "country": "US",
            "currency": "USD",
            "salary_unit": "YEAR",
            "top_features": top_features,
            "summary_explanation": explanation,
            "total_features_evaluated": len(raw_feature_names),
        }

    def _get_india_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """Calculates exact feature importance from the India Gradient Boosting pipeline."""
        if self._india_pipeline is None:
            self._load_india_model()
            if self._india_pipeline is None:
                raise FileNotFoundError("India model pipeline not loaded.")

        preprocessor = self._india_pipeline.named_steps["preprocessor"]
        regressor = self._india_pipeline.named_steps["regressor"]

        if not hasattr(regressor, "feature_importances_"):
            raise ValueError("India regressor does not expose feature_importances_.")

        raw_feature_names = preprocessor.get_feature_names_out()
        importances = regressor.feature_importances_
        total_importance = float(np.sum(importances)) or 1.0

        feature_records = []
        for raw_name, imp in zip(raw_feature_names, importances):
            pct = (float(imp) / total_importance) * 100
            readable_name = raw_name
            category = "Other"

            if raw_name.startswith("num__experience_avg_yrs"):
                readable_name = "Years of Professional Experience"
                category = "Experience"
            elif raw_name.startswith("num__company_rating"):
                readable_name = "Company Reputation Rating"
                category = "Company"
            elif raw_name.startswith("num__skills_count"):
                readable_name = "Technical Skills Count"
                category = "Competency"
            elif raw_name.startswith("num__remote_ratio"):
                readable_name = "Workplace Flexibility (Remote Ratio)"
                category = "Workplace"
            elif raw_name.startswith("num__work_year"):
                readable_name = "Market Cycle Year"
                category = "Temporal"
            elif raw_name.startswith("cat__role_standard_"):
                role = raw_name.replace("cat__role_standard_", "")
                readable_name = f"Job Role: {role}"
                category = "Role"
            elif raw_name.startswith("cat__primary_city_clean_"):
                city = raw_name.replace("cat__primary_city_clean_", "")
                readable_name = f"Tech Hub City: {city}"
                category = "Geography"
            elif raw_name.startswith("cat__experience_level_"):
                tier = raw_name.replace("cat__experience_level_", "")
                readable_name = f"Seniority Tier: {tier}"
                category = "Experience"
            elif raw_name.startswith("cat__company_size_"):
                size = raw_name.replace("cat__company_size_", "")
                readable_name = f"Company Scale: {size}"
                category = "Company"
            elif raw_name.startswith("cat__employment_type_"):
                emp = raw_name.replace("cat__employment_type_", "")
                readable_name = f"Employment Type: {emp}"
                category = "Contract"

            feature_records.append({
                "feature": readable_name,
                "raw_feature": str(raw_name),
                "importance": round(float(imp), 6),
                "percentage": round(float(pct), 2),
                "category": category,
            })

        sorted_features = sorted(feature_records, key=lambda x: x["importance"], reverse=True)
        top_features = sorted_features[:top_n]

        top_driver = top_features[0]["feature"]
        top_pct = top_features[0]["percentage"]
        second_driver = top_features[1]["feature"]
        second_pct = top_features[1]["percentage"]

        explanation = (
            f"According to the trained India Gradient Boosting regressor, {top_driver} is the primary driver "
            f"accounting for {top_pct}% of salary variance in Indian tech compensation, followed by {second_driver} ({second_pct}%)."
        )

        return {
            "model_name": self._india_model_name,
            "country": "IN",
            "currency": "INR",
            "salary_unit": "LPA",
            "top_features": top_features,
            "summary_explanation": explanation,
            "total_features_evaluated": len(raw_feature_names),
        }

    # -------------------------------------------------------------------------
    # 8. HEALTH CHECK
    # -------------------------------------------------------------------------
    def health_check(self) -> Dict[str, Any]:
        """Reports availability and health of both country prediction pipelines."""
        us_exists = os.path.exists(config.US_MODEL_PATH)
        india_exists = os.path.exists(config.INDIA_MODEL_PATH)

        status_str = "healthy" if (us_exists and india_exists) else ("degraded" if (us_exists or india_exists) else "unhealthy")

        return {
            "status": status_str,
            "models": {
                "US": {
                    "loaded": self._us_pipeline is not None,
                    "exists": us_exists,
                    "path": config.US_MODEL_PATH,
                    "model_name": self._us_model_name,
                    "currency": "USD",
                    "unit": "YEAR",
                },
                "IN": {
                    "loaded": self._india_pipeline is not None,
                    "exists": india_exists,
                    "path": config.INDIA_MODEL_PATH,
                    "model_name": self._india_model_name,
                    "currency": "INR",
                    "unit": "LPA",
                },
            },
        }


# Global singleton instance
prediction_service = PredictionService()
