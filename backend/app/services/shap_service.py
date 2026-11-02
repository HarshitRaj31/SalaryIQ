
"""
SalaryIQ Explainable AI Service
SHAP-based local and global salary prediction explanations.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
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


# ---------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------

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

FEATURE_METADATA = {
    "experience_level": {
        "title": "Experience Seniority",
        "category": "Seniority",
    },
    "employee_residence": {
        "title": "Employee Residence",
        "category": "Geography",
    },
    "company_location": {
        "title": "Company Location",
        "category": "Geography",
    },
    "job_title": {
        "title": "Job Title / Role",
        "category": "Role",
    },
    "work_year": {
        "title": "Work Year & Market Cycle",
        "category": "Temporal",
    },
    "remote_ratio": {
        "title": "Workplace Setting",
        "category": "Workplace",
    },
    "company_size": {
        "title": "Company Size / Scale",
        "category": "Company",
    },
    "employment_type": {
        "title": "Employment Contract Type",
        "category": "Contract",
    },
}

FEATURE_KEYS = list(FEATURE_METADATA.keys())


def format_money(value: float) -> str:
    """Format a value as USD."""
    return f"${abs(round(value)):,.0f}"


# ---------------------------------------------------------
# SHAP service
# ---------------------------------------------------------

class ShapExplainabilityService:
    """
    SHAP explainability service for the trained salary model.

    Supports:
    - Local TreeSHAP explanations
    - SHAP additivity validation
    - Grouped feature contributions
    - Waterfall chart data
    - Global mean absolute SHAP importance
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._instance._initialized = False
            cls._instance._pipeline = None
            cls._instance._preprocessor = None
            cls._instance._regressor = None
            cls._instance._explainer = None
            cls._instance._base_value = None
            cls._instance._feature_names = []
            cls._instance._global_cache = None

        return cls._instance

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    def _initialize_explainer(self):
        """Load the trained pipeline and initialize TreeExplainer."""

        if self._initialized:
            return

        if shap is None:
            raise RuntimeError(
                "SHAP is not installed in the backend Python environment."
            )

        model_path = config.MODEL_PATH

        if not os.path.exists(model_path):
            fallback_path = os.path.join(
                config.ML_DIR,
                "models",
                "random_forest_regressor.pkl",
            )

            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Trained model not found: {model_path}"
                )

        self._pipeline = joblib.load(model_path)

        if not hasattr(self._pipeline, "named_steps"):
            raise ValueError(
                "The loaded model must be a scikit-learn Pipeline."
            )

        self._preprocessor = self._pipeline.named_steps["preprocessor"]
        self._regressor = self._pipeline.named_steps["regressor"]

        self._feature_names = list(
            self._preprocessor.get_feature_names_out()
        )

        try:
            self._explainer = shap.TreeExplainer(self._regressor)

            self._base_value = self._extract_base_value(
                self._explainer.expected_value
            )

        except Exception as exc:
            self._explainer = None
            self._base_value = None

            raise RuntimeError(
                f"Unable to initialize TreeSHAP: {exc}"
            ) from exc

        self._initialized = True

        logger.info(
            "SHAP initialized successfully. Model=%s, baseline=%.2f",
            type(self._regressor).__name__,
            self._base_value,
        )

    def _ensure_initialized(self):
        if not self._initialized:
            self._initialize_explainer()

        if self._explainer is None:
            raise RuntimeError(
                "SHAP explainer is unavailable."
            )

    # -----------------------------------------------------
    # SHAP output normalization
    # -----------------------------------------------------

    @staticmethod
    def _extract_base_value(expected_value) -> float:
        """Convert SHAP expected value into a scalar."""

        values = np.asarray(expected_value).reshape(-1)

        if len(values) != 1:
            raise ValueError(
                "This service currently supports single-output regression."
            )

        return float(values[0])

    def _get_shap_values(self, transformed) -> np.ndarray:
        """Return normalized SHAP values for one or more rows."""

        try:
            explanation = self._explainer(transformed)
            values = explanation.values

        except Exception:
            values = self._explainer.shap_values(transformed)

        values = np.asarray(values)

        # Shape: (samples, features, outputs)
        if values.ndim == 3:
            if values.shape[2] != 1:
                raise ValueError(
                    "Multi-output SHAP explanations are not supported."
                )
            values = values[:, :, 0]

        # Shape: (samples, features)
        if values.ndim == 1:
            values = values.reshape(1, -1)

        if values.ndim != 2:
            raise ValueError(
                f"Unexpected SHAP output shape: {values.shape}"
            )

        return values

    # -----------------------------------------------------
    # Feature grouping
    # -----------------------------------------------------

    @staticmethod
    def _group_shap_values(
        feature_names: List[str],
        shap_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Aggregate one-hot encoded SHAP values into their
        original input feature groups.
        """

        grouped = {
            key: 0.0
            for key in FEATURE_KEYS
        }

        for feature_name, shap_value in zip(
            feature_names,
            shap_values,
        ):
            matched = False

            for key in FEATURE_KEYS:
                if (
                    feature_name == key
                    or feature_name.endswith(f"__{key}")
                    or feature_name.startswith(f"cat__{key}_")
                    or feature_name.startswith(f"num__{key}")
                ):
                    grouped[key] += float(shap_value)
                    matched = True
                    break

            if not matched:
                logger.warning(
                    "Could not map transformed feature: %s",
                    feature_name,
                )

        return grouped

    @staticmethod
    def _display_value(key: str, raw_value: Any) -> str:

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
            labels = {
                100: "100% Fully Remote",
                50: "50% Hybrid",
                0: "On-site (0%)",
            }
            return labels.get(raw_value, f"{raw_value}% Remote")

        return str(raw_value)

    # -----------------------------------------------------
    # Local explanation
    # -----------------------------------------------------

    def explain_prediction(
        self,
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:

        self._ensure_initialized()

        clean_profile = {
            "work_year": int(profile.get("work_year", 2023)),
            "experience_level": str(
                profile.get("experience_level", "SE")
            ),
            "employment_type": str(
                profile.get("employment_type", "FT")
            ),
            "job_title": str(
                profile.get("job_title", "Data Scientist")
            ),
            "employee_residence": str(
                profile.get("employee_residence", "US")
            ),
            "remote_ratio": int(
                profile.get("remote_ratio", 100)
            ),
            "company_location": str(
                profile.get("company_location", "US")
            ),
            "company_size": str(
                profile.get("company_size", "M")
            ),
        }

        df = pd.DataFrame([clean_profile])

        transformed = self._preprocessor.transform(df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        prediction = float(
            self._regressor.predict(transformed)[0]
        )

        shap_matrix = self._get_shap_values(transformed)
        raw_shap = shap_matrix[0]

        if len(raw_shap) != len(self._feature_names):
            raise ValueError(
                "SHAP feature count does not match preprocessor output."
            )

        base_value = float(self._base_value)
        actual_shap_sum = float(np.sum(raw_shap))

        reconstructed_prediction = (
            base_value + actual_shap_sum
        )

        # Validate the SHAP additivity property.
        if not np.isclose(
            reconstructed_prediction,
            prediction,
            rtol=1e-5,
            atol=1.0,
        ):
            raise ValueError(
                "SHAP additivity validation failed. "
                f"Prediction={prediction:.2f}, "
                f"Reconstructed={reconstructed_prediction:.2f}"
            )

        grouped_shap = self._group_shap_values(
            self._feature_names,
            raw_shap,
        )

        grouped_sum = sum(grouped_shap.values())

        if not np.isclose(
            grouped_sum,
            actual_shap_sum,
            rtol=1e-5,
            atol=1.0,
        ):
            raise ValueError(
                "Grouped SHAP contributions do not match "
                "the original SHAP contribution sum."
            )

        # Build feature contribution records.
        feature_contributions = []

        for key, shap_value in grouped_shap.items():

            metadata = FEATURE_METADATA[key]
            raw_value = clean_profile[key]

            direction = (
                "positive"
                if shap_value >= 0
                else "negative"
            )

            percentage = (
                shap_value / abs(base_value) * 100
                if base_value != 0
                else 0.0
            )

            sign = "+" if shap_value >= 0 else "-"

            feature_contributions.append({
                "field": key,
                "feature_title": metadata["title"],
                "category": metadata["category"],
                "value": self._display_value(key, raw_value),
                "raw_value": raw_value,
                "shap_value": round(shap_value, 2),
                "formatted_shap": (
                    f"{sign}{format_money(shap_value)}"
                ),
                "direction": direction,
                "percentage_impact": round(percentage, 2),
                "abs_impact": abs(shap_value),
            })

        feature_contributions.sort(
            key=lambda item: item["abs_impact"],
            reverse=True,
        )

        top_positive = [
            item for item in feature_contributions
            if item["direction"] == "positive"
        ]

        top_negative = [
            item for item in feature_contributions
            if item["direction"] == "negative"
        ]

        # Build waterfall data.
        waterfall_steps = []

        running_total = base_value

        waterfall_steps.append({
            "name": "Base Market Average",
            "feature": "Expected Base Value",
            "value": round(base_value),
            "delta": round(base_value),
            "step_bottom": 0,
            "step_height": round(base_value),
            "formatted": format_money(base_value),
            "direction": "base",
            "type": "base",
            "running_total": round(running_total),
        })

        for item in feature_contributions:

            delta = item["shap_value"]

            previous_total = running_total
            running_total += delta

            waterfall_steps.append({
                "name": item["feature_title"],
                "feature": (
                    f"{item['feature_title']}: {item['value']}"
                ),
                "value": round(running_total),
                "delta": round(delta),
                "step_bottom": round(
                    min(previous_total, running_total)
                ),
                "step_height": round(abs(delta)),
                "formatted": item["formatted_shap"],
                "direction": item["direction"],
                "type": item["direction"],
                "running_total": round(running_total),
            })

        waterfall_steps.append({
            "name": "Final Prediction",
            "feature": (
                f"Predicted Salary ({clean_profile['job_title']})"
            ),
            "value": round(prediction),
            "delta": round(prediction),
            "step_bottom": 0,
            "step_height": round(prediction),
            "formatted": format_money(prediction),
            "direction": "total",
            "type": "total",
            "running_total": round(prediction),
        })

        # Build natural-language explanation.
        narrative_parts = [
            (
                f"The predicted annual salary of "
                f"{format_money(prediction)} is "
                f"{'+' if actual_shap_sum >= 0 else '-'}"
                f"{format_money(actual_shap_sum)} compared "
                f"to the model's SHAP baseline of "
                f"{format_money(base_value)}."
            )
        ]

        if top_positive:
            item = top_positive[0]
            narrative_parts.append(
                f"The strongest positive factor is "
                f"{item['feature_title']} "
                f"({item['value']}), contributing "
                f"{item['formatted_shap']}."
            )

        if top_negative:
            item = top_negative[0]
            narrative_parts.append(
                f"The strongest negative factor is "
                f"{item['feature_title']} "
                f"({item['value']}), contributing "
                f"{item['formatted_shap']}."
            )

        narrative = " ".join(narrative_parts)

        model_name = type(self._regressor).__name__

        return {
            "status": "success",
            "model_name": f"{model_name} (TreeSHAP)",
            "base_value": round(base_value, 2),
            "formatted_base_value": format_money(base_value),
            "predicted_salary": round(prediction, 2),
            "formatted_predicted_salary": format_money(prediction),
            "total_shap_delta": round(actual_shap_sum, 2),
            "sum_shap_contributions": round(actual_shap_sum, 2),
            "formatted_total_delta": (
                f"{'+' if actual_shap_sum >= 0 else '-'}"
                f"{format_money(actual_shap_sum)}"
            ),
            "feature_contributions": feature_contributions,
            "top_positive_factors": top_positive[:3],
            "top_negative_factors": top_negative[:3],
            "waterfall_chart_data": waterfall_steps,
            "narrative_explanation": narrative,
            "narrative": narrative,
            "input_profile": clean_profile,
            "additivity_check": {
                "passed": True,
                "prediction": round(prediction, 2),
                "reconstructed_prediction": round(
                    reconstructed_prediction, 2
                ),
                "difference": round(
                    abs(prediction - reconstructed_prediction), 6
                ),
            },
        }

    # -----------------------------------------------------
    # Global SHAP importance
    # -----------------------------------------------------

    def get_global_shap_importance(
        self,
        top_n: int = 10,
        sample_size: int = 200,
    ) -> Dict[str, Any]:
        """
        Calculate global feature importance using mean absolute SHAP.

        Uses a deterministic sample from cleaned_salary.csv.
        """

        self._ensure_initialized()

        if self._global_cache is not None:
            return {
                **self._global_cache,
                "top_global_features": (
                    self._global_cache["global_importance"][:top_n]
                ),
                "global_importance": (
                    self._global_cache["global_importance"][:top_n]
                ),
            }

        dataset_path = os.path.join(
            config.ML_DIR,
            "data",
            "processed",
            "cleaned_salary.csv",
        )

        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(
                config.ML_DIR,
                "data",
                "raw",
                "ds_salaries.csv",
            )

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                "Salary dataset not found. Expected cleaned_salary.csv "
                "or ds_salaries.csv."
            )

        df = pd.read_csv(dataset_path)

        missing_columns = [
            key for key in FEATURE_KEYS
            if key not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required features: {missing_columns}"
            )

        # Use only the model input columns.
        X = df[FEATURE_KEYS].copy()

        X = X.sample(
            n=min(sample_size, len(X)),
            random_state=42,
        )

        transformed = self._preprocessor.transform(X)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed = np.asarray(transformed)

        shap_matrix = self._get_shap_values(transformed)

        if shap_matrix.shape[1] != len(self._feature_names):
            raise ValueError(
                "Global SHAP feature count does not match model features."
            )

        # Mean absolute SHAP for each encoded feature.
        encoded_importance = np.mean(
            np.abs(shap_matrix),
            axis=0,
        )

        # Aggregate encoded columns into original input features.
        grouped_importance = {
            key: 0.0
            for key in FEATURE_KEYS
        }

        for feature_name, importance in zip(
            self._feature_names,
            encoded_importance,
        ):
            for key in FEATURE_KEYS:
                if (
                    feature_name == key
                    or feature_name.endswith(f"__{key}")
                    or feature_name.startswith(f"cat__{key}_")
                    or feature_name.startswith(f"num__{key}")
                ):
                    grouped_importance[key] += float(importance)
                    break

        total_importance = sum(grouped_importance.values())

        if total_importance <= 0:
            raise ValueError(
                "Global SHAP importance calculation returned zero."
            )

        records = []

        for key, importance in grouped_importance.items():

            metadata = FEATURE_METADATA[key]

            percentage = (
                importance / total_importance
            ) * 100

            records.append({
                "feature": metadata["title"],
                "field": key,
                "category": metadata["category"],
                "importance": round(importance, 4),
                "percentage": round(percentage, 2),
            })

        records.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        model_name = type(self._regressor).__name__

        result = {
            "model": f"{model_name} (TreeSHAP)",
            "importance_method": "Mean Absolute SHAP",
            "sample_size": len(X),
            "base_value": round(float(self._base_value), 2),
            "global_importance": records,
        }

        self._global_cache = result

        return {
            **result,
            "top_global_features": records[:top_n],
            "global_importance": records[:top_n],
        }


# ---------------------------------------------------------
# Global service instance
# ---------------------------------------------------------

shap_service = ShapExplainabilityService()
