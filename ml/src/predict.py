"""
Production Salary Inference Module for Job Salary Prediction.

Provides the `SalaryPredictor` service and standalone functions for:
- Single-instance salary prediction with confidence bounds
- Batch salary predictions
- Input validation, normalization, and intelligent defaults
- Metadata discovery (valid job titles, experience tiers, countries)
- CLI interface for standalone command-line predictions and demo benchmarking
"""

import argparse
import json
import os
import sys
import warnings

# Avoid matplotlib cache issues if imported elsewhere
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml/ directory
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")


class SalaryPredictor:
    """
    Production-ready salary predictor that wraps scikit-learn pipeline inference.
    Handles input validation, defaults, confidence intervals, and metadata lookups.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.metadata = self._load_metadata()
        self.mae_error = self._load_model_error()

    def _load_model(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at '{path}'. Please run 'train.py' first."
            )
        return joblib.load(path)

    def _load_metadata(self) -> dict:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "categorical_features": [
                "experience_level",
                "employment_type",
                "job_title",
                "employee_residence",
                "company_location",
                "company_size",
            ],
            "numerical_features": ["work_year", "remote_ratio"],
        }

    def _load_model_error(self) -> float:
        """Loads MAE for confidence range estimation; defaults to $39,000 if not found."""
        if os.path.exists(COMPARISON_PATH):
            try:
                with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
                    comp = json.load(f)
                best_name = comp.get("best_model", "Gradient Boosting Regressor")
                return float(comp.get("metrics", {}).get(best_name, {}).get("MAE", 39067.0))
            except Exception:
                pass
        return 39067.0

    def get_valid_options(self) -> dict:
        """Returns the dictionary of recognized categories for form dropdowns."""
        return self.metadata.get("categories", {})

    def sanitize_input(self, raw_input: dict) -> dict:
        """
        Validates, normalizes, and applies sensible defaults to input data.
        """
        sanitized = dict(raw_input)

        # Standard defaults for optional attributes
        company_loc = str(sanitized.get("company_location", "US")).strip().upper()
        sanitized["company_location"] = company_loc
        sanitized["employee_residence"] = str(
            sanitized.get("employee_residence", company_loc)
        ).strip().upper()

        sanitized["work_year"] = int(sanitized.get("work_year", 2023))
        sanitized["employment_type"] = str(
            sanitized.get("employment_type", "FT")
        ).strip().upper()

        sanitized["experience_level"] = str(
            sanitized.get("experience_level", "SE")
        ).strip().upper()

        sanitized["company_size"] = str(
            sanitized.get("company_size", "M")
        ).strip().upper()

        sanitized["remote_ratio"] = int(sanitized.get("remote_ratio", 100))
        sanitized["job_title"] = str(
            sanitized.get("job_title", "Data Scientist")
        ).strip()

        return sanitized

    def predict(self, input_data: dict) -> dict:
        """
        Predicts salary for a single job profile dictionary.

        Parameters:
            input_data: dictionary containing features (e.g. job_title, experience_level, etc.)

        Returns:
            Dictionary containing predicted salary, formatted strings, and range estimates.
        """
        sanitized = self.sanitize_input(input_data)
        df_input = pd.DataFrame([sanitized])

        raw_pred = float(self.model.predict(df_input)[0])
        pred_salary = max(10000.0, raw_pred)  # Sensible global salary floor

        lower_bound = max(5000.0, pred_salary - self.mae_error)
        upper_bound = pred_salary + self.mae_error

        return {
            "predicted_salary_usd": round(pred_salary, 2),
            "formatted_salary": f"${round(pred_salary):,}",
            "salary_range": {
                "min": round(lower_bound, 2),
                "max": round(upper_bound, 2),
                "formatted_min": f"${round(lower_bound):,}",
                "formatted_max": f"${round(upper_bound):,}",
            },
            "confidence_margin_mae": round(self.mae_error, 2),
            "input_features": sanitized,
            "model_version": os.path.basename(self.model_path),
        }

    def predict_batch(self, input_list: list) -> list:
        """
        Performs vectorized batch prediction for a list of input dictionaries.
        """
        if not input_list:
            return []

        sanitized_list = [self.sanitize_input(item) for item in input_list]
        df_batch = pd.DataFrame(sanitized_list)

        raw_preds = self.model.predict(df_batch)

        results = []
        for sanitized, pred in zip(sanitized_list, raw_preds):
            pred_salary = max(10000.0, float(pred))
            lower_bound = max(5000.0, pred_salary - self.mae_error)
            upper_bound = pred_salary + self.mae_error
            results.append({
                "predicted_salary_usd": round(pred_salary, 2),
                "formatted_salary": f"${round(pred_salary):,}",
                "salary_range": {
                    "min": round(lower_bound, 2),
                    "max": round(upper_bound, 2),
                    "formatted_min": f"${round(lower_bound):,}",
                    "formatted_max": f"${round(upper_bound):,}",
                },
                "input_features": sanitized,
            })
        return results


# Global singleton instance for easy import across backend services
_global_predictor = None


def get_predictor() -> SalaryPredictor:
    """Returns a singleton SalaryPredictor instance."""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = SalaryPredictor()
    return _global_predictor


def predict_salary(input_data: dict) -> dict:
    """Convenience helper function for single-instance salary prediction."""
    return get_predictor().predict(input_data)


def predict_batch(input_list: list) -> list:
    """Convenience helper function for batch salary prediction."""
    return get_predictor().predict_batch(input_list)


def run_demo():
    """Runs a demonstration across diverse industry personas."""
    predictor = get_predictor()

    demo_personas = [
        {
            "persona": "Senior Data Scientist (US Remote)",
            "features": {
                "work_year": 2023,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Data Scientist",
                "employee_residence": "US",
                "remote_ratio": 100,
                "company_location": "US",
                "company_size": "M",
            },
        },
        {
            "persona": "Entry-level Data Analyst (US Hybrid)",
            "features": {
                "work_year": 2023,
                "experience_level": "EN",
                "employment_type": "FT",
                "job_title": "Data Analyst",
                "employee_residence": "US",
                "remote_ratio": 50,
                "company_location": "US",
                "company_size": "M",
            },
        },
        {
            "persona": "Executive / Director of Data Science (US)",
            "features": {
                "work_year": 2023,
                "experience_level": "EX",
                "employment_type": "FT",
                "job_title": "Director of Data Science",
                "employee_residence": "US",
                "remote_ratio": 100,
                "company_location": "US",
                "company_size": "L",
            },
        },
        {
            "persona": "Machine Learning Engineer (UK On-site)",
            "features": {
                "work_year": 2023,
                "experience_level": "SE",
                "employment_type": "FT",
                "job_title": "Machine Learning Engineer",
                "employee_residence": "GB",
                "remote_ratio": 0,
                "company_location": "GB",
                "company_size": "L",
            },
        },
        {
            "persona": "Mid-level Data Engineer (India Remote)",
            "features": {
                "work_year": 2023,
                "experience_level": "MI",
                "employment_type": "FT",
                "job_title": "Data Engineer",
                "employee_residence": "IN",
                "remote_ratio": 100,
                "company_location": "IN",
                "company_size": "M",
            },
        },
    ]

    print("\n" + "=" * 80)
    print("                     JOB SALARY PREDICTION BENCHMARK DEMO")
    print("=" * 80)
    print(f"{'Persona / Archetype':<40} | {'Predicted Salary':<16} | {'Expected Range':<22}")
    print("-" * 80)

    for item in demo_personas:
        res = predictor.predict(item["features"])
        range_str = f"{res['salary_range']['formatted_min']} - {res['salary_range']['formatted_max']}"
        print(f"{item['persona']:<40} | {res['formatted_salary']:<16} | {range_str:<22}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Job Salary Prediction Inference CLI")
    parser.add_argument("--job-title", type=str, default="Data Scientist", help="Job Title")
    parser.add_argument(
        "--experience-level",
        type=str,
        default="SE",
        choices=["EN", "MI", "SE", "EX"],
        help="Experience level tier",
    )
    parser.add_argument(
        "--employment-type",
        type=str,
        default="FT",
        choices=["FT", "PT", "CT", "FL"],
        help="Employment type",
    )
    parser.add_argument(
        "--company-location",
        type=str,
        default="US",
        help="2-letter ISO country code of employer (e.g. US, GB, CA, IN, DE)",
    )
    parser.add_argument(
        "--company-size",
        type=str,
        default="M",
        choices=["S", "M", "L"],
        help="Company size (S: <50, M: 50-250, L: >250)",
    )
    parser.add_argument(
        "--remote-ratio",
        type=int,
        default=100,
        choices=[0, 50, 100],
        help="Remote work ratio (0: On-site, 50: Hybrid, 100: Remote)",
    )
    parser.add_argument("--work-year", type=int, default=2023, help="Work year (e.g. 2023)")
    parser.add_argument("--json", type=str, default=None, help="Input profile as JSON string")
    parser.add_argument(
        "--demo", action="store_true", help="Run benchmark demonstration across multiple roles"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.json:
        try:
            profile = json.loads(args.json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON input: {e}")
            sys.exit(1)
    else:
        profile = {
            "job_title": args.job_title,
            "experience_level": args.experience_level,
            "employment_type": args.employment_type,
            "company_location": args.company_location,
            "company_size": args.company_size,
            "remote_ratio": args.remote_ratio,
            "work_year": args.work_year,
        }

    predictor = get_predictor()
    result = predictor.predict(profile)

    print("\n" + "=" * 60)
    print("               SALARY PREDICTION RESULT")
    print("=" * 60)
    print(f"  Target Job Title:       {result['input_features']['job_title']}")
    print(f"  Experience Level:       {result['input_features']['experience_level']}")
    print(f"  Company Location:       {result['input_features']['company_location']}")
    print(f"  Remote Status:          {result['input_features']['remote_ratio']}%")
    print(f"  Company Size:           {result['input_features']['company_size']}")
    print("-" * 60)
    print(f"  ESTIMATED SALARY:       {result['formatted_salary']} USD / year")
    print(f"  Expected Range (±MAE):  {result['salary_range']['formatted_min']} - {result['salary_range']['formatted_max']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

