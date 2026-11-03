"""
Comprehensive Test Suite for Dual-Country SalaryIQ Backend Integration.
Tests all 10 requirements:
1. India model loads successfully.
2. US model loads successfully.
3. India prediction returns INR/LPA with formatted string.
4. US prediction returns USD/year with formatted string.
5. Invalid country returns validation error (HTTP 422).
6. Missing required fields return validation errors.
7. Invalid numerical values are rejected (out-of-bounds work_year, rating, etc.).
8. Prediction history saves both country types and allows filtering.
9. Existing US API functionality remains 100% backward compatible.
10. Feature importance works independently for both models.
11. SHAP explanations work for both India and US models.
12. Health check reports both models active.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import HTTPException
from pydantic import ValidationError

from app.core import config
from app.history_service import history_service
from app.routes.prediction import (
    explain_prediction_endpoint,
    get_feature_importance_endpoint,
    get_metadata_endpoint,
    get_prediction_history,
    health_check_endpoint,
    predict_salary_endpoint,
)
from app.schemas.salary import SalaryPredictionRequest
from app.services.prediction_service import prediction_service
from app.services.shap_service import shap_service


def run_all_tests():
    passed = 0
    failed = 0

    print("=" * 80)
    print("STARTING DUAL-COUNTRY SALARYIQ BACKEND TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: India Model Loading
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Testing India Model Loading...")
    try:
        assert os.path.exists(config.INDIA_MODEL_PATH), f"India model missing at {config.INDIA_MODEL_PATH}"
        assert prediction_service._india_pipeline is not None, "India pipeline not loaded"
        assert prediction_service._india_model_name == "Gradient Boosting Regressor"
        print(f"  PASS: India model loaded successfully ({prediction_service._india_model_name}).")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 2: US Model Loading
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Testing US Model Loading...")
    try:
        assert os.path.exists(config.US_MODEL_PATH), f"US model missing at {config.US_MODEL_PATH}"
        assert prediction_service._us_pipeline is not None, "US pipeline not loaded"
        assert prediction_service._us_model_name == "Random Forest Regressor"
        print(f"  PASS: US model loaded successfully ({prediction_service._us_model_name}).")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 3: India Prediction Returns INR / LPA
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Testing India Prediction (INR / LPA)...")
    try:
        india_req = SalaryPredictionRequest(
            country="IN",
            role_standard="Data Scientist",
            experience_level="SE",
            primary_city_clean="Bangalore",
            company_size="L",
            remote_ratio=50,
            experience_avg_yrs=6.0,
            company_rating=4.2,
            skills_count=7,
            work_year=2026,
        )
        res = predict_salary_endpoint(india_req)
        assert res["country"] == "IN", f"Expected country IN, got {res['country']}"
        assert res["currency"] == "INR", f"Expected currency INR, got {res['currency']}"
        assert res["salary_unit"] == "LPA", f"Expected unit LPA, got {res['salary_unit']}"
        assert "₹" in res["formatted_salary"] and "LPA" in res["formatted_salary"]
        assert 1.5 <= res["predicted_salary"] <= 100.0, f"Unrealistic salary: {res['predicted_salary']} LPA"
        assert res["model_used"] == "Gradient Boosting Regressor"
        assert "salary_range" in res and res["salary_range"]["min"] < res["salary_range"]["max"]
        print(f"  PASS: India Prediction = {res['formatted_salary']} | Range: {res['salary_range']['formatted_min']} - {res['salary_range']['formatted_max']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 4: US Prediction Returns USD / Year
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Testing US Prediction (USD / Year)...")
    try:
        us_req = SalaryPredictionRequest(
            country="US",
            job_title="Data Scientist",
            experience_level="SE",
            employment_type="FT",
            company_location="US",
            employee_residence="US",
            company_size="M",
            remote_ratio=100,
            work_year=2023,
        )
        res = predict_salary_endpoint(us_req)
        assert res["country"] == "US", f"Expected country US, got {res['country']}"
        assert res["currency"] == "USD", f"Expected currency USD, got {res['currency']}"
        assert res["salary_unit"] == "YEAR", f"Expected unit YEAR, got {res['salary_unit']}"
        assert "$" in res["formatted_salary"] and "/ year" in res["formatted_salary"]
        assert 20000.0 <= res["predicted_salary"] <= 500000.0, f"Unrealistic salary: {res['predicted_salary']}"
        assert res["model_used"] == "Random Forest Regressor"
        assert "salary_range" in res and res["salary_range"]["min"] < res["salary_range"]["max"]
        print(f"  PASS: US Prediction = {res['formatted_salary']} | Range: {res['salary_range']['formatted_min']} - {res['salary_range']['formatted_max']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 5: Invalid Country Code Returns HTTP 422
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing Invalid Country Code Rejection (HTTP 422)...")
    try:
        invalid_req = SalaryPredictionRequest(
            country="INVALID_XYZ",
            job_title="Data Scientist",
            experience_level="SE",
        )
        try:
            predict_salary_endpoint(invalid_req)
            print("  FAIL: Expected HTTP 422 exception but prediction succeeded.")
            failed += 1
        except HTTPException as he:
            assert he.status_code == 422, f"Expected status 422, got {he.status_code}"
            print(f"  PASS: Rejected invalid country code with HTTP 422: '{he.detail}'.")
            passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 6: Missing Required Field Handling / Defaults
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Testing Schema Validation on Missing / Incomplete Fields...")
    try:
        # Omitting experience_level in raw schema initialization
        try:
            SalaryPredictionRequest.model_validate({"job_title": "Data Scientist", "experience_level": None})
            # experience_level is required str
            print("  FAIL: Expected validation error on null experience_level.")
            failed += 1
        except ValidationError:
            print("  PASS: Pydantic correctly raised ValidationError on null experience_level.")
            passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 7: Invalid Numerical Values Rejected (HTTP 422)
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Testing Invalid Numerical Values (out-of-bounds work_year)...")
    try:
        bad_num_req = SalaryPredictionRequest(
            country="US",
            job_title="Data Scientist",
            experience_level="SE",
            work_year=1850,  # Invalid historical year
        )
        try:
            predict_salary_endpoint(bad_num_req)
            print("  FAIL: Did not reject out-of-bounds work_year 1850.")
            failed += 1
        except HTTPException as he:
            assert he.status_code == 422
            print(f"  PASS: Rejected invalid work_year with HTTP 422: '{he.detail}'.")
            passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 8: Prediction History Dual-Country Storage & Filtering
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Testing Prediction History Storage and Country Filtering...")
    try:
        # Fetch history filtered by IN and US
        in_history = get_prediction_history(country="IN")
        us_history = get_prediction_history(country="US")
        all_history = get_prediction_history()

        assert in_history["total"] >= 1, "Expected at least 1 India record in history"
        assert us_history["total"] >= 1, "Expected at least 1 US record in history"
        assert all_history["total"] >= in_history["total"] + us_history["total"] - 1

        first_in = in_history["history"][0]
        assert first_in["country"] == "IN"
        assert first_in["currency"] == "INR"
        assert first_in["salary_unit"] == "LPA"
        assert "₹" in first_in["formatted_salary"]

        first_us = us_history["history"][0]
        assert first_us["country"] == "US"
        assert first_us["currency"] == "USD"
        assert first_us["salary_unit"] == "YEAR"
        assert "$" in first_us["formatted_salary"]

        print(f"  PASS: Dual-country history verified! Stored records: Total={all_history['total']}, US={us_history['total']}, IN={in_history['total']}.")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 9: Existing Legacy US Prediction Compatibility
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Testing Backward Compatibility with Legacy US Request (no country field)...")
    try:
        # Legacy frontend request payload that did not contain country
        legacy_req = SalaryPredictionRequest(
            job_title="Machine Learning Engineer",
            experience_level="EX",
            employment_type="FT",
            company_location="US",
            company_size="L",
            remote_ratio=100,
            work_year=2023,
        )
        res = predict_salary_endpoint(legacy_req)
        assert res["country"] == "US"
        assert res["currency"] == "USD"
        assert "predicted_salary_usd" in res
        assert res["predicted_salary_usd"] > 100000.0
        assert "model_version" in res
        print(f"  PASS: Legacy US request automatically routed and predicted: {res['formatted_salary']} ({res['predicted_salary_usd']} USD).")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 10: Feature Importance Works Independently for Both Models
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Testing Feature Importance Independently for India and US...")
    try:
        us_fi = get_feature_importance_endpoint(top_n=5, country="US")
        us_country = us_fi["country"] if isinstance(us_fi, dict) else us_fi.country
        us_model = us_fi["model_name"] if isinstance(us_fi, dict) else us_fi.model_name
        us_features = us_fi["top_features"] if isinstance(us_fi, dict) else us_fi.top_features

        assert us_country == "US"
        assert us_model == "Random Forest Regressor"
        assert len(us_features) == 5
        us_top = us_features[0]

        in_fi = get_feature_importance_endpoint(top_n=5, country="IN")
        in_country = in_fi["country"] if isinstance(in_fi, dict) else in_fi.country
        in_model = in_fi["model_name"] if isinstance(in_fi, dict) else in_fi.model_name
        in_features = in_fi["top_features"] if isinstance(in_fi, dict) else in_fi.top_features

        assert in_country == "IN"
        assert in_model == "Gradient Boosting Regressor"
        assert len(in_features) == 5
        in_top = in_features[0]

        us_top_name = us_top["feature"] if isinstance(us_top, dict) else us_top.feature
        us_top_pct = us_top["percentage"] if isinstance(us_top, dict) else us_top.percentage
        in_top_name = in_top["feature"] if isinstance(in_top, dict) else in_top.feature
        in_top_pct = in_top["percentage"] if isinstance(in_top, dict) else in_top.percentage

        print(f"  PASS: US Top Feature: {us_top_name} ({us_top_pct}%)")
        print(f"  PASS: India Top Feature: {in_top_name} ({in_top_pct}%)")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 11: SHAP Explanations Work for Both Models
    # -------------------------------------------------------------------------
    print("\n[TEST 11] Testing SHAP Explanations for India and US Models...")
    try:
        # India SHAP
        in_shap = shap_service.explain_prediction({
            "country": "IN",
            "role_standard": "Data Scientist",
            "experience_level": "SE",
            "primary_city_clean": "Bangalore",
            "company_size": "L",
            "remote_ratio": 50,
            "experience_avg_yrs": 6.0,
            "company_rating": 4.2,
            "skills_count": 7,
            "work_year": 2026,
        })
        assert in_shap["country"] == "IN"
        assert in_shap["currency"] == "INR"
        assert "₹" in in_shap["formatted_predicted"]
        assert len(in_shap["waterfall"]) >= 3

        # US SHAP
        us_shap = shap_service.explain_prediction({
            "country": "US",
            "job_title": "Data Scientist",
            "experience_level": "SE",
            "employment_type": "FT",
            "company_location": "US",
            "company_size": "M",
            "remote_ratio": 100,
            "work_year": 2023,
        })
        assert us_shap["country"] == "US"
        assert us_shap["currency"] == "USD"
        assert "$" in us_shap["formatted_predicted"]
        assert len(us_shap["waterfall"]) >= 3

        print(f"  PASS: India SHAP Base={in_shap['formatted_base']}, Predicted={in_shap['formatted_predicted']}")
        print(f"  PASS: US SHAP Base={us_shap['formatted_base']}, Predicted={us_shap['formatted_predicted']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 12: Health Check Reports Both Models Active
    # -------------------------------------------------------------------------
    print("\n[TEST 12] Testing Health Check Status...")
    try:
        health = health_check_endpoint()
        assert health["status"] == "healthy"
        assert health["models"]["US"]["exists"] is True
        assert health["models"]["IN"]["exists"] is True
        print(f"  PASS: Health check reports status={health['status']} | US={health['models']['US']['model_name']} | IN={health['models']['IN']['model_name']}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} PASSED / {failed} FAILED (TOTAL {passed + failed})")
    print("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
