"""
Data Cleaning and Preprocessing Pipeline for Indian Tech Jobs Dataset
File: clean_india_data.py

Cleans, validates, and standardizes 'indian_tech_jobs_2026.csv' to produce
a high-quality dataset for machine learning salary prediction in India (LPA / INR / USD).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clean_india_data")

# Default conversion rate: 1 USD = 84.0 INR (approximate market benchmark)
DEFAULT_USD_INR_RATE = 84.0


def find_raw_data_file() -> Path:
    """
    Locates the raw Indian tech jobs CSV file across standard candidate directories.
    """
    current_file = Path(__file__).resolve()
    # Candidate search directories
    search_dirs = [
        current_file.parent.parent / "data" / "raw",
        current_file.parent / "data" / "raw",
        Path.cwd() / "Job-Salary-Prediction" / "ml" / "data" / "raw",
        Path.cwd() / "ml" / "data" / "raw",
        Path.cwd() / "data" / "raw",
        Path.cwd(),
    ]

    filename_patterns = [
        "indian_tech_jobs_2026.csv",
        "india teech job.csv",
        "india_tech_jobs.csv",
        "*indian*tech*jobs*.csv",
    ]

    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in filename_patterns:
            matches = list(d.glob(pattern))
            if matches:
                logger.info(f"Located raw dataset at: {matches[0]}")
                return matches[0]

    raise FileNotFoundError(
        "Could not find 'indian_tech_jobs_2026.csv' or 'india teech job.csv' in expected paths."
    )


def clean_primary_city(city_val: str) -> str:
    """
    Cleans sub-localities (e.g., 'Mumbai(Powai)' -> 'Mumbai') and normalizes metro clusters.
    """
    if pd.isna(city_val):
        return "Other"

    city = str(city_val).strip()
    # Strip sub-locality parentheses, e.g. "Mumbai(Powai)" -> "Mumbai"
    if "(" in city:
        city = city.split("(")[0].strip()

    city_lower = city.lower()

    if any(m in city_lower for m in ["bangalore", "bengaluru"]):
        return "Bangalore"
    if any(m in city_lower for m in ["mumbai", "bombay"]):
        return "Mumbai"
    if any(m in city_lower for m in ["pune"]):
        return "Pune"
    if any(m in city_lower for m in ["hyderabad", "secunderabad"]):
        return "Hyderabad"
    if any(m in city_lower for m in ["chennai", "madras"]):
        return "Chennai"
    if any(m in city_lower for m in ["noida", "gurgaon", "gurugram", "delhi", "faridabad", "ghaziabad"]):
        return "Delhi / NCR"
    if any(m in city_lower for m in ["kolkata", "calcutta"]):
        return "Kolkata"
    if any(m in city_lower for m in ["ahmedabad"]):
        return "Ahmedabad"
    if any(m in city_lower for m in ["remote"]):
        return "Remote"

    return city if city else "Other"


def clean_role_category(role_val: str, title_val: str = "") -> str:
    """
    Standardizes tech roles into top representative career tracks.
    """
    combined = f"{role_val} {title_val}".lower()

    if any(k in combined for k in ["data scientist", "scientist", "applied scientist"]):
        return "Data Scientist"
    if any(k in combined for k in ["machine learning", "ml engineer", "mlops", "deep learning", "ai engineer"]):
        return "Machine Learning Engineer"
    if any(k in combined for k in ["data engineer", "etl", "big data", "databricks"]):
        return "Data Engineer"
    if any(k in combined for k in ["data analyst", "business intelligence", "bi analyst", "bi developer", "power bi", "tableau"]):
        return "Data Analyst"
    if any(k in combined for k in ["business analyst", "product analyst", "business consultant"]):
        return "Business Analyst"
    if any(k in combined for k in ["python developer", "python engineer"]):
        return "Python Developer"
    if any(k in combined for k in ["devops", "cloud", "aws", "azure", "kubernetes"]):
        return "Cloud & DevOps"
    if any(k in combined for k in ["software engineer", "developer", "backend", "full stack"]):
        return "Software Engineer"

    # Default to raw role category if title or fallback
    return str(role_val).strip() if pd.notna(role_val) else "Other Tech"


def map_experience_tier(tier_val: str, min_yrs: float = 0.0, max_yrs: float = 0.0) -> Tuple[str, str]:
    """
    Maps Indian experience buckets to standard 4-tier codes matching SalaryIQ:
    - EN: Entry-level (0-2 Yrs)
    - MI: Mid-level (3-5 Yrs)
    - SE: Senior-level (6-8 Yrs)
    - EX: Executive / Lead (9+ Yrs)
    """
    avg_yrs = (min_yrs + max_yrs) / 2.0 if (pd.notna(min_yrs) and pd.notna(max_yrs)) else 0.0
    tier_str = str(tier_val).lower() if pd.notna(tier_val) else ""

    if "lead" in tier_str or "architect" in tier_str or "9+" in tier_str or avg_yrs >= 9:
        return "EX", "Executive / Lead (EX)"
    if "senior" in tier_str or "6-8" in tier_str or avg_yrs >= 6:
        return "SE", "Senior-level (SE)"
    if "mid" in tier_str or "3-5" in tier_str or avg_yrs >= 3:
        return "MI", "Mid-level (MI)"
    return "EN", "Entry-level (EN)"


def map_work_mode(mode_val: str) -> Tuple[int, str]:
    """
    Maps work mode ('On-site', 'Hybrid', 'Remote') to standard remote_ratio percentage.
    """
    mode = str(mode_val).lower().strip() if pd.notna(mode_val) else ""
    if "remote" in mode:
        return 100, "Remote"
    if "hybrid" in mode:
        return 50, "Hybrid"
    return 0, "On-site"


def map_company_size(size_val: str) -> Tuple[str, str]:
    """
    Maps company size bucket to standard S / M / L codes.
    """
    size_str = str(size_val).lower().strip() if pd.notna(size_val) else ""
    if "large" in size_str or "1000+" in size_str:
        return "L", "Large (1000+)"
    if "mid" in size_str or "100-999" in size_str:
        return "M", "Medium (100-999)"
    return "S", "Small (<100)"


def clean_indian_tech_jobs(
    input_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    min_salary_lpa: float = 1.5,
    max_salary_lpa: float = 120.0,
    usd_inr_rate: float = DEFAULT_USD_INR_RATE,
) -> pd.DataFrame:
    """
    Full data cleaning and preprocessing pipeline.

    Parameters:
        input_path: Path to raw CSV file (or auto-detected if None)
        output_dir: Directory to save cleaned CSV and metadata (default: ml/data/processed)
        min_salary_lpa: Minimum annual salary in Lakhs Per Annum (filters stipends < 1.5 LPA)
        max_salary_lpa: Maximum annual salary cutoff in LPA
        usd_inr_rate: Conversion exchange rate for USD calculation

    Returns:
        pd.DataFrame: Cleaned, preprocessed DataFrame ready for machine learning.
    """
    raw_path = Path(input_path) if input_path else find_raw_data_file()
    logger.info(f"Loading raw dataset from: {raw_path}")

    df = pd.read_csv(raw_path)
    initial_rows = len(df)
    logger.info(f"Initial shape: {df.shape} ({initial_rows:,} rows, {df.shape[1]} columns)")

    # -------------------------------------------------------------
    # 1. Filter Disclosed & Paid Salaries
    # -------------------------------------------------------------
    # Disclosed salary check
    df_paid = df[df["salary_disclosed"] == True].copy()
    logger.info(f"Rows with salary_disclosed == True: {len(df_paid):,} (dropped {initial_rows - len(df_paid):,} undisclosed)")

    # Exclude unpaid / volunteer jobs
    df_paid = df_paid[df_paid["salary_raw"] != "Unpaid"].copy()

    # Exclude zero or missing salary midpoints
    df_paid = df_paid[df_paid["salary_midpoint_lpa"] > 0].copy()

    # Filter out internship stipends and anomalous sub-minimums (< 1.5 LPA)
    # Exclude title internships
    intern_mask = df_paid["job_title"].str.contains(r"\b(?:intern|internship|apprentice|articleship)\b", case=False, na=False)
    df_paid = df_paid[~intern_mask].copy()

    # Realistic full-time professional salary bounds (1.5 LPA to 120 LPA)
    df_paid = df_paid[
        (df_paid["salary_midpoint_lpa"] >= min_salary_lpa)
        & (df_paid["salary_midpoint_lpa"] <= max_salary_lpa)
    ].copy()

    logger.info(f"Rows after salary validation (>= {min_salary_lpa} LPA & professional): {len(df_paid):,}")

    # -------------------------------------------------------------
    # 2. Clean & Standardize Experience
    # -------------------------------------------------------------
    df_paid["experience_min_yrs"] = pd.to_numeric(df_paid["experience_min_yrs"], errors="coerce").fillna(0.0)
    df_paid["experience_max_yrs"] = pd.to_numeric(df_paid["experience_max_yrs"], errors="coerce").fillna(df_paid["experience_min_yrs"])
    df_paid["experience_avg_yrs"] = (df_paid["experience_min_yrs"] + df_paid["experience_max_yrs"]) / 2.0

    exp_mappings = df_paid.apply(
        lambda row: map_experience_tier(
            row.get("experience_tier", ""),
            row.get("experience_min_yrs", 0),
            row.get("experience_max_yrs", 0),
        ),
        axis=1,
    )
    df_paid["experience_level"] = [m[0] for m in exp_mappings]
    df_paid["experience_level_label"] = [m[1] for m in exp_mappings]

    # -------------------------------------------------------------
    # 3. Clean & Standardize Roles & Job Titles
    # -------------------------------------------------------------
    df_paid["role_standard"] = df_paid.apply(
        lambda row: clean_role_category(row.get("role_category", ""), row.get("job_title", "")),
        axis=1,
    )

    # -------------------------------------------------------------
    # 4. Clean & Standardize City / Location
    # -------------------------------------------------------------
    df_paid["primary_city_clean"] = df_paid["primary_city"].apply(clean_primary_city)

    # -------------------------------------------------------------
    # 5. Clean Work Mode & Remote Ratio
    # -------------------------------------------------------------
    mode_mappings = df_paid["work_mode"].apply(map_work_mode)
    df_paid["remote_ratio"] = [m[0] for m in mode_mappings]
    df_paid["work_mode_clean"] = [m[1] for m in mode_mappings]

    # -------------------------------------------------------------
    # 6. Clean Company Size Bucket
    # -------------------------------------------------------------
    size_mappings = df_paid["company_size_bucket"].apply(map_company_size)
    df_paid["company_size"] = [m[0] for m in size_mappings]
    df_paid["company_size_label"] = [m[1] for m in size_mappings]

    # -------------------------------------------------------------
    # 7. Clean Company Ratings & Skills
    # -------------------------------------------------------------
    median_rating = round(df_paid["company_rating"].median(), 2)
    df_paid["company_rating"] = df_paid["company_rating"].fillna(median_rating)

    df_paid["skills_count"] = pd.to_numeric(df_paid["skills_count"], errors="coerce").fillna(0).astype(int)
    df_paid["skills_required"] = df_paid["skills_required"].fillna("").astype(str)

    # Top in-demand technology binary indicators
    df_paid["has_python"] = df_paid["skills_required"].str.contains(r"\bpython\b", case=False, na=False).astype(int)
    df_paid["has_sql"] = df_paid["skills_required"].str.contains(r"\bsql\b", case=False, na=False).astype(int)
    df_paid["has_ml"] = df_paid["skills_required"].str.contains(r"\b(?:machine learning|ml|deep learning)\b", case=False, na=False).astype(int)
    df_paid["has_cloud"] = df_paid["skills_required"].str.contains(r"\b(?:aws|azure|gcp|cloud)\b", case=False, na=False).astype(int)

    # -------------------------------------------------------------
    # 8. Compute Standard Multi-Currency Targets
    # -------------------------------------------------------------
    # Primary target in LPA (Lakhs Per Annum)
    df_paid["salary_in_lpa"] = df_paid["salary_midpoint_lpa"].round(2)

    # Annual target in Indian Rupees (INR)
    df_paid["salary_in_inr"] = (df_paid["salary_in_lpa"] * 100000.0).round(0).astype(int)

    # Formatted INR strings (safe ASCII + unicode rupee symbol)
    df_paid["formatted_salary_inr"] = df_paid["salary_in_lpa"].apply(lambda v: f"INR {v:,.2f} LPA")
    df_paid["formatted_salary_inr_symbol"] = df_paid["salary_in_lpa"].apply(lambda v: f"₹{v:,.2f} LPA")

    # Annual target in USD (converted at standard rate)
    df_paid["salary_in_usd"] = (df_paid["salary_in_inr"] / usd_inr_rate).round(0).astype(int)
    df_paid["formatted_salary_usd"] = df_paid["salary_in_usd"].apply(lambda v: f"${v:,}")

    # Standard macroeconomic year
    df_paid["work_year"] = 2026
    df_paid["company_location"] = "IN"
    df_paid["employee_residence"] = "IN"
    df_paid["employment_type"] = "FT"

    # Remove any duplicate job listings
    duplicates_before = df_paid.duplicated(subset=["job_title", "company_name", "location", "salary_in_lpa"]).sum()
    df_clean = df_paid.drop_duplicates(subset=["job_title", "company_name", "location", "salary_in_lpa"]).copy()
    logger.info(f"Removed duplicate job postings: {duplicates_before} (Cleaned total: {len(df_clean):,})")

    # -------------------------------------------------------------
    # 9. Structure Final Preprocessed Datasets
    # -------------------------------------------------------------
    # Comprehensive cleaned dataset
    columns_comprehensive = [
        "job_id",
        "job_title",
        "role_standard",
        "company_name",
        "company_rating",
        "primary_city_clean",
        "location",
        "experience_level",
        "experience_level_label",
        "experience_min_yrs",
        "experience_max_yrs",
        "experience_avg_yrs",
        "work_mode_clean",
        "remote_ratio",
        "company_size",
        "company_size_label",
        "salary_in_lpa",
        "salary_min_lpa",
        "salary_max_lpa",
        "salary_in_inr",
        "formatted_salary_inr",
        "salary_in_usd",
        "formatted_salary_usd",
        "skills_count",
        "has_python",
        "has_sql",
        "has_ml",
        "has_cloud",
        "skills_required",
        "company_location",
        "employee_residence",
        "employment_type",
        "work_year",
    ]

    # Model-ready schema matching scikit-learn training pipeline
    columns_model_ready = [
        "work_year",
        "experience_level",
        "employment_type",
        "role_standard",
        "primary_city_clean",
        "remote_ratio",
        "company_size",
        "experience_avg_yrs",
        "company_rating",
        "skills_count",
        "salary_in_lpa",
        "salary_in_usd",
    ]

    # Deduplicate across model features to eliminate redundant job postings
    model_df = df_clean[columns_model_ready].drop_duplicates(keep="first").copy()
    final_df = df_clean.loc[model_df.index][columns_comprehensive].copy().reset_index(drop=True)
    model_df = model_df.reset_index(drop=True)
    logger.info(f"Final deduplicated dataset: {len(model_df):,} records (0 duplicates)")

    # -------------------------------------------------------------
    # 10. Determine Output Directory & Save
    # -------------------------------------------------------------
    if output_dir:
        out_dir = Path(output_dir)
    else:
        current_file = Path(__file__).resolve()
        candidate_proc = [
            current_file.parent.parent / "data" / "processed",
            Path.cwd() / "Job-Salary-Prediction" / "ml" / "data" / "processed",
            Path.cwd() / "ml" / "data" / "processed",
            Path.cwd() / "data" / "processed",
        ]
        out_dir = next((d for d in candidate_proc if d.parent.exists()), Path("data/processed"))

    out_dir.mkdir(parents=True, exist_ok=True)

    comprehensive_csv = out_dir / "cleaned_indian_tech_jobs.csv"
    model_ready_csv = out_dir / "india_salary_model_ready.csv"
    summary_json = out_dir / "india_data_summary.json"

    final_df.to_csv(comprehensive_csv, index=False)
    logger.info(f"Saved comprehensive cleaned dataset to: {comprehensive_csv}")

    model_df.to_csv(model_ready_csv, index=False)
    logger.info(f"Saved model-ready dataset to: {model_ready_csv}")

    # Generate and save summary statistics metadata
    summary_stats = {
        "dataset_name": "Indian Tech Jobs Salary Dataset (2026)",
        "source_file": raw_path.name,
        "initial_raw_records": initial_rows,
        "cleaned_records": len(final_df),
        "target_currency_primary": "INR (Lakhs Per Annum - LPA)",
        "target_currency_secondary": "USD",
        "usd_inr_exchange_rate": usd_inr_rate,
        "salary_statistics_lpa": {
            "mean": round(float(final_df["salary_in_lpa"].mean()), 2),
            "std": round(float(final_df["salary_in_lpa"].std()), 2),
            "min": round(float(final_df["salary_in_lpa"].min()), 2),
            "p25": round(float(final_df["salary_in_lpa"].quantile(0.25)), 2),
            "median": round(float(final_df["salary_in_lpa"].median()), 2),
            "p75": round(float(final_df["salary_in_lpa"].quantile(0.75)), 2),
            "max": round(float(final_df["salary_in_lpa"].max()), 2),
        },
        "salary_statistics_usd": {
            "mean": int(final_df["salary_in_usd"].mean()),
            "median": int(final_df["salary_in_usd"].median()),
            "min": int(final_df["salary_in_usd"].min()),
            "max": int(final_df["salary_in_usd"].max()),
        },
        "role_distribution": final_df["role_standard"].value_counts().to_dict(),
        "experience_tier_distribution": final_df["experience_level"].value_counts().to_dict(),
        "primary_city_distribution": final_df["primary_city_clean"].value_counts().to_dict(),
        "work_mode_distribution": final_df["work_mode_clean"].value_counts().to_dict(),
        "company_size_distribution": final_df["company_size"].value_counts().to_dict(),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"Saved preprocessing summary to: {summary_json}")

    return final_df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Indian Tech Jobs Preprocessing Pipeline")
    logger.info("=" * 60)

    cleaned_data = clean_indian_tech_jobs()

    logger.info("\n" + "=" * 60)
    logger.info("Cleaning Pipeline Completed Successfully!")
    logger.info(f"Final Cleaned Records: {len(cleaned_data):,}")
    logger.info("=" * 60)
    logger.info(
        f"Salary Distribution (LPA): Min={cleaned_data['salary_in_lpa'].min()} | "
        f"Median={cleaned_data['salary_in_lpa'].median()} | "
        f"Mean={cleaned_data['salary_in_lpa'].mean():.2f} | "
        f"Max={cleaned_data['salary_in_lpa'].max()}"
    )
    logger.info("Sample 3 Cleaned Records:")
    sample_cols = [
        "role_standard",
        "experience_level",
        "primary_city_clean",
        "work_mode_clean",
        "formatted_salary_inr",
        "formatted_salary_usd",
    ]
    print(cleaned_data[sample_cols].head(3).to_string(index=False))
