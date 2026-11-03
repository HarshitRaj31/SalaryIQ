"""
Deduplication and Validation Script for Model-Ready Indian Tech Jobs Dataset
File: fix_model_dataset.py

Identifies and removes duplicate rows from 'india_salary_model_ready.csv' and
'cleaned_indian_tech_jobs.csv' to ensure strict data integrity for ML training.
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fix_model_dataset")


def get_processed_data_paths() -> Tuple[Path, Path, Path]:
    """
    Resolves file paths for processed CSVs and summary metadata.
    """
    current_file = Path(__file__).resolve()
    candidate_dirs = [
        current_file.parent.parent / "data" / "processed",
        current_file.parent / "data" / "processed",
        Path.cwd() / "Job-Salary-Prediction" / "ml" / "data" / "processed",
        Path.cwd() / "ml" / "data" / "processed",
        Path.cwd() / "data" / "processed",
    ]

    proc_dir = next((d for d in candidate_dirs if d.exists()), None)
    if proc_dir is None:
        raise FileNotFoundError("Could not find 'ml/data/processed' directory.")

    model_ready_path = proc_dir / "india_salary_model_ready.csv"
    comprehensive_path = proc_dir / "cleaned_indian_tech_jobs.csv"
    summary_path = proc_dir / "india_data_summary.json"

    return model_ready_path, comprehensive_path, summary_path


def fix_and_deduplicate_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes the 79 duplicate rows from the model dataset and synchronizes
    both processed CSVs and summary metadata.
    """
    model_ready_path, comp_path, summary_path = get_processed_data_paths()

    logger.info("=" * 65)
    logger.info("Starting Dataset Deduplication Pipeline")
    logger.info("=" * 65)

    # 1. Load model-ready dataset
    if not model_ready_path.exists():
        raise FileNotFoundError(f"Model-ready dataset not found at: {model_ready_path}")

    df_model = pd.read_csv(model_ready_path)
    initial_model_count = len(df_model)
    logger.info(f"Loaded 'india_salary_model_ready.csv': {initial_model_count:,} total records")

    # 2. Audit duplicates
    dup_mask = df_model.duplicated(keep="first")
    num_duplicates = dup_mask.sum()
    unique_count = initial_model_count - num_duplicates

    logger.info(f"Duplicate rows detected: {num_duplicates}")
    logger.info(f"Unique rows after exact deduplication: {unique_count:,}")

    if num_duplicates > 0:
        logger.info("\nSample of identified duplicate rows:")
        sample_dups = df_model[dup_mask].head(5)
        print(
            sample_dups[
                ["role_standard", "experience_level", "primary_city_clean", "remote_ratio", "salary_in_lpa"]
            ].to_string(index=False)
        )
        print()

    # 3. Deduplicate model-ready dataframe
    df_model_clean = df_model.drop_duplicates(keep="first").reset_index(drop=True)
    assert len(df_model_clean) == unique_count, "Deduplication count mismatch!"
    assert df_model_clean.duplicated().sum() == 0, "Residual duplicates still present!"

    # 4. Save deduplicated model-ready dataset
    df_model_clean.to_csv(model_ready_path, index=False)
    logger.info(f"Saved deduplicated model-ready dataset: {model_ready_path} ({len(df_model_clean):,} rows)")

    # 5. Synchronize comprehensive dataset if present
    df_comp_clean = None
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        logger.info(f"\nLoaded 'cleaned_indian_tech_jobs.csv': {len(df_comp):,} records")

        # Deduplicate comprehensive dataset across matching feature subset
        feature_subset = [
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
        ]
        available_subset = [c for c in feature_subset if c in df_comp.columns]
        df_comp_clean = df_comp.drop_duplicates(subset=available_subset, keep="first").reset_index(drop=True)

        logger.info(f"Deduplicated 'cleaned_indian_tech_jobs.csv' down to {len(df_comp_clean):,} rows")
        df_comp_clean.to_csv(comp_path, index=False)
        logger.info(f"Saved deduplicated comprehensive dataset: {comp_path}")

    # 6. Update summary metadata JSON
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            summary["cleaned_records"] = len(df_model_clean)
            summary["deduplication_audit"] = {
                "records_before_deduplication": initial_model_count,
                "duplicates_removed": int(num_duplicates),
                "unique_records": len(df_model_clean),
                "is_deduplicated": True,
            }
            summary["salary_statistics_lpa"] = {
                "mean": round(float(df_model_clean["salary_in_lpa"].mean()), 2),
                "std": round(float(df_model_clean["salary_in_lpa"].std()), 2),
                "min": round(float(df_model_clean["salary_in_lpa"].min()), 2),
                "p25": round(float(df_model_clean["salary_in_lpa"].quantile(0.25)), 2),
                "median": round(float(df_model_clean["salary_in_lpa"].median()), 2),
                "p75": round(float(df_model_clean["salary_in_lpa"].quantile(0.75)), 2),
                "max": round(float(df_model_clean["salary_in_lpa"].max()), 2),
            }
            summary["salary_statistics_usd"] = {
                "mean": int(df_model_clean["salary_in_usd"].mean()),
                "median": int(df_model_clean["salary_in_usd"].median()),
                "min": int(df_model_clean["salary_in_usd"].min()),
                "max": int(df_model_clean["salary_in_usd"].max()),
            }
            summary["role_distribution"] = df_model_clean["role_standard"].value_counts().to_dict()
            summary["experience_tier_distribution"] = df_model_clean["experience_level"].value_counts().to_dict()
            summary["primary_city_distribution"] = df_model_clean["primary_city_clean"].value_counts().to_dict()
            summary["company_size_distribution"] = df_model_clean["company_size"].value_counts().to_dict()

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Updated metadata summary: {summary_path}")
        except Exception as e:
            logger.warning(f"Could not update summary JSON: {e}")

    logger.info("=" * 65)
    logger.info("Deduplication Pipeline Completed Successfully!")
    logger.info(f"Final Cleaned Dataset: {len(df_model_clean):,} records (0 duplicates)")
    logger.info("=" * 65)

    return df_model_clean, df_comp_clean


if __name__ == "__main__":
    fix_and_deduplicate_dataset()

