
"""
SalaryIQ - Exploratory Data Analysis
Indian Tech Jobs Salary Dataset

Input:
    ml/data/processed/cleaned_indian_tech_jobs.csv

Output:
    ml/reports/eda/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_indian_tech_jobs.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("SalaryIQ-EDA")


# ============================================================
# 2. LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

TARGET = "salary_in_lpa"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(subset=[TARGET]).copy()

logger.info("Dataset loaded: %s", df.shape)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. SALARY STATISTICS
# ============================================================

salary = df[TARGET]

print("\n========== SALARY STATISTICS ==========")

print(salary.describe())

salary_stats = {
    "count": int(salary.count()),
    "mean_lpa": round(float(salary.mean()), 2),
    "median_lpa": round(float(salary.median()), 2),
    "std_lpa": round(float(salary.std()), 2),
    "minimum_lpa": round(float(salary.min()), 2),
    "maximum_lpa": round(float(salary.max()), 2),
    "p25_lpa": round(float(salary.quantile(0.25)), 2),
    "p75_lpa": round(float(salary.quantile(0.75)), 2)
}

pd.DataFrame([salary_stats]).to_csv(
    OUTPUT_DIR / "salary_statistics.csv",
    index=False
)


# ============================================================
# 4. SALARY DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x=TARGET,
    bins=30,
    kde=True
)

plt.title("Distribution of Indian Tech Salaries")
plt.xlabel("Annual Salary (LPA)")
plt.ylabel("Number of Job Listings")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_salary_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 5. SALARY BY JOB ROLE
# ============================================================

if "role_standard" in df.columns:

    role_stats = (
        df.groupby("role_standard")[TARGET]
        .agg(["count", "mean", "median"])
        .query("count >= 10")
        .sort_values("median", ascending=False)
    )

    print("\n========== SALARY BY ROLE ==========")
    print(role_stats.round(2))

    role_stats.to_csv(
        OUTPUT_DIR / "salary_by_role.csv"
    )

    if not role_stats.empty:

        plt.figure(figsize=(12, 7))

        sns.barplot(
            data=(
                df[df["role_standard"].isin(role_stats.index)]
                .groupby("role_standard", as_index=False)
                [TARGET].median()
                .sort_values(TARGET, ascending=False)
            ),
            x=TARGET,
            y="role_standard"
        )

        plt.title("Median Salary by Job Role")
        plt.xlabel("Median Annual Salary (LPA)")
        plt.ylabel("Job Role")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "02_salary_by_role.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 6. SALARY BY EXPERIENCE LEVEL
# ============================================================

if "experience_level" in df.columns:

    experience_order = [
        "EN", "MI", "SE", "EX", "Unknown"
    ]

    exp_stats = (
        df.groupby("experience_level")[TARGET]
        .agg(["count", "mean", "median"])
    )

    print("\n========== SALARY BY EXPERIENCE ==========")
    print(exp_stats.round(2))

    exp_stats.to_csv(
        OUTPUT_DIR / "salary_by_experience.csv"
    )

    present_order = [
        x for x in experience_order
        if x in df["experience_level"].unique()
    ]

    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="experience_level",
        y=TARGET,
        order=present_order
    )

    plt.title("Salary Distribution by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Annual Salary (LPA)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "03_salary_by_experience.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. SALARY BY CITY
# ============================================================

if "primary_city_clean" in df.columns:

    city_stats = (
        df.groupby("primary_city_clean")[TARGET]
        .agg(["count", "mean", "median"])
        .query("count >= 10")
        .sort_values("median", ascending=False)
    )

    print("\n========== SALARY BY CITY ==========")
    print(city_stats.round(2))

    city_stats.to_csv(
        OUTPUT_DIR / "salary_by_city.csv"
    )

    if not city_stats.empty:

        city_plot = (
            df[
                df["primary_city_clean"].isin(
                    city_stats.index
                )
            ]
            .groupby("primary_city_clean")[TARGET]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )

        plt.figure(figsize=(11, 7))

        sns.barplot(
            data=city_plot,
            x=TARGET,
            y="primary_city_clean"
        )

        plt.title("Median Salary by Indian City")
        plt.xlabel("Median Annual Salary (LPA)")
        plt.ylabel("City")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "04_salary_by_city.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 8. SALARY BY WORK MODE
# ============================================================

if "work_mode_clean" in df.columns:

    work_stats = (
        df.groupby("work_mode_clean")[TARGET]
        .agg(["count", "mean", "median"])
    )

    print("\n========== SALARY BY WORK MODE ==========")
    print(work_stats.round(2))

    work_stats.to_csv(
        OUTPUT_DIR / "salary_by_work_mode.csv"
    )

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="work_mode_clean",
        y=TARGET
    )

    plt.title("Salary by Work Mode")
    plt.xlabel("Work Mode")
    plt.ylabel("Annual Salary (LPA)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "05_salary_by_work_mode.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. SALARY BY COMPANY SIZE
# ============================================================

if "company_size_label" in df.columns:

    company_stats = (
        df.groupby("company_size_label")[TARGET]
        .agg(["count", "mean", "median"])
    )

    print("\n========== SALARY BY COMPANY SIZE ==========")
    print(company_stats.round(2))

    company_stats.to_csv(
        OUTPUT_DIR / "salary_by_company_size.csv"
    )

    plt.figure(figsize=(9, 6))

    sns.boxplot(
        data=df,
        x="company_size_label",
        y=TARGET
    )

    plt.title("Salary by Company Size")
    plt.xlabel("Company Size")
    plt.ylabel("Annual Salary (LPA)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "06_salary_by_company_size.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. SKILL-WISE SALARY ANALYSIS
# ============================================================

skill_columns = [
    "has_python",
    "has_sql",
    "has_ml",
    "has_cloud"
]

available_skills = [
    col for col in skill_columns
    if col in df.columns
]

skill_results = []

for skill in available_skills:

    for value, label in [
        (1, "Skill present"),
        (0, "Skill not indicated")
    ]:

        subset = df[df[skill] == value]

        skill_results.append({
            "skill": skill,
            "group": label,
            "count": len(subset),
            "median_salary_lpa": (
                round(float(subset[TARGET].median()), 2)
                if not subset.empty else np.nan
            )
        })

skill_df = pd.DataFrame(skill_results)

print("\n========== SKILL-WISE SALARY ==========")
print(skill_df.to_string(index=False))

skill_df.to_csv(
    OUTPUT_DIR / "salary_by_skills.csv",
    index=False
)


# ============================================================
# 11. EXPERIENCE VS SALARY
# ============================================================

if "experience_avg_yrs" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=df,
        x="experience_avg_yrs",
        y=TARGET,
        scatter_kws={"alpha": 0.35},
        line_kws={"color": "red"}
    )

    plt.title("Experience vs Annual Salary")
    plt.xlabel("Average Experience (Years)")
    plt.ylabel("Annual Salary (LPA)")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "07_experience_vs_salary.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 12. NUMERIC CORRELATION
# ============================================================

numeric_df = df.select_dtypes(
    include=[np.number]
)

if not numeric_df.empty:

    correlation = numeric_df.corr()

    print("\n========== NUMERIC CORRELATION ==========")
    print(correlation.round(2))

    correlation.to_csv(
        OUTPUT_DIR / "numeric_correlation.csv"
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )

    plt.title("Numeric Feature Correlation")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "08_correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 13. SAVE FINAL EDA REPORT
# ============================================================

report = {
    "dataset": "SalaryIQ Indian Tech Jobs",
    "total_records": int(len(df)),
    "total_features": int(len(df.columns)),
    "salary_statistics": salary_stats,
    "output_directory": str(OUTPUT_DIR)
}

with open(
    OUTPUT_DIR / "eda_summary.json",
    "w",
    encoding="utf-8"
) as file:

    import json
    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


logger.info("=" * 55)
logger.info("SALARYIQ EDA COMPLETED")
logger.info("=" * 55)
logger.info("Records analyzed: %s", len(df))
logger.info("Charts and reports saved to: %s", OUTPUT_DIR)

print("\nEDA completed successfully!")
print("Output directory:", OUTPUT_DIR)
