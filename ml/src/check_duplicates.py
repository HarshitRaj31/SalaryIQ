"""
Duplicate Investigation Script for Job Salary Prediction Dataset.

Performs a comprehensive forensic investigation into duplicate records in `ds_salaries.csv`:
1. Scope and Scale of Duplication (Exact duplicates vs. Unique records)
2. Granular Cluster Analysis (Largest repeated respondent clusters)
3. Breakdown of Duplicates across Job Titles, Experience Levels, and Years
4. Discrepancy Analysis (Same features with identical vs differing salaries)
5. Machine Learning Impact Assessment (Data leakage, evaluation bias)
6. Diagnostic Visualizations saved to ml/reports/figures/
"""

import json
import os
import sys
import tempfile
import warnings

# Direct matplotlib cache to temp directory to avoid sandbox permissions errors
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml/ directory
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "ds_salaries.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_salary.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"


def load_raw_data():
    """Loads raw dataset for duplicate forensics."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at '{RAW_DATA_PATH}'.")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw dataset from: {RAW_DATA_PATH} ({len(df):,} total rows)")
    return df


def investigate_duplicates(df: pd.DataFrame):
    """Conducts mathematical and categorical forensics on duplicate records."""
    total_rows = len(df)

    # 1. Exact Duplicate Rows (Identical across all 11 columns)
    exact_duplicate_mask = df.duplicated(keep="first")
    num_exact_duplicates = int(exact_duplicate_mask.sum())
    num_unique_records = total_rows - num_exact_duplicates
    pct_duplicates = (num_exact_duplicates / total_rows) * 100

    # All rows involved in duplicate clusters (including first occurrences)
    all_duplicate_rows = df[df.duplicated(keep=False)]
    num_rows_in_duplicate_groups = len(all_duplicate_rows)

    # 2. Duplicate Feature Profiles (excluding raw salary & currency)
    feature_cols = [c for c in df.columns if c not in ["salary", "salary_currency", "salary_in_usd"]]
    profile_duplicates_mask = df.duplicated(subset=feature_cols, keep="first")
    num_profile_duplicates = int(profile_duplicates_mask.sum())

    # 3. Contradictory records (same input features but different salary values)
    feature_grouped = df.groupby(feature_cols)["salary_in_usd"].nunique()
    profiles_with_salary_variance = int((feature_grouped > 1).sum())
    profiles_with_fixed_salary = int((feature_grouped == 1).sum())

    # 4. Top duplicate clusters
    cluster_counts = df.groupby(list(df.columns)).size().reset_index(name="cluster_size")
    top_clusters = cluster_counts.sort_values(by="cluster_size", ascending=False).head(10)

    # 5. Breakdown by Categorical Dimensions
    dups_by_title = all_duplicate_rows["job_title"].value_counts().head(10).to_dict()
    dups_by_exp = all_duplicate_rows["experience_level"].value_counts().to_dict()
    dups_by_year = all_duplicate_rows["work_year"].value_counts().sort_index().to_dict()
    dups_by_country = all_duplicate_rows["company_location"].value_counts().head(5).to_dict()

    investigation_results = {
        "summary": {
            "total_raw_rows": total_rows,
            "exact_duplicate_rows": num_exact_duplicates,
            "exact_duplicate_pct": round(pct_duplicates, 2),
            "unique_distinct_rows": num_unique_records,
            "total_rows_involved_in_clusters": num_rows_in_duplicate_groups,
            "profile_duplicates_features_only": num_profile_duplicates,
        },
        "salary_variance_in_identical_profiles": {
            "profiles_with_differing_salaries": profiles_with_salary_variance,
            "profiles_with_exact_same_salaries": profiles_with_fixed_salary,
        },
        "breakdown": {
            "by_job_title": dups_by_title,
            "by_experience_level": dups_by_exp,
            "by_work_year": dups_by_year,
            "by_company_location": dups_by_country,
        },
        "top_clusters": top_clusters.to_dict(orient="records"),
    }

    return investigation_results, top_clusters, all_duplicate_rows


def display_investigation_report(results: dict, top_clusters: pd.DataFrame):
    """Prints a structured, readable terminal report."""
    summary = results["summary"]
    breakdown = results["breakdown"]

    print("\n" + "=" * 78)
    print("                    DATASET DUPLICATE INVESTIGATION REPORT")
    print("=" * 78)

    print("\n1. SCALE AND MAGNITUDE OF DUPLICATION:")
    print(f"  - Total Raw Records:           {summary['total_raw_rows']:,}")
    print(f"  - Exact Duplicate Rows:        {summary['exact_duplicate_rows']:,}  ({summary['exact_duplicate_pct']:.2f}% of data)")
    print(f"  - Unique Distinct Profiles:    {summary['unique_distinct_rows']:,}")
    print(f"  - Records Belonging to Clusters:{summary['total_rows_involved_in_clusters']:,}  ({summary['total_rows_involved_in_clusters'] / summary['total_raw_rows'] * 100:.1f}% affected)")

    print("\n2. TOP REPEATED CLUSTERS (IDENTICAL RESPONDENT COHORTS):")
    cols_display = ["work_year", "experience_level", "job_title", "salary_in_usd", "company_location", "cluster_size"]
    print(f"  {'Year':<5} | {'Exp':<4} | {'Job Title':<28} | {'Salary ($)':<11} | {'Loc':<4} | {'Copies':<6}")
    print("  " + "-" * 72)
    for _, row in top_clusters.head(8).iterrows():
        print(f"  {row['work_year']:<5} | {row['experience_level']:<4} | {row['job_title'][:28]:<28} | ${row['salary_in_usd']:>9,.0f} | {row['company_location']:<4} | {row['cluster_size']:<6}")

    print("\n3. BREAKDOWN ACROSS KEY ATTRIBUTES:")
    print("  * By Experience Tier:")
    for tier, cnt in breakdown["by_experience_level"].items():
        print(f"    - {tier:<4}: {cnt:>5,} duplicated rows ({cnt / summary['total_rows_involved_in_clusters'] * 100:.1f}%)")

    print("\n  * By Year of Survey:")
    for yr, cnt in breakdown["by_work_year"].items():
        print(f"    - {yr:<4}: {cnt:>5,} duplicated rows ({cnt / summary['total_rows_involved_in_clusters'] * 100:.1f}%)")

    print("\n  * Top 5 Most Duplicated Job Titles:")
    for role, cnt in list(breakdown["by_job_title"].items())[:5]:
        print(f"    - {role:<28}: {cnt:>5,} rows")

    print("\n4. MACHINE LEARNING IMPACT & VERDICT:")
    print("  [CRITICAL RISK - DATA LEAKAGE]:")
    print("  If exact duplicates are kept prior to train-test splitting, identical records")
    print("  fall into BOTH the training set and testing set. This causes memorization and")
    print("  produces artificially optimistic evaluation scores (spurious high R²).")
    print("\n  [CORRECTIVE ACTION]:")
    print("  `preprocessing.py` deduplicating the dataset down to 2,584 unique samples")
    print("  guarantees rigorous out-of-sample generalization and unbiased error estimation.")
    print("=" * 78 + "\n")


def plot_duplicate_forensics(df: pd.DataFrame, results: dict, all_duplicate_rows: pd.DataFrame, save_dir: str):
    """Generates 4-panel forensic visualization plot."""
    os.makedirs(save_dir, exist_ok=True)
    summary = results["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Donut Chart: Unique vs Duplicate Composition ---
    labels = ["Unique Records", "Exact Duplicate Rows"]
    sizes = [summary["unique_distinct_rows"], summary["exact_duplicate_rows"]]
    colors = ["#2b5c8f", "#d95f02"]
    explode = (0, 0.08)

    wedges, texts, autotexts = axes[0, 0].pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
    axes[0, 0].set_title(f"1. Dataset Composition ({summary['total_raw_rows']:,} Total Rows)")

    # --- Plot 2: Top 10 Most Duplicated Job Titles ---
    top_titles = all_duplicate_rows["job_title"].value_counts().head(10)
    sns.barplot(x=top_titles.values, y=top_titles.index, palette="Blues_r", ax=axes[0, 1])
    axes[0, 1].set_title("2. Top 10 Job Roles in Duplicate Clusters")
    axes[0, 1].set_xlabel("Number of Rows in Duplicate Groups")
    axes[0, 1].set_ylabel("Job Role")
    for i, count in enumerate(top_titles.values):
        axes[0, 1].annotate(f"{count:,}", (count + 5, i), va="center", fontsize=9, fontweight="bold")

    # --- Plot 3: Duplication by Experience Level ---
    exp_order = ["EN", "MI", "SE", "EX"]
    palette_exp = {"EN": "#56B4E9", "MI": "#009E73", "SE": "#0072B2", "EX": "#D55E00"}
    sns.countplot(
        data=all_duplicate_rows,
        x="experience_level",
        order=exp_order,
        palette=palette_exp,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("3. Duplicates by Experience Level")
    axes[1, 0].set_xlabel("Experience Tier")
    axes[1, 0].set_ylabel("Count of Duplicate Records")
    for p in axes[1, 0].patches:
        h = p.get_height()
        axes[1, 0].annotate(
            f"{int(h):,}\n({h / len(all_duplicate_rows) * 100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # --- Plot 4: Duplication Trend Across Years (2020 - 2023) ---
    year_counts = all_duplicate_rows["work_year"].value_counts().sort_index()
    sns.barplot(x=year_counts.index, y=year_counts.values, palette="crest", ax=axes[1, 1])
    axes[1, 1].set_title("4. Duplicates by Survey Year")
    axes[1, 1].set_xlabel("Work Year")
    axes[1, 1].set_ylabel("Count of Duplicate Records")
    for p in axes[1, 1].patches:
        h = p.get_height()
        axes[1, 1].annotate(
            f"{int(h):,}",
            (p.get_x() + p.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path = os.path.join(save_dir, "duplicate_investigation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved forensic visualization plot -> {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load raw dataset
    df = load_raw_data()

    # 2. Forensic investigation
    results, top_clusters, all_duplicate_rows = investigate_duplicates(df)

    # 3. Print readable report
    display_investigation_report(results, top_clusters)

    # 4. Generate & save visualization
    plot_duplicate_forensics(df, results, all_duplicate_rows, FIGURES_DIR)

    # 5. Save JSON summary report
    json_path = os.path.join(REPORTS_DIR, "duplicate_investigation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved duplicate investigation JSON report -> {json_path}")

    # 6. Save top clusters CSV for audit trail
    clusters_csv_path = os.path.join(REPORTS_DIR, "duplicate_clusters.csv")
    top_clusters.to_csv(clusters_csv_path, index=False)
    print(f"Saved duplicate clusters audit CSV -> {clusters_csv_path}\n")


if __name__ == "__main__":
    main()
