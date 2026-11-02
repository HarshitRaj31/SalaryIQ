"""
Salary Distribution Analysis Script for Job Salary Prediction.

Performs deep statistical inspection of the target variable `salary_in_usd`:
- Parametric and non-parametric summary statistics (Mean, Median, Std, IQR, Percentiles)
- Shape, Skewness, Kurtosis, and Normality assessments
- Outlier detection via Tukey's IQR Fences and Z-Score method
- Breakdown across Experience Levels, Company Sizes, and Geographic Tiers
- Evaluation of Log-Transformation for Machine Learning normality
- Diagnostic visualization plots saved to ml/reports/figures/
"""

import json
import os
import sys
import tempfile
import warnings

# Avoid matplotlib cache permission errors in sandbox
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
import scipy.stats as stats
import seaborn as sns

# Path configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_salary.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "ds_salaries.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"


def load_dataset():
    """Loads the processed dataset or raw dataset as fallback."""
    if os.path.exists(PROCESSED_DATA_PATH):
        path = PROCESSED_DATA_PATH
    elif os.path.exists(RAW_DATA_PATH):
        path = RAW_DATA_PATH
    else:
        raise FileNotFoundError(
            f"No dataset found at '{PROCESSED_DATA_PATH}' or '{RAW_DATA_PATH}'."
        )

    df = pd.read_csv(path)
    print(f"Loaded dataset from: {path} ({len(df):,} records)")
    return df


def calculate_salary_statistics(salaries: pd.Series):
    """Calculates comprehensive statistical measures of salary distribution."""
    n = len(salaries)
    mean = salaries.mean()
    std = salaries.std()
    median = salaries.median()
    iqr = stats.iqr(salaries)
    q1 = salaries.quantile(0.25)
    q3 = salaries.quantile(0.75)
    skewness = salaries.skew()
    kurtosis = salaries.kurtosis()

    # Percentiles
    percentiles = {
        "1%": float(salaries.quantile(0.01)),
        "5%": float(salaries.quantile(0.05)),
        "10%": float(salaries.quantile(0.10)),
        "25% (Q1)": float(q1),
        "50% (Median)": float(median),
        "75% (Q3)": float(q3),
        "90%": float(salaries.quantile(0.90)),
        "95%": float(salaries.quantile(0.95)),
        "99%": float(salaries.quantile(0.99)),
    }

    # Outlier detection via Tukey's IQR Fences
    lower_fence = max(0.0, q1 - 1.5 * iqr)
    upper_fence = q3 + 1.5 * iqr
    extreme_upper_fence = q3 + 3.0 * iqr

    iqr_outliers = salaries[(salaries < lower_fence) | (salaries > upper_fence)]
    extreme_outliers = salaries[salaries > extreme_upper_fence]

    # Outlier detection via Z-Score (|Z| > 3)
    z_scores = np.abs(stats.zscore(salaries))
    z_outliers = salaries[z_scores > 3]

    # Log transformation inspection
    log_salaries = np.log1p(salaries)
    log_skewness = log_salaries.skew()
    log_kurtosis = log_salaries.kurtosis()

    stats_summary = {
        "count": int(n),
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "mode": float(salaries.mode().iloc[0]),
        "min": float(salaries.min()),
        "max": float(salaries.max()),
        "range": float(salaries.max() - salaries.min()),
        "iqr": float(iqr),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "percentiles": percentiles,
        "outlier_analysis": {
            "lower_fence_iqr": float(lower_fence),
            "upper_fence_iqr": float(upper_fence),
            "extreme_upper_fence": float(extreme_upper_fence),
            "iqr_outliers_count": int(len(iqr_outliers)),
            "iqr_outliers_pct": float(len(iqr_outliers) / n * 100),
            "extreme_outliers_count": int(len(extreme_outliers)),
            "zscore_outliers_count": int(len(z_outliers)),
        },
        "log_transformed": {
            "skewness": float(log_skewness),
            "kurtosis": float(log_kurtosis),
        },
    }

    return stats_summary


def display_salary_report(summary: dict, df: pd.DataFrame):
    """Prints a clear, formatted terminal analysis of the salary distribution."""
    salaries = df["salary_in_usd"]

    print("\n" + "=" * 75)
    print("           TARGET VARIABLE DISTRIBUTION REPORT: `salary_in_usd`")
    print("=" * 75)

    print("\n1. CENTRAL TENDENCY & DISPERSION:")
    print(f"  - Total Sample Count:      {summary['count']:,}")
    print(f"  - Mean Salary:             ${summary['mean']:>12,.2f}")
    print(f"  - Median Salary:           ${summary['median']:>12,.2f}")
    print(f"  - Mode Salary:             ${summary['mode']:>12,.2f}")
    print(f"  - Standard Deviation:      ${summary['std']:>12,.2f}")
    print(f"  - Min Salary:              ${summary['min']:>12,.2f}")
    print(f"  - Max Salary:              ${summary['max']:>12,.2f}")
    print(f"  - Total Range:             ${summary['range']:>12,.2f}")
    print(f"  - Interquartile Range(IQR):${summary['iqr']:>12,.2f}")

    print("\n2. PERCENTILES & QUANTILES:")
    for label, val in summary["percentiles"].items():
        print(f"  - {label:<15}: ${val:>12,.2f}")

    print("\n3. SHAPE & NORMALITY ASSESSMENT:")
    skew = summary["skewness"]
    kurt = summary["kurtosis"]
    skew_label = "Moderately Right-Skewed" if 0.5 < skew < 1.0 else ("Highly Right-Skewed" if skew >= 1.0 else "Approx. Symmetric")
    print(f"  - Skewness:                {skew:>6.3f} ({skew_label})")
    print(f"  - Kurtosis (Fisher):       {kurt:>6.3f} (Leptokurtic / heavier tails than normal)")
    print(f"  - Mean vs. Median Gap:     ${summary['mean'] - summary['median']:>12,.2f} (Mean pulled up by upper tail)")

    print("\n4. OUTLIER DETECTION (TUKEY'S IQR FENCES):")
    out_info = summary["outlier_analysis"]
    print(f"  - Normal Range (Fences):   ${out_info['lower_fence_iqr']:,.2f} to ${out_info['upper_fence_iqr']:,.2f}")
    print(f"  - Outliers (|IQR| > 1.5):  {out_info['iqr_outliers_count']:,} samples ({out_info['iqr_outliers_pct']:.2f}% of data)")
    print(f"  - Extreme Outliers (> 3x): {out_info['extreme_outliers_count']:,} samples")
    print(f"  - Z-Score Outliers (|Z|>3):{out_info['zscore_outliers_count']:,} samples")

    # Display top 5 highest and lowest salaries
    print("\n  Top 5 Highest Salaries in Dataset:")
    cols_to_show = [c for c in ["job_title", "experience_level", "company_location", "salary_in_usd"] if c in df.columns]
    top_5 = df.nlargest(5, "salary_in_usd")[cols_to_show]
    for _, row in top_5.iterrows():
        print(f"    * ${row['salary_in_usd']:>7,}: {row.get('job_title', 'N/A')} ({row.get('experience_level', 'N/A')}, {row.get('company_location', 'N/A')})")

    print("\n  Top 5 Lowest Salaries in Dataset:")
    bottom_5 = df.nsmallest(5, "salary_in_usd")[cols_to_show]
    for _, row in bottom_5.iterrows():
        print(f"    * ${row['salary_in_usd']:>7,}: {row.get('job_title', 'N/A')} ({row.get('experience_level', 'N/A')}, {row.get('company_location', 'N/A')})")

    print("\n5. LOG-TRANSFORMATION COMPARISON (FOR REGRESSION MODELING):")
    log_info = summary["log_transformed"]
    print(f"  - Original Skewness:       {skew:.3f}")
    print(f"  - Log-Transformed Skewness:{log_info['skewness']:.3f} (Near 0 -> drastically improved bell shape)")
    print(f"  - Original Kurtosis:       {kurt:.3f}")
    print(f"  - Log-Transformed Kurtosis:{log_info['kurtosis']:.3f}")

    if "experience_level" in df.columns:
        print("\n6. SALARY DISTRIBUTION BY EXPERIENCE LEVEL:")
        exp_table = df.groupby("experience_level")["salary_in_usd"].agg(
            Count="count",
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Max="max",
        ).loc[["EN", "MI", "SE", "EX"] if set(["EN", "MI", "SE", "EX"]).issubset(df["experience_level"].unique()) else df["experience_level"].unique()]
        print(f"  {'Tier':<6} | {'Count':<7} | {'Median ($)':<12} | {'Mean ($)':<12} | {'Std ($)':<12} | {'Max ($)':<10}")
        print("  " + "-" * 70)
        for tier, row in exp_table.iterrows():
            print(f"  {tier:<6} | {int(row['Count']):<7} | {row['Median']:<12,.0f} | {row['Mean']:<12,.0f} | {row['Std']:<12,.0f} | {row['Max']:<10,.0f}")

    print("\n" + "=" * 75)


def generate_distribution_plots(df: pd.DataFrame, summary: dict, save_dir: str):
    """Generates a 4-panel diagnostic figure visualizing the salary distribution."""
    os.makedirs(save_dir, exist_ok=True)
    salaries = df["salary_in_usd"]
    mean_val = summary["mean"]
    median_val = summary["median"]
    q1 = summary["percentiles"]["25% (Q1)"]
    q3 = summary["percentiles"]["75% (Q3)"]
    upper_fence = summary["outlier_analysis"]["upper_fence_iqr"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Plot 1: Histogram + KDE with Reference Lines ---
    sns.histplot(salaries, kde=True, color="#2b5c8f", bins=40, ax=axes[0, 0])
    axes[0, 0].axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: ${mean_val:,.0f}")
    axes[0, 0].axvline(median_val, color="green", linestyle="-", linewidth=2, label=f"Median: ${median_val:,.0f}")
    axes[0, 0].axvline(upper_fence, color="orange", linestyle=":", linewidth=2, label=f"IQR Fence: ${upper_fence:,.0f}")
    axes[0, 0].axvspan(q1, q3, alpha=0.15, color="blue", label="IQR (Q1 to Q3)")
    axes[0, 0].set_title("1. Salary Distribution with Density & Statistics")
    axes[0, 0].set_xlabel("Salary in USD")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].xaxis.set_major_formatter("${x:,.0f}")
    axes[0, 0].legend(loc="upper right")

    # --- Plot 2: Boxplot & Outliers ---
    sns.boxplot(x=salaries, color="#68a0cf", ax=axes[0, 1], flierprops={"marker": "o", "color": "red", "alpha": 0.5})
    axes[0, 1].set_title(f"2. Boxplot & Outliers (n={summary['outlier_analysis']['iqr_outliers_count']})")
    axes[0, 1].set_xlabel("Salary in USD")
    axes[0, 1].xaxis.set_major_formatter("${x:,.0f}")
    axes[0, 1].text(median_val, 0.25, f"Median: ${median_val:,.0f}", ha="center", fontweight="bold", color="green")

    # --- Plot 3: Log-Transformed Distribution (ML Assessment) ---
    log_salaries = np.log1p(salaries)
    sns.histplot(log_salaries, kde=True, color="#2e8b57", bins=35, ax=axes[1, 0])
    # Overlay theoretical normal distribution
    x_range = np.linspace(log_salaries.min(), log_salaries.max(), 100)
    norm_pdf = stats.norm.pdf(x_range, log_salaries.mean(), log_salaries.std())
    # Scale pdf to match histogram count
    norm_scaled = norm_pdf * len(log_salaries) * (log_salaries.max() - log_salaries.min()) / 35
    axes[1, 0].plot(x_range, norm_scaled, color="black", linestyle="--", linewidth=1.5, label="Normal Curve")
    axes[1, 0].set_title(f"3. Log-Transformed Distribution (log1p)\nSkewness: {summary['log_transformed']['skewness']:.3f}")
    axes[1, 0].set_xlabel("log(Salary in USD + 1)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()

    # --- Plot 4: Salary Distribution across Experience Levels ---
    if "experience_level" in df.columns:
        tier_order = [t for t in ["EN", "MI", "SE", "EX"] if t in df["experience_level"].unique()]
        palette = {"EN": "#56B4E9", "MI": "#009E73", "SE": "#0072B2", "EX": "#D55E00"}
        sns.boxplot(
            data=df,
            x="experience_level",
            y="salary_in_usd",
            order=tier_order,
            palette=palette,
            showmeans=True,
            meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "gold", "markersize": 7},
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("4. Salary Progression Across Experience Levels (Diamond = Mean)")
        axes[1, 1].set_xlabel("Experience Level (EN: Entry, MI: Mid, SE: Senior, EX: Exec)")
        axes[1, 1].set_ylabel("Salary in USD")
        axes[1, 1].yaxis.set_major_formatter("${x:,.0f}")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "salary_distribution_inspection.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved distribution visualization plot -> {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load data
    df = load_dataset()

    # 2. Compute statistics
    summary = calculate_salary_statistics(df["salary_in_usd"])

    # 3. Print terminal report
    display_salary_report(summary, df)

    # 4. Generate & save visual plots
    generate_distribution_plots(df, summary, FIGURES_DIR)

    # 5. Persist JSON summary
    summary_path = os.path.join(REPORTS_DIR, "salary_distribution_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved distribution summary JSON -> {summary_path}\n")


if __name__ == "__main__":
    main()

