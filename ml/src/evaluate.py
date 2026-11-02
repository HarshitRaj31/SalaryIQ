"""
Model Evaluation and Diagnostic Visualization Script for Job Salary Prediction.

Calculates comprehensive regression performance metrics (MAE, MedAE, RMSE, R², MAPE)
and generates diagnostic visualizations:
1. Model Performance Benchmark Comparison
2. Actual vs. Predicted Salaries with reference line
3. Residuals Diagnostics (Residuals vs Predicted & Residuals Distribution)
4. Feature Importance Analysis (Top predictive salary drivers)
"""

import json
import os
import sys
import tempfile
import warnings

# Direct matplotlib cache to temp directory to avoid sandbox permission errors
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ml/ directory
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_salary.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Visual formatting settings
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11


def load_test_data():
    """Loads cleaned dataset and generates the consistent test split."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at '{DATA_PATH}'.")

    df = pd.read_csv(DATA_PATH)
    target_col = "salary_in_usd"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Matching the exact split from train.py (test_size=0.2, random_state=42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Loaded test dataset: {X_test.shape[0]} samples with {X_test.shape[1]} features.")
    return X_test, y_test


def load_available_models():
    """Loads all trained pipeline models from ml/models/."""
    model_files = {
        "Linear Regression": os.path.join(MODELS_DIR, "linear_regression.pkl"),
        "Random Forest Regressor": os.path.join(MODELS_DIR, "random_forest_regressor.pkl"),
        "Gradient Boosting Regressor": os.path.join(MODELS_DIR, "gradient_boosting_regressor.pkl"),
    }

    loaded_models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
            print(f"Loaded model pipeline: {name}")
        else:
            print(f"Notice: Model file not found for {name} at {path}")

    # Fallback to best_model.pkl if individual models missing
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not loaded_models and os.path.exists(best_path):
        loaded_models["Best Model"] = joblib.load(best_path)
        print("Loaded 'Best Model' from best_model.pkl")

    if not loaded_models:
        raise FileNotFoundError(f"No trained models found in '{MODELS_DIR}'. Run 'train.py' first.")

    return loaded_models


def compute_metrics(y_true, y_pred):
    """Calculates comprehensive evaluation metrics."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2_Score": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "Explained_Variance": float(explained_variance_score(y_true, y_pred)),
    }


def evaluate_all_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates each model and collects predictions and metrics."""
    metrics_report = {}
    predictions = {}

    print("\n" + "=" * 70)
    print("DETAILED PERFORMANCE EVALUATION ON TEST SET")
    print("=" * 70)

    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred
        metrics = compute_metrics(y_test, y_pred)
        metrics_report[name] = metrics

        print(f"\nModel: {name}")
        print(f"  - Mean Absolute Error (MAE):    ${metrics['MAE']:>10,.2f}")
        print(f"  - Median Absolute Error (MedAE):${metrics['MedAE']:>10,.2f}")
        print(f"  - Root Mean Squared Error (RMSE):${metrics['RMSE']:>10,.2f}")
        print(f"  - R^2 Score:                    {metrics['R2_Score']:>10.4f}")
        print(f"  - Mean Abs % Error (MAPE):      {metrics['MAPE'] * 100:>10.2f}%")
        print(f"  - Explained Variance:           {metrics['Explained_Variance']:>10.4f}")

    return metrics_report, predictions


def plot_metrics_comparison(metrics_report: dict, save_dir: str):
    """Generates comparison bar chart of key evaluation metrics."""
    df_metrics = pd.DataFrame(metrics_report).T.reset_index()
    df_metrics.rename(columns={"index": "Model"}, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. MAE Comparison
    sns.barplot(data=df_metrics, x="Model", y="MAE", palette="Blues_d", ax=axes[0])
    axes[0].set_title("Mean Absolute Error (Lower is Better)")
    axes[0].set_ylabel("MAE ($)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15, ha="right")
    axes[0].yaxis.set_major_formatter("${x:,.0f}")
    for p in axes[0].patches:
        axes[0].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. RMSE Comparison
    sns.barplot(data=df_metrics, x="Model", y="RMSE", palette="Reds_d", ax=axes[1])
    axes[1].set_title("Root Mean Squared Error (Lower is Better)")
    axes[1].set_ylabel("RMSE ($)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")
    axes[1].yaxis.set_major_formatter("${x:,.0f}")
    for p in axes[1].patches:
        axes[1].annotate(f"${p.get_height():,.0f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. R2 Score Comparison
    sns.barplot(data=df_metrics, x="Model", y="R2_Score", palette="Greens_d", ax=axes[2])
    axes[2].set_title("R^2 Score (Higher is Better)")
    axes[2].set_ylabel("R^2 Score")
    axes[2].set_ylim(0, max(0.5, df_metrics["R2_Score"].max() * 1.25))
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=15, ha="right")
    for p in axes[2].patches:
        axes[2].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                         ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "model_comparison_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved metric comparison plot -> {output_path}")


def plot_actual_vs_predicted(y_test, predictions: dict, metrics_report: dict, save_dir: str):
    """Generates Actual vs. Predicted scatter plots with diagonal reference line."""
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    min_val = min(y_test.min(), min(preds.min() for preds in predictions.values()))
    max_val = max(y_test.max(), max(preds.max() for preds in predictions.values()))

    colors = ["#2b5c8f", "#2e8b57", "#d95f02"]

    for i, (name, y_pred) in enumerate(predictions.items()):
        ax = axes[i]
        color = colors[i % len(colors)]

        # Scatter
        ax.scatter(y_test, y_pred, alpha=0.55, edgecolors="none", color=color, s=35, label="Predictions")

        # Ideal y = x line
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

        r2 = metrics_report[name]["R2_Score"]
        rmse = metrics_report[name]["RMSE"]
        mae = metrics_report[name]["MAE"]

        ax.set_title(f"{name}\nR^2 = {r2:.4f} | RMSE = ${rmse:,.0f}")
        ax.set_xlabel("Actual Salary ($)")
        if i == 0:
            ax.set_ylabel("Predicted Salary ($)")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        ax.legend(loc="upper left")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "actual_vs_predicted.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved actual vs. predicted plot -> {output_path}")


def plot_residuals_diagnostics(y_test, predictions: dict, save_dir: str):
    """Generates Residuals vs Predicted and Residuals Distribution plots."""
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))

    if n_models == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    colors = ["#2b5c8f", "#2e8b57", "#d95f02"]

    for i, (name, y_pred) in enumerate(predictions.items()):
        residuals = y_test - y_pred
        color = colors[i % len(colors)]

        # Top row: Residuals vs Predicted
        ax_top = axes[0, i]
        ax_top.scatter(y_pred, residuals, alpha=0.55, color=color, s=30)
        ax_top.axhline(0, color="red", linestyle="--", linewidth=2)
        ax_top.set_title(f"{name}\nResiduals vs. Predicted")
        ax_top.set_xlabel("Predicted Salary ($)")
        if i == 0:
            ax_top.set_ylabel("Residuals ($)")
        ax_top.xaxis.set_major_formatter("${x:,.0f}")
        ax_top.yaxis.set_major_formatter("${x:,.0f}")

        # Bottom row: Residuals Distribution
        ax_bottom = axes[1, i]
        sns.histplot(residuals, kde=True, color=color, bins=30, ax=ax_bottom)
        ax_bottom.axvline(0, color="red", linestyle="--", linewidth=2)
        ax_bottom.set_title(f"Residuals Distribution\n(Mean: ${residuals.mean():,.0f})")
        ax_bottom.set_xlabel("Residual Error ($)")
        if i == 0:
            ax_bottom.set_ylabel("Frequency")
        ax_bottom.xaxis.set_major_formatter("${x:,.0f}")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "residuals_diagnostics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residuals diagnostics plot -> {output_path}")


def plot_feature_importance(models: dict, save_dir: str, top_n: int = 15):
    """Extracts and visualizes feature importances for tree-based models."""
    tree_models = {}
    for name, pipeline in models.items():
        regressor = pipeline.named_steps.get("regressor")
        if hasattr(regressor, "feature_importances_"):
            tree_models[name] = pipeline

    if not tree_models:
        print("No tree-based models with feature_importances_ found to plot.")
        return

    fig, axes = plt.subplots(len(tree_models), 1, figsize=(12, 6 * len(tree_models)))
    if len(tree_models) == 1:
        axes = [axes]

    for i, (name, pipeline) in enumerate(tree_models.items()):
        ax = axes[i]
        preprocessor = pipeline.named_steps["preprocessor"]
        regressor = pipeline.named_steps["regressor"]

        raw_features = preprocessor.get_feature_names_out()
        # Clean feature names for clear visualization
        clean_features = [
            f.replace("cat__", "").replace("num__", "").replace("_", " ")
            for f in raw_features
        ]

        importances = regressor.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": clean_features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_n)

        sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax)
        ax.set_title(f"Top {top_n} Predictive Features - {name}")
        ax.set_xlabel("Relative Importance")
        ax.set_ylabel("Feature Name")

        for p in ax.patches:
            val = p.get_width()
            ax.annotate(f"{val:.3f} ({val * 100:.1f}%)",
                        (val + 0.005, p.get_y() + p.get_height() / 2.0),
                        va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = os.path.join(save_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot -> {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load Data
    X_test, y_test = load_test_data()

    # 2. Load Models
    models = load_available_models()

    # 3. Evaluate Performance
    metrics_report, predictions = evaluate_all_models(models, X_test, y_test)

    # 4. Save JSON Report
    metrics_json_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
    print(f"\nSaved metrics summary -> {metrics_json_path}")

    # 5. Visualizations
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC VISUALIZATION PLOTS")
    print("=" * 70)
    plot_metrics_comparison(metrics_report, FIGURES_DIR)
    plot_actual_vs_predicted(y_test, predictions, metrics_report, FIGURES_DIR)
    plot_residuals_diagnostics(y_test, predictions, FIGURES_DIR)
    plot_feature_importance(models, FIGURES_DIR)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND VISUALIZATIONS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

