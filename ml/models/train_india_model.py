"""
Job Salary Prediction System - India Tech Jobs ML Training Pipeline
File: train_india_model.py

Features:
- Automatic detection of the salary target column (e.g., salary_in_lpa, salary_midpoint_lpa, salary_in_inr, salary_in_usd).
- Removal of salary leakage and identifier columns (e.g., salary_in_usd, salary_in_inr, job_id, job_url, etc.).
- Missing-value imputation (SimpleImputer: median for numerical, mode for categorical).
- Categorical encoding (OneHotEncoder with handle_unknown='ignore').
- Feature scaling for Linear Regression (StandardScaler on numerical features).
- 80/20 train-test split (random_state=42).
- 5-fold cross-validation (KFold with shuffle=True, random_state=42).
- Comparison of all three models:
    1. Linear Regression (Baseline with feature scaling)
    2. Random Forest Regressor (with hyperparameter tuning)
    3. Gradient Boosting Regressor (with hyperparameter tuning)
- Saving the best model and its metadata (pkl pipelines, JSON comparison, and JSON feature metadata).
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrainIndiaModel")

# -----------------------------------------------------------------------------
# DEFAULT PATH RESOLUTION
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name in ["models", "src"] else SCRIPT_DIR
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


def find_default_dataset() -> Path:
    """
    Locates the processed Indian Tech Jobs dataset in standard locations.
    Prioritizes 'india_salary_model_ready.csv', falling back to 'cleaned_indian_tech_jobs.csv'.
    """
    candidate_paths = [
        DATA_DIR / "india_salary_model_ready.csv",
        DATA_DIR / "cleaned_indian_tech_jobs.csv",
        Path.cwd() / "Job-Salary-Prediction" / "ml" / "data" / "processed" / "india_salary_model_ready.csv",
        Path.cwd() / "Job-Salary-Prediction" / "ml" / "data" / "processed" / "cleaned_indian_tech_jobs.csv",
        Path.cwd() / "ml" / "data" / "processed" / "india_salary_model_ready.csv",
        Path.cwd() / "ml" / "data" / "processed" / "cleaned_indian_tech_jobs.csv",
    ]

    for p in candidate_paths:
        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError(
        f"Could not find processed dataset in any standard location. Checked: {[str(p) for p in candidate_paths]}"
    )


# -----------------------------------------------------------------------------
# 1. AUTOMATIC DETECTION OF SALARY TARGET COLUMN
# -----------------------------------------------------------------------------
PRIORITY_TARGET_CANDIDATES = [
    "salary_in_lpa",
    "salary_midpoint_lpa",
    "salary_in_inr",
    "salary_in_usd",
    "salary",
    "ctc_in_lpa",
    "ctc",
]


def detect_target_column(df: pd.DataFrame, user_override: Optional[str] = None) -> Tuple[str, str]:
    """
    Automatically detects the primary salary target column from candidate names.
    Returns (target_column_name, target_unit).
    """
    if user_override and user_override in df.columns:
        logger.info(f"[TARGET DETECTION] Using user-specified target column: '{user_override}'")
        target_col = user_override
    else:
        # Check priority candidate list
        target_col = None
        for candidate in PRIORITY_TARGET_CANDIDATES:
            if candidate in df.columns:
                target_col = candidate
                break

        # Fallback heuristic: search any column matching 'salary' or 'target'
        if target_col is None:
            matches = [
                c for c in df.columns
                if "salary" in c.lower() or "ctc" in c.lower() or "target" in c.lower()
            ]
            if matches:
                target_col = matches[0]

        if target_col is None:
            raise ValueError(
                f"Automatic target detection failed. No recognized salary column found in {df.columns.tolist()}."
            )

    # Determine display unit
    col_lower = target_col.lower()
    if "lpa" in col_lower:
        unit = "LPA"
    elif "inr" in col_lower:
        unit = "INR"
    elif "usd" in col_lower:
        unit = "USD"
    else:
        unit = "Currency"

    # Validate target column numeric conversion
    y_raw = pd.to_numeric(df[target_col], errors="coerce")
    valid_count = y_raw.notnull().sum()
    if valid_count == 0:
        raise ValueError(f"Target column '{target_col}' contains no valid numeric values.")

    target_min = y_raw.min()
    target_max = y_raw.max()
    target_median = y_raw.median()
    target_mean = y_raw.mean()

    logger.info(
        f"[TARGET DETECTION] Successfully detected salary target: '{target_col}' ({unit}) | "
        f"Valid rows: {valid_count:,} | Min: {target_min:.2f} {unit} | "
        f"Median: {target_median:.2f} {unit} | Mean: {target_mean:.2f} {unit} | Max: {target_max:.2f} {unit}"
    )

    return target_col, unit


# -----------------------------------------------------------------------------
# 2. REMOVAL OF SALARY LEAKAGE AND IDENTIFIER COLUMNS
# -----------------------------------------------------------------------------
KNOWN_SALARY_LEAKAGE_NAMES = [
    "salary_in_lpa",
    "salary_midpoint_lpa",
    "salary_min_lpa",
    "salary_max_lpa",
    "salary_in_inr",
    "formatted_salary_inr",
    "salary_in_usd",
    "formatted_salary_usd",
    "salary_raw",
    "salary_tier",
    "salary_disclosed",
    "salary_negotiable",
    "ctc",
    "ctc_in_lpa",
    "stipend",
    "compensation",
    "pay",
]

KNOWN_IDENTIFIERS_AND_METADATA = [
    "job_id",
    "id",
    "job_url",
    "url",
    "link",
    "posted_date_raw",
    "posted_date",
    "days_since_posted",
    "scraped_at",
    "data_source",
    "timestamp",
    "job_description",
    "description",
    "skills_required",
    # Redundant / raw high-cardinality columns replaced by standardized features
    "company_name",
    "location",
    "job_title",  # Standardized as 'role_standard'
    "experience_level_label",  # Standardized as 'experience_level'
    "company_size_label",  # Standardized as 'company_size'
    "work_mode_clean",  # Standardized as 'remote_ratio'
    "experience_min_yrs",  # Summarized in 'experience_avg_yrs'
    "experience_max_yrs",  # Summarized in 'experience_avg_yrs'
    "company_location",  # Redundant with primary_city_clean
    "employee_residence",  # Redundant with primary_city_clean
]


def remove_leakage_and_identifiers(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Identifies and purges salary leakage columns, identifiers, and unstructured text columns.
    Returns (X, y, leakage_columns_dropped, identifier_columns_dropped).
    """
    logger.info("-" * 75)
    logger.info("[AUDIT] Auditing features for salary leakage and non-predictive identifiers...")

    # Extract target series and drop NaN targets
    df_clean = df.dropna(subset=[target_col]).copy()
    y = pd.to_numeric(df_clean[target_col], errors="coerce")
    valid_mask = y.notnull()
    df_clean = df_clean[valid_mask]
    y = y[valid_mask]

    leakage_dropped = []
    identifiers_dropped = []

    cols_to_evaluate = [col for col in df_clean.columns if col != target_col]

    for col in cols_to_evaluate:
        col_lower = col.lower().strip()

        # Check salary leakage
        is_leakage = (
            col_lower in KNOWN_SALARY_LEAKAGE_NAMES
            or "salary" in col_lower
            or ("ctc" in col_lower and len(col_lower) <= 10)
            or ("stipend" in col_lower)
            or ("pay" in col_lower and "payment" not in col_lower)
            or (col_lower in ["salary_in_usd", "salary_in_inr", "salary_in_lpa"])
        )

        if is_leakage:
            leakage_dropped.append(col)
            logger.info(f"  [SALARY LEAKAGE REMOVED] Dropped '{col}' to prevent data leakage.")
            continue

        # Check identifiers and non-predictive metadata
        is_identifier = (
            col_lower in KNOWN_IDENTIFIERS_AND_METADATA
            or col_lower.endswith("_id")
            or col_lower.startswith("id_")
            or col_lower in ["job_id", "url", "job_url"]
            or "posted_date" in col_lower
            or "scraped" in col_lower
            or "description" in col_lower
        )

        # High cardinality text / unique identifier check (if all values unique and non-numeric)
        if not is_identifier and df_clean[col].dtype == object:
            if df_clean[col].nunique() >= len(df_clean) * 0.95 and len(df_clean) > 50:
                is_identifier = True

        if is_identifier:
            identifiers_dropped.append(col)
            logger.info(f"  [IDENTIFIER REMOVED] Dropped '{col}' (non-predictive/identifier/redundant).")
            continue

    all_dropped = set(leakage_dropped + identifiers_dropped + [target_col])
    remaining_features = [col for col in df_clean.columns if col not in all_dropped]

    X = df_clean[remaining_features].copy()

    logger.info(
        f"[DATA PREPARATION] Total features retained for modeling: {X.shape[1]} | "
        f"Leakage columns removed: {len(leakage_dropped)} | "
        f"Identifiers/Metadata removed: {len(identifiers_dropped)}"
    )
    logger.info(f"[FEATURES RETAINED]: {list(X.columns)}")

    return X, y, leakage_dropped, identifiers_dropped


# -----------------------------------------------------------------------------
# 3. FEATURE TYPE DETECTION & PREPROCESSING PIPELINES
# -----------------------------------------------------------------------------
def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separates categorical and numerical features for targeted preprocessing.
    """
    categorical_features = []
    numerical_features = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numerical_features.append(col)
        else:
            categorical_features.append(col)

    logger.info(f"[FEATURE PARTITION] Categorical ({len(categorical_features)}): {categorical_features}")
    logger.info(f"[FEATURE PARTITION] Numerical ({len(numerical_features)}): {numerical_features}")

    return categorical_features, numerical_features


def build_preprocessors(
    categorical_features: List[str],
    numerical_features: List[str],
) -> Tuple[ColumnTransformer, ColumnTransformer]:
    """
    Constructs dedicated preprocessing pipelines:
    1. linear_preprocessor:
       - Numerical: SimpleImputer(strategy='median') -> StandardScaler()
       - Categorical: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')
    2. tree_preprocessor:
       - Numerical: SimpleImputer(strategy='median') (no scaling needed for tree algorithms)
       - Categorical: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')
    """
    # Categorical encoder pipeline (shared across models)
    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Numerical pipeline WITH feature scaling (for Linear Regression)
    num_transformer_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Numerical pipeline WITHOUT feature scaling (for tree-based models: RF & GB)
    num_transformer_unscaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Linear Regression Preprocessor (with StandardScaler)
    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer_scaled, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Tree-based Preprocessor (Preserving natural feature scale)
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer_unscaled, numerical_features),
            ("cat", cat_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return linear_preprocessor, tree_preprocessor


# -----------------------------------------------------------------------------
# 4. EVALUATION UTILITIES
# -----------------------------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates standard regression evaluation metrics: MAE, RMSE, R2, MAPE.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100.0
    except Exception:
        mape = 0.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2,
        "MAPE": mape,
    }


def evaluate_5fold_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: KFold,
) -> Dict[str, float]:
    """
    Performs 5-fold cross-validation on the training set and computes
    CV RMSE, CV MAE, and CV R2 (mean and std).
    """
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    cv_res = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    cv_rmse_mean = float(-cv_res["test_rmse"].mean())
    cv_rmse_std = float(cv_res["test_rmse"].std())
    cv_mae_mean = float(-cv_res["test_mae"].mean())
    cv_mae_std = float(cv_res["test_mae"].std())
    cv_r2_mean = float(cv_res["test_r2"].mean())
    cv_r2_std = float(cv_res["test_r2"].std())

    return {
        "CV_RMSE": cv_rmse_mean,
        "CV_RMSE_STD": cv_rmse_std,
        "CV_MAE": cv_mae_mean,
        "CV_MAE_STD": cv_mae_std,
        "CV_R2": cv_r2_mean,
        "CV_R2_STD": cv_r2_std,
    }


# -----------------------------------------------------------------------------
# 5. MODEL TRAINING & COMPARISON PIPELINE
# -----------------------------------------------------------------------------
def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    linear_preprocessor: ColumnTransformer,
    tree_preprocessor: ColumnTransformer,
    cv: KFold,
    tune_hyperparameters: bool = True,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], str]:
    """
    Trains and compares all 3 models:
    1. Linear Regression (Baseline with StandardScaler)
    2. Random Forest Regressor (with 5-fold CV & hyperparameter optimization)
    3. Gradient Boosting Regressor (with 5-fold CV & hyperparameter optimization)
    """
    results: Dict[str, Dict[str, Any]] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    # =========================================================================
    # MODEL 1: LINEAR REGRESSION (BASELINE)
    # =========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("MODEL 1 / 3: LINEAR REGRESSION (BASELINE WITH FEATURE SCALING)")
    logger.info("=" * 75)

    linear_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(linear_preprocessor)),
            ("regressor", LinearRegression()),
        ]
    )

    logger.info("Running 5-fold cross-validation for Linear Regression...")
    linear_cv = evaluate_5fold_cv(linear_pipe, X_train, y_train, cv)

    # Fit on full training set and evaluate on test set
    linear_pipe.fit(X_train, y_train)
    linear_test_preds = linear_pipe.predict(X_test)
    linear_test_metrics = calculate_metrics(y_test, linear_test_preds)

    results["Linear Regression"] = {
        **linear_cv,
        **linear_test_metrics,
        "Hyperparameters": {"fit_intercept": True},
    }
    fitted_pipelines["Linear Regression"] = linear_pipe

    logger.info(
        f"Linear Regression Results -> "
        f"5-Fold CV RMSE: {linear_cv['CV_RMSE']:.3f} (+/- {linear_cv['CV_RMSE_STD']:.3f}) | "
        f"5-Fold CV MAE: {linear_cv['CV_MAE']:.3f} | "
        f"Test MAE: {linear_test_metrics['MAE']:.3f} | "
        f"Test RMSE: {linear_test_metrics['RMSE']:.3f} | "
        f"Test R2: {linear_test_metrics['R2_Score']:.4f}"
    )

    # =========================================================================
    # MODEL 2: RANDOM FOREST REGRESSOR
    # =========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("MODEL 2 / 3: RANDOM FOREST REGRESSOR (5-FOLD CV & TUNING)")
    logger.info("=" * 75)

    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(tree_preprocessor)),
            ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    if tune_hyperparameters:
        logger.info("Tuning Random Forest hyperparameters via RandomizedSearchCV (5-fold CV)...")
        rf_param_distributions = {
            "regressor__n_estimators": [100, 150, 200, 250],
            "regressor__max_depth": [None, 8, 12, 16, 20],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__max_features": ["sqrt", 0.7, 1.0],
        }

        rf_search = RandomizedSearchCV(
            estimator=rf_pipe,
            param_distributions=rf_param_distributions,
            n_iter=15,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        rf_search.fit(X_train, y_train)
        best_rf_pipe = rf_search.best_estimator_
        rf_best_params = rf_search.best_params_
        logger.info(f"Best Random Forest Hyperparameters: {rf_best_params}")
    else:
        best_rf_pipe = rf_pipe
        best_rf_pipe.fit(X_train, y_train)
        rf_best_params = {"n_estimators": 100, "random_state": 42}

    logger.info("Evaluating 5-fold CV for best Random Forest model...")
    rf_cv = evaluate_5fold_cv(best_rf_pipe, X_train, y_train, cv)

    rf_test_preds = best_rf_pipe.predict(X_test)
    rf_test_metrics = calculate_metrics(y_test, rf_test_preds)

    results["Random Forest Regressor"] = {
        **rf_cv,
        **rf_test_metrics,
        "Hyperparameters": rf_best_params,
    }
    fitted_pipelines["Random Forest Regressor"] = best_rf_pipe

    logger.info(
        f"Random Forest Results -> "
        f"5-Fold CV RMSE: {rf_cv['CV_RMSE']:.3f} (+/- {rf_cv['CV_RMSE_STD']:.3f}) | "
        f"5-Fold CV MAE: {rf_cv['CV_MAE']:.3f} | "
        f"Test MAE: {rf_test_metrics['MAE']:.3f} | "
        f"Test RMSE: {rf_test_metrics['RMSE']:.3f} | "
        f"Test R2: {rf_test_metrics['R2_Score']:.4f}"
    )

    # =========================================================================
    # MODEL 3: GRADIENT BOOSTING REGRESSOR
    # =========================================================================
    logger.info("\n" + "=" * 75)
    logger.info("MODEL 3 / 3: GRADIENT BOOSTING REGRESSOR (5-FOLD CV & TUNING)")
    logger.info("=" * 75)

    gb_pipe = Pipeline(
        steps=[
            ("preprocessor", clone(tree_preprocessor)),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]
    )

    if tune_hyperparameters:
        logger.info("Tuning Gradient Boosting hyperparameters via RandomizedSearchCV (5-fold CV)...")
        gb_param_distributions = {
            "regressor__n_estimators": [100, 150, 200, 250],
            "regressor__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
            "regressor__max_depth": [3, 4, 5, 6],
            "regressor__min_samples_split": [2, 4, 6, 8],
            "regressor__min_samples_leaf": [1, 2, 4],
            "regressor__subsample": [0.75, 0.85, 0.95, 1.0],
        }

        gb_search = RandomizedSearchCV(
            estimator=gb_pipe,
            param_distributions=gb_param_distributions,
            n_iter=15,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gb_search.fit(X_train, y_train)
        best_gb_pipe = gb_search.best_estimator_
        gb_best_params = gb_search.best_params_
        logger.info(f"Best Gradient Boosting Hyperparameters: {gb_best_params}")
    else:
        best_gb_pipe = gb_pipe
        best_gb_pipe.fit(X_train, y_train)
        gb_best_params = {"n_estimators": 100, "random_state": 42}

    logger.info("Evaluating 5-fold CV for best Gradient Boosting model...")
    gb_cv = evaluate_5fold_cv(best_gb_pipe, X_train, y_train, cv)

    gb_test_preds = best_gb_pipe.predict(X_test)
    gb_test_metrics = calculate_metrics(y_test, gb_test_preds)

    results["Gradient Boosting Regressor"] = {
        **gb_cv,
        **gb_test_metrics,
        "Hyperparameters": gb_best_params,
    }
    fitted_pipelines["Gradient Boosting Regressor"] = best_gb_pipe

    logger.info(
        f"Gradient Boosting Results -> "
        f"5-Fold CV RMSE: {gb_cv['CV_RMSE']:.3f} (+/- {gb_cv['CV_RMSE_STD']:.3f}) | "
        f"5-Fold CV MAE: {gb_cv['CV_MAE']:.3f} | "
        f"Test MAE: {gb_test_metrics['MAE']:.3f} | "
        f"Test RMSE: {gb_test_metrics['RMSE']:.3f} | "
        f"Test R2: {gb_test_metrics['R2_Score']:.4f}"
    )

    # Determine Best Model based on lowest 5-Fold CV RMSE
    best_model_name = min(results.keys(), key=lambda m: results[m]["CV_RMSE"])

    return fitted_pipelines, results, best_model_name


# -----------------------------------------------------------------------------
# 6. COMPARISON REPORT & DISPLAY
# -----------------------------------------------------------------------------
def display_model_comparison(results: Dict[str, Dict[str, Any]], best_model_name: str, unit: str):
    """
    Renders an ASCII comparison table ranking all three models across CV and Test metrics.
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            f"CV RMSE ({unit})": f"{metrics['CV_RMSE']:.3f} +/- {metrics['CV_RMSE_STD']:.2f}",
            f"CV MAE ({unit})": f"{metrics['CV_MAE']:.3f}",
            "CV R2": f"{metrics['CV_R2']:.4f}",
            f"Test MAE ({unit})": f"{metrics['MAE']:.3f}",
            f"Test RMSE ({unit})": f"{metrics['RMSE']:.3f}",
            "Test R2": f"{metrics['R2_Score']:.4f}",
        })

    df_comp = pd.DataFrame(rows)
    df_comp.sort_values(by=f"Test R2", ascending=False, inplace=True)

    print("\n" + "=" * 90)
    print("                 MODEL EVALUATION & 5-FOLD CV COMPARISON TABLE")
    print("=" * 90)
    print(df_comp.to_string(index=False))
    print("=" * 90)
    print(f" WINNING MODEL: {best_model_name}")
    print(f" -> Best 5-Fold Cross-Validation RMSE: {results[best_model_name]['CV_RMSE']:.3f} {unit}")
    print(f" -> Test Set MAE: {results[best_model_name]['MAE']:.3f} {unit}")
    print(f" -> Test Set RMSE: {results[best_model_name]['RMSE']:.3f} {unit}")
    print(f" -> Test Set R2 Score: {results[best_model_name]['R2_Score']:.4f}")
    print("=" * 90 + "\n")


# -----------------------------------------------------------------------------
# 7. ARTIFACT PERSISTENCE
# -----------------------------------------------------------------------------
def save_artifacts(
    fitted_pipelines: Dict[str, Pipeline],
    best_model_name: str,
    results: Dict[str, Dict[str, Any]],
    target_col: str,
    unit: str,
    categorical_features: List[str],
    numerical_features: List[str],
    leakage_dropped: List[str],
    identifiers_dropped: List[str],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
):
    """
    Saves trained models, comparison report, and feature metadata to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[PERSISTENCE] Saving model artifacts to: {output_dir}")

    # 1. Save individual model pipelines
    filename_map = {
        "Linear Regression": "linear_regression_india.pkl",
        "Random Forest Regressor": "random_forest_india.pkl",
        "Gradient Boosting Regressor": "gradient_boosting_india.pkl",
    }

    for name, pipeline in fitted_pipelines.items():
        fname = filename_map.get(name, f"{name.lower().replace(' ', '_')}_india.pkl")
        save_path = output_dir / fname
        joblib.dump(pipeline, save_path)
        logger.info(f"  [SAVED] {name} -> {save_path.name}")

    # 2. Save best model as best_india_model.pkl
    best_model_path = output_dir / "best_india_model.pkl"
    joblib.dump(fitted_pipelines[best_model_name], best_model_path)
    logger.info(f"  [SAVED BEST MODEL] {best_model_name} -> {best_model_path.name}")

    # 3. Save model comparison results JSON
    comparison_payload = {
        "best_model": best_model_name,
        "selection_metric": "5-fold CV RMSE",
        "target": target_col,
        "target_unit": unit,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_records": len(X),
        "metrics": results,
    }
    comparison_path = output_dir / "india_model_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=4)
    logger.info(f"  [SAVED COMPARISON JSON] -> {comparison_path.name}")

    # 4. Save feature metadata JSON
    categories_dict = {}
    for col in categorical_features:
        unique_vals = X[col].dropna().unique().tolist()
        try:
            unique_vals = sorted(unique_vals)
        except Exception:
            pass
        categories_dict[col] = unique_vals

    numerical_ranges = {}
    for col in numerical_features:
        series = pd.to_numeric(X[col], errors="coerce").dropna()
        numerical_ranges[col] = {
            "min": float(series.min()) if len(series) > 0 else 0.0,
            "max": float(series.max()) if len(series) > 0 else 0.0,
            "median": float(series.median()) if len(series) > 0 else 0.0,
            "mean": float(series.mean()) if len(series) > 0 else 0.0,
            "std": float(series.std()) if len(series) > 0 else 0.0,
        }

    target_stats = {
        "min": float(y.min()),
        "25%": float(y.quantile(0.25)),
        "50%": float(y.median()),
        "75%": float(y.quantile(0.75)),
        "max": float(y.max()),
        "mean": float(y.mean()),
        "std": float(y.std()),
    }

    feature_metadata = {
        "target": target_col,
        "target_unit": unit,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "all_input_features": list(X.columns),
        "categories": categories_dict,
        "numerical_ranges": numerical_ranges,
        "target_stats": target_stats,
        "leakage_columns_dropped": leakage_dropped,
        "identifier_columns_dropped": identifiers_dropped,
        "best_model": best_model_name,
        "best_model_metrics": results[best_model_name],
        "created_at": datetime.datetime.now().isoformat(),
    }

    metadata_path = output_dir / "india_feature_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=4)
    logger.info(f"  [SAVED FEATURE METADATA] -> {metadata_path.name}")


# -----------------------------------------------------------------------------
# 8. VERIFICATION / SMOKE TEST
# -----------------------------------------------------------------------------
def verify_saved_model(output_dir: Path, target_col: str, unit: str):
    """
    Loads the saved best model and verifies it performs an end-to-end inference test.
    """
    best_model_path = output_dir / "best_india_model.pkl"
    metadata_path = output_dir / "india_feature_metadata.json"

    if not best_model_path.exists() or not metadata_path.exists():
        logger.warning("Verification skipped: model or metadata file not found.")
        return

    pipeline = joblib.load(best_model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Build representative test sample
    sample_data: Dict[str, Any] = {}
    for col in meta["categorical_features"]:
        cats = meta["categories"].get(col, [])
        sample_data[col] = cats[0] if cats else "Unknown"

    for col in meta["numerical_features"]:
        stat = meta["numerical_ranges"].get(col, {})
        sample_data[col] = stat.get("median", 0.0)

    # Specific realistic values if matching column names
    if "role_standard" in sample_data:
        sample_data["role_standard"] = "Data Scientist"
    if "experience_level" in sample_data:
        sample_data["experience_level"] = "SE"
    if "primary_city_clean" in sample_data:
        sample_data["primary_city_clean"] = "Bangalore"
    if "company_size" in sample_data:
        sample_data["company_size"] = "L"
    if "experience_avg_yrs" in sample_data:
        sample_data["experience_avg_yrs"] = 5.0
    if "company_rating" in sample_data:
        sample_data["company_rating"] = 4.2
    if "skills_count" in sample_data:
        sample_data["skills_count"] = 6
    if "remote_ratio" in sample_data:
        sample_data["remote_ratio"] = 50
    if "work_year" in sample_data:
        sample_data["work_year"] = 2026

    df_sample = pd.DataFrame([sample_data])
    predicted_val = float(pipeline.predict(df_sample)[0])

    logger.info("\n" + "=" * 75)
    logger.info("VERIFICATION & INFERENCE TEST ON SAVED BEST MODEL")
    logger.info("=" * 75)
    logger.info(f"Sample Input: {sample_data}")
    logger.info(f"Predicted Salary: {predicted_val:.2f} {unit}")
    logger.info("=" * 75)


# -----------------------------------------------------------------------------
# MAIN CLI ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train, Evaluate, and Compare India Tech Jobs Salary Prediction Models."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to processed CSV dataset (default: auto-detected in ml/data/processed/).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Salary target column override (default: auto-detected, e.g. salary_in_lpa).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained models and metadata (default: ml/models/).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip RandomizedSearchCV hyperparameter search to speed up training.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    # Determine input data path
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"Specified dataset does not exist: {data_path}")
    else:
        data_path = find_default_dataset()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR

    logger.info("=" * 75)
    logger.info("JOB SALARY PREDICTION - INDIA TECH JOBS MODEL TRAINING")
    logger.info("=" * 75)
    logger.info(f"Dataset path: {data_path.resolve()}")
    logger.info(f"Models directory: {output_dir.resolve()}")
    logger.info(f"Random seed: {args.random_state}")

    # 1. Load dataset
    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Automatic detection of salary target
    target_col, unit = detect_target_column(df, user_override=args.target)

    # 3. Purge salary leakage and identifier columns
    X, y, leakage_dropped, identifiers_dropped = remove_leakage_and_identifiers(df, target_col)

    # 4. Feature type detection and Preprocessing Pipelines
    categorical_features, numerical_features = identify_feature_types(X)
    linear_preprocessor, tree_preprocessor = build_preprocessors(categorical_features, numerical_features)

    # 5. 80/20 Train-Test Split
    logger.info("\n" + "-" * 75)
    logger.info("[SPLIT] Executing 80/20 Train-Test split (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=args.random_state
    )
    logger.info(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
    logger.info(f"  Testing set:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})")

    # 6. 5-Fold Cross-Validation Setup
    cv = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
    logger.info(f"[CROSS-VALIDATION] 5-Fold KFold CV configured (shuffle=True, random_state={args.random_state}).")

    # 7. Train and compare models
    fitted_pipelines, results, best_model_name = train_and_compare_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        linear_preprocessor=linear_preprocessor,
        tree_preprocessor=tree_preprocessor,
        cv=cv,
        tune_hyperparameters=(not args.no_tune),
    )

    # 8. Display comparison results
    display_model_comparison(results, best_model_name, unit)

    # 9. Save artifacts
    save_artifacts(
        fitted_pipelines=fitted_pipelines,
        best_model_name=best_model_name,
        results=results,
        target_col=target_col,
        unit=unit,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        leakage_dropped=leakage_dropped,
        identifiers_dropped=identifiers_dropped,
        X=X,
        y=y,
        output_dir=output_dir,
    )

    # 10. Smoke test & verification
    verify_saved_model(output_dir, target_col, unit)

    logger.info("\n" + "=" * 75)
    logger.info("INDIA TECH JOBS SALARY MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()

