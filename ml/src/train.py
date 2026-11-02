
"""
Advanced Model Training and Hyperparameter Tuning
Job Salary Prediction Using Regression

Models:
1. Linear Regression (baseline)
2. Random Forest (RandomizedSearchCV)
3. Gradient Boosting (RandomizedSearchCV)

Uses:
- 80/20 train-test split
- 5-fold cross-validation
- RMSE-based model selection
- MAE, RMSE, R2 final evaluation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "cleaned_salary.csv"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# BUILD PREPROCESSOR
# --------------------------------------------------

def build_preprocessor(categorical_features, numerical_features):

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ]
    )


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "R2_Score": float(
            r2_score(y_true, y_pred)
        )
    }


# --------------------------------------------------
# SAVE MODEL ARTIFACTS
# --------------------------------------------------

def save_artifacts(
    fitted_pipelines,
    best_model_name,
    results,
    categorical_features,
    numerical_features,
    X
):

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save individual models
    model_files = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest Regressor": "random_forest_regressor.pkl",
        "Gradient Boosting Regressor": "gradient_boosting_regressor.pkl"
    }

    for name, pipeline in fitted_pipelines.items():

        model_path = os.path.join(
            MODELS_DIR,
            model_files[name]
        )

        joblib.dump(pipeline, model_path)

        print(f"Saved {name}: {model_path}")

    # Save best model
    best_model_path = os.path.join(
        MODELS_DIR, "best_model.pkl"
    )

    joblib.dump(
        fitted_pipelines[best_model_name],
        best_model_path
    )

    print("\nBest model saved:", best_model_path)

    # Save comparison results
    comparison_path = os.path.join(
        MODELS_DIR, "model_comparison.json"
    )

    with open(comparison_path, "w", encoding="utf-8") as f:

        json.dump(
            {
                "best_model": best_model_name,
                "selection_metric": "5-fold CV RMSE",
                "metrics": results
            },
            f,
            indent=4
        )

    # Save feature metadata
    metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target": "salary_in_usd",
        "categories": {
            col: sorted(
                X[col].dropna().unique().tolist()
            )
            for col in categorical_features
        }
    }

    metadata_path = os.path.join(
        MODELS_DIR, "feature_metadata.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Feature metadata saved:", metadata_path)


# --------------------------------------------------
# MAIN TRAINING PROCESS
# --------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("JOB SALARY PREDICTION - ADVANCED MODEL TRAINING")
    print("=" * 75)

    # 1. Load dataset
    df = load_data()

    target = "salary_in_usd"

    if target not in df.columns:
        raise ValueError(f"Target column missing: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "company_location",
        "company_size"
    ]

    numerical_features = [
        "work_year",
        "remote_ratio"
    ]

    expected_features = (
        categorical_features + numerical_features
    )

    missing_features = [
        col for col in expected_features
        if col not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing input features: {missing_features}"
        )

    X = X[expected_features]

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Cross-validation setup
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    preprocessor = build_preprocessor(
        categorical_features,
        numerical_features
    )

    fitted_pipelines = {}
    results = {}

    # --------------------------------------------------
    # MODEL 1: LINEAR REGRESSION
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL 1: LINEAR REGRESSION")
    print("=" * 65)

    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("regressor", LinearRegression())
        ]
    )

    linear_cv_scores = cross_val_score(
        linear_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    linear_pipeline.fit(X_train, y_train)

    linear_predictions = linear_pipeline.predict(X_test)

    linear_metrics = calculate_metrics(
        y_test,
        linear_predictions
    )

    linear_cv_rmse = float(-linear_cv_scores.mean())

    results["Linear Regression"] = {
        "CV_RMSE": linear_cv_rmse,
        "CV_RMSE_STD": float(linear_cv_scores.std()),
        **linear_metrics
    }

    fitted_pipelines["Linear Regression"] = linear_pipeline

    print("Cross-validation RMSE:", round(linear_cv_rmse, 2))
    print("Test MAE:", round(linear_metrics["MAE"], 2))
    print("Test RMSE:", round(linear_metrics["RMSE"], 2))
    print("Test R2:", round(linear_metrics["R2_Score"], 4))

    # --------------------------------------------------
    # MODEL 2: RANDOM FOREST
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL 2: RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 65)

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=1
                )
            )
        ]
    )

    rf_parameters = {
        "regressor__n_estimators": [200, 300, 400, 500],
        "regressor__max_depth": [None, 10, 15, 20, 30],
        "regressor__min_samples_split": randint(2, 11),
        "regressor__min_samples_leaf": randint(1, 5),
        "regressor__max_features": ["sqrt", 0.7, 1.0]
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_parameters,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_

    rf_predictions = best_rf.predict(X_test)

    rf_metrics = calculate_metrics(
        y_test,
        rf_predictions
    )

    rf_cv_rmse = float(-rf_search.best_score_)

    results["Random Forest Regressor"] = {
        "CV_RMSE": rf_cv_rmse,
        "CV_RMSE_STD": float(
            rf_search.cv_results_["std_test_score"][
                rf_search.best_index_
            ]
        ),
        **rf_metrics,
        "Best_Parameters": rf_search.best_params_
    }

    fitted_pipelines["Random Forest Regressor"] = best_rf

    print("\nBest Random Forest Parameters:")
    print(rf_search.best_params_)
    print("Best CV RMSE:", round(rf_cv_rmse, 2))
    print("Test MAE:", round(rf_metrics["MAE"], 2))
    print("Test RMSE:", round(rf_metrics["RMSE"], 2))
    print("Test R2:", round(rf_metrics["R2_Score"], 4))

    # --------------------------------------------------
    # MODEL 3: GRADIENT BOOSTING
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL 3: GRADIENT BOOSTING HYPERPARAMETER TUNING")
    print("=" * 65)

    gb_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "regressor",
                GradientBoostingRegressor(
                    random_state=42
                )
            )
        ]
    )

    gb_parameters = {
        "regressor__n_estimators": randint(100, 501),
        "regressor__learning_rate": uniform(0.02, 0.18),
        "regressor__max_depth": randint(2, 6),
        "regressor__min_samples_split": randint(2, 11),
        "regressor__min_samples_leaf": randint(1, 5),
        "regressor__subsample": uniform(0.7, 0.3)
    }

    gb_search = RandomizedSearchCV(
        estimator=gb_pipeline,
        param_distributions=gb_parameters,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    gb_search.fit(X_train, y_train)

    best_gb = gb_search.best_estimator_

    gb_predictions = best_gb.predict(X_test)

    gb_metrics = calculate_metrics(
        y_test,
        gb_predictions
    )

    gb_cv_rmse = float(-gb_search.best_score_)

    results["Gradient Boosting Regressor"] = {
        "CV_RMSE": gb_cv_rmse,
        "CV_RMSE_STD": float(
            gb_search.cv_results_["std_test_score"][
                gb_search.best_index_
            ]
        ),
        **gb_metrics,
        "Best_Parameters": gb_search.best_params_
    }

    fitted_pipelines["Gradient Boosting Regressor"] = best_gb

    print("\nBest Gradient Boosting Parameters:")
    print(gb_search.best_params_)
    print("Best CV RMSE:", round(gb_cv_rmse, 2))
    print("Test MAE:", round(gb_metrics["MAE"], 2))
    print("Test RMSE:", round(gb_metrics["RMSE"], 2))
    print("Test R2:", round(gb_metrics["R2_Score"], 4))

    # --------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    comparison = pd.DataFrame(results).T

    comparison = comparison.sort_values(
        by="CV_RMSE",
        ascending=True
    )

    print(
        comparison[
            ["CV_RMSE", "CV_RMSE_STD", "MAE", "RMSE", "R2_Score"]
        ].to_string()
    )

    # Select using cross-validation only
    best_model_name = comparison.index[0]

    print("\nSELECTED MODEL:", best_model_name)

    print(
        "Best CV RMSE:",
        round(results[best_model_name]["CV_RMSE"], 2)
    )

    # --------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------

    save_artifacts(
        fitted_pipelines,
        best_model_name,
        results,
        categorical_features,
        numerical_features,
        X
    )

    print("\n" + "=" * 75)
    print("MODEL TRAINING AND TUNING COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
