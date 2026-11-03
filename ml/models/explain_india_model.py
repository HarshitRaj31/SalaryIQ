
from pathlib import Path
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_india_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "india_salary_model_ready.csv"
REPORT_DIR = BASE_DIR / "reports" / "shap"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load trained model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Remove target and salary leakage columns
drop_columns = [
    "salary_in_lpa",
    "salary_in_usd",
    "salary_in_inr",
    "salary_min_lpa",
    "salary_max_lpa",
    "salary_midpoint_lpa"
]

X = df.drop(columns=drop_columns, errors="ignore")

# Explain the model's transformed feature space
preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

X_transformed = preprocessor.transform(X)

feature_names = preprocessor.get_feature_names_out()

# SHAP Tree Explainer
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_transformed)

# Global feature importance
plt.figure()
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig(REPORT_DIR / "shap_summary.png", dpi=300)
plt.close()

# Mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))

print("\nSHAP analysis completed successfully!")
print(f"Reports saved at: {REPORT_DIR}")
