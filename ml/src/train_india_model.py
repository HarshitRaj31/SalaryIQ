"""
Job Salary Prediction System - India Tech Jobs ML Training Pipeline
File: ml/src/train_india_model.py

Re-exports and provides access to the India tech jobs model training pipeline
from ml/models/train_india_model.py.
"""

from pathlib import Path
import sys

# Add ml/models to path for imports
models_dir = Path(__file__).resolve().parent.parent / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from train_india_model import (
    main,
    detect_target_column,
    remove_leakage_and_identifiers,
    identify_feature_types,
    build_preprocessors,
    evaluate_5fold_cv,
    train_and_compare_models,
    save_artifacts,
    verify_saved_model,
)

if __name__ == "__main__":
    main()

