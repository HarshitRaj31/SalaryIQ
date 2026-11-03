"""
Deduplication and Validation Script for Model-Ready Indian Tech Jobs Dataset
File: fix_model_dataset.py

Re-exports and provides access to fix_and_deduplicate_dataset pipeline in ml/src directory.
"""

from pathlib import Path
import sys

# Import from ml/models/fix_model_dataset.py
models_dir = Path(__file__).resolve().parent.parent / "models"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from fix_model_dataset import fix_and_deduplicate_dataset, get_processed_data_paths

if __name__ == "__main__":
    fix_and_deduplicate_dataset()

