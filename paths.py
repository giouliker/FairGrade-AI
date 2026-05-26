"""
Repository-root paths for FairGrade AI.

Resolved from this file's location so training and inference work regardless of
the process current working directory (e.g. Streamlit Community Cloud).
"""

from __future__ import annotations

from pathlib import Path

# Directory containing paths.py (project root).
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
FAIR_MODEL_PATH = MODELS_DIR / "fair_model.pt"
METRICS_PATH = MODELS_DIR / "metrics.json"


def ensure_models_dir() -> Path:
    """Create the models output directory if it does not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR
