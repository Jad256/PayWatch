# src/config.py

from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
# This file lives in <project>/src/config.py,
# so parent.parent is your project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── Data & model paths ────────────────────────────────────────────────────────
DATA_PATH   = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH  = PROJECT_ROOT / "models" / "trained_model.pkl"
SCALER_PATH = PROJECT_ROOT / "data" / "scaler.pkl"

# ─── Other constants ───────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
SMOTE_RATIO  = 0.1
