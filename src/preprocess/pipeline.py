# src/preprocess/pipeline.py

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# pull in the default ratio from config
from src.config import SMOTE_RATIO 

def build_pipeline(smote_ratio: float = SMOTE_RATIO) -> Pipeline:
    """Return a ready-to-fit pipeline: SMOTE → Scaling → RandomForest."""
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ))
    ])


def build_gbm_pipeline(smote_ratio : float = SMOTE_RATIO) -> Pipeline:
    """
    Build a pipeline using LightGBM:
      1) SMOTE → 2) StandardScaler → 3) LGBMClassifier
    """
    return Pipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=42)),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42
        ))
    ])
