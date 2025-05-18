# src/preprocess/pipeline.py

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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
