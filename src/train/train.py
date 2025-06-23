# src/train/train.py

import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import TEST_SIZE, RANDOM_STATE, MODEL_PATH, SMOTE_RATIO
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline

def main():
    # 0) Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # 1) Load & concatenate all raw CSVs
    df = load_and_normalize()
    logging.info(f"Loaded & concatenated data (shape={df.shape})")

    # 2) Feature engineering
    df = add_time_features(df)
    df = add_volume_features(df)
    logging.info("Added time-of-day and tx_per_minute features")

    # 3) Clean data
    df = clean(df)
    logging.info(f"After clean() (shape={df.shape})")

    # 4) Split into features and label
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    logging.info(f"Train/test split: train={X_train.shape}, test={X_test.shape}")

    # 5) Build & fit the RandomForest pipeline
    pipeline = build_pipeline(smote_ratio=SMOTE_RATIO)
    pipeline.fit(X_train, y_train)
    logging.info("Pipeline training complete")

    # 6) Feature importances (optional)
    importances = pipeline.named_steps["model"].feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    logging.info("Top 10 features:\n%s", feat_imp.head(10).to_string(index=False))

    # 7) Evaluate on hold-out set
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))
    logging.info("Classification Report:\n%s", classification_report(y_test, preds))
    logging.info("ROC AUC Score: %.4f", roc_auc_score(y_test, probs))

    # 8) Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    logging.info(f"Saved pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
