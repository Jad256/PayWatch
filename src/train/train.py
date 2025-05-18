# src/train/train.py

import logging
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SMOTE_RATIO, MODEL_PATH
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline

def main():
    # 1) Load raw CSV
    df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded data from {DATA_PATH} (shape={df.shape})")

    # 2) Clean it
    df = clean(df)
    logging.info(f"After clean() (shape={df.shape})")

    # 3) Split into features & label
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    logging.info(f"Train/Test split: train={X_train.shape}, test={X_test.shape}")

    # 4) Build & fit pipeline
    pipeline = build_pipeline(smote_ratio=SMOTE_RATIO)
    pipeline.fit(X_train, y_train)
    logging.info("Pipeline training complete.")

    # 5) Evaluate
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:,1]
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, preds))
    logging.info("Classification Report:\n%s", classification_report(y_test, preds))
    logging.info("ROC AUC Score: %.4f", roc_auc_score(y_test, probs))

    # 6) Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    logging.info(f"Saved pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
