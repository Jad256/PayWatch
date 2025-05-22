# src/train/train_gbm_mlflow.py

import logging
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, MODEL_PATH
from src.data.feature_engineering import add_time_features
from src.data.clean import clean
from src.preprocess.pipeline import build_gbm_pipeline

def main():
    # 1) Load & prep
    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)
    df = clean(df)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # 2) Set up MLflow & autolog ONCE
    mlflow.set_experiment("CreditCardFraud_GBM")
    mlflow.sklearn.autolog(log_input_examples=True)

    with mlflow.start_run():
        # 3) Build our LightGBM pipeline
        pipeline = build_gbm_pipeline(smote_ratio=0.1)

        # 4) Fit exactly ONCE, without extra eval_set / early_stopping args
        pipeline.fit(X_train, y_train)

        # 5) Manually infer & log signature + input example
        preds_train = pipeline.predict(X_train)
        signature = infer_signature(X_train, preds_train)
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,
            input_example=X_train.head(2)
        )

        # 6) Evaluate on hold-out & log metric
        probs = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("test_auc", float(auc))

        print(f"Logged GBM run with test AUC: {auc:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
