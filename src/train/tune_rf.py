# src/train/tune_rf.py

import logging
import joblib
import pandas as pd

from scipy.stats import randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SMOTE_RATIO, MODEL_PATH
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean
from src.preprocess.pipeline import build_pipeline  # builds SMOTE → Scaler → RandomForest


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # 1) Load & preprocess raw data
    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)

    # 2) Split features/label
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    logging.info(f"Data split | train={X_train.shape}, test={X_test.shape}")

    # 3) Build base RandomForest pipeline
    pipeline = build_pipeline(smote_ratio=SMOTE_RATIO)

    # 4) Define hyperparameter search space for RandomForest
    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": [None] + list(range(5, 31, 5)),
        "model__max_features": ["sqrt", "log2", 0.2, 0.5],
        "model__min_samples_split": randint(2, 11),
        "model__min_samples_leaf": randint(1, 5)
    }

    # 5) Set up RandomizedSearchCV with stratified folds
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=10,             # try 10 random combinations (increase if you have time)
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,              # adjust to control CPU usage
        verbose=2,
        random_state=RANDOM_STATE
    )

    # 6) Execute the search
    logging.info("Starting RandomForest hyperparameter search...")
    search.fit(X_train, y_train)

    # 7) Log best parameters and CV score
    best_params = search.best_params_
    best_cv_auc = search.best_score_
    logging.info(f"Best RF params: {best_params}")
    logging.info(f"Best RF CV AUC: {best_cv_auc:.4f}")

    # 8) Evaluate the best pipeline on the hold-out test set
    best_rf = search.best_estimator_
    test_probs = best_rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    logging.info(f"RF Test ROC AUC: {test_auc:.4f}")

    # 9) Persist the tuned pipeline to MODEL_PATH
    joblib.dump(best_rf, MODEL_PATH)
    logging.info(f"Saved tuned RF pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
