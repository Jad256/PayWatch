# src/train/tune_gbm.py

import logging
import joblib
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE, SMOTE_RATIO, MODEL_PATH
from src.data.feature_engineering import add_time_features
from src.data.clean import clean
from src.preprocess.pipeline import build_gbm_pipeline

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # 1) Load & prep data
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
    logging.info(f"Data split | train={X_train.shape}, test={X_test.shape}")

    # 2) Base LightGBM pipeline
    pipeline = build_gbm_pipeline(smote_ratio=SMOTE_RATIO)

    # 3) Define search space for LGBMClassifier inside the pipeline
    param_dist = {
        "model__n_estimators": randint(100, 1000),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__num_leaves": randint(20, 150),
        "model__max_depth": randint(3, 15),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__min_child_samples": randint(5, 100)
    }

    # 4) Set up randomized search with stratified folds
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,               # try 20 random combinations
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,                # limit cores so your machine stays responsive
        verbose=2,
        random_state=RANDOM_STATE
    )

    # 5) Run the search
    logging.info("Starting LightGBM hyperparameter search...")
    search.fit(X_train, y_train)

    # 6) Best CV results
    logging.info(f"Best params: {search.best_params_}")
    logging.info(f"Best CV AUC: {search.best_score_:.4f}")

    # 7) Evaluate on hold‐out test
    best_pipe = search.best_estimator_
    test_probs = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    logging.info(f"Test ROC AUC: {test_auc:.4f}")

    # 8) Persist the tuned pipeline
    joblib.dump(best_pipe, MODEL_PATH)
    logging.info(f"Saved tuned GBM pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
