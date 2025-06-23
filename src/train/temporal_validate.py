# src/train/temporal_validate.py

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from src.config import DATA_PATH, MODEL_PATH
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean

def main():
    # 1) Load raw data
    df = pd.read_csv(DATA_PATH)

    # 2) Derive time-of-day features (if you’re using them)
    df = add_time_features(df)
    df = add_volume_features(df)

    # 3) Clean
    df = clean(df)

    # 4) Compute time threshold at 80th percentile
    threshold = df["Time"].quantile(0.8)
    print(f"Using Time <= {threshold:.0f}s for training, > {threshold:.0f}s for unseen test")

    # 5) Split temporally
    train_df = df[df["Time"] <= threshold]
    test_df  = df[df["Time"]  > threshold]

    X_train = train_df.drop(["Class", "Time"], axis=1)
    y_train = train_df["Class"]
    X_test  = test_df.drop(["Class", "Time"], axis=1)
    y_test  = test_df["Class"]

    print(f"Train shape: {X_train.shape}, Test (unseen) shape: {X_test.shape}")

    # 6) Retrain your **best** model on the full train_df
    #    (or load a saved, tuned pipeline)
    pipe = joblib.load(MODEL_PATH)
    pipe.fit(X_train, y_train)

    # 7) Evaluate on the truly unseen test set
    probs = pipe.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)
    print(f"Temporal hold-out ROC AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
