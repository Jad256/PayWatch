# src/train/cross_validate.py

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import DATA_PATH, RANDOM_STATE, MODEL_PATH
from src.data.feature_engineering import add_time_features
from src.data.clean import clean

def main():
    # 1) Load & prep data
    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)
    df = clean(df)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    # 2) Load the trained pipeline from MODEL_PATH
    pipe = joblib.load(MODEL_PATH)

    # 3) Stratified 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 4) Compute ROC AUC per fold
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # 5) Print results
    print("AUC per fold : ", [round(s, 4) for s in scores])
    print("Mean AUC     : ", round(scores.mean(), 4))
    print("Std. Dev.    : ", round(scores.std(), 4))

if __name__ == "__main__":
    main()
