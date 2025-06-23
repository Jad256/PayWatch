# src/train/cross_validate.py

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import RANDOM_STATE, MODEL_PATH
from src.data.load_all import load_and_normalize
from src.data.feature_engineering import add_time_features, add_volume_features
from src.data.clean import clean

def main():
    # 1) Load & preprocess
    df = load_and_normalize()
    df = add_time_features(df)
    df = add_volume_features(df)
    df = clean(df)
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    # 2) Load trained pipeline
    pipe = joblib.load(MODEL_PATH)

    # 3) 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # 4) Report AUCs
    print("AUC per fold :", [round(s, 4) for s in scores])
    print("Mean AUC     :", round(scores.mean(), 4))
    print("Std. Dev.    :", round(scores.std(), 4))

if __name__ == "__main__":
    main()
