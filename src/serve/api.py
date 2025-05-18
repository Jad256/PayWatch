# src/serve/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from src.config import MODEL_PATH

app = FastAPI(
    title="Credit Card Fraud Detector",
    description="Predicts whether a transaction is fraud",
    version="1.0.0"
)

# 1) Define the request schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# 2) Load your trained pipeline once at startup
pipeline = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(tx: Transaction):
    """
    Accept a single transaction and return:
      - fraud: bool
      - score: float probability of fraud
    """
    # Turn the incoming data into the right shape
    vals = np.array(list(tx.dict().values())).reshape(1, -1)
    pred = pipeline.predict(vals)[0]
    prob = pipeline.predict_proba(vals)[0, 1]

    return {"fraud": bool(pred), "score": float(prob)}
