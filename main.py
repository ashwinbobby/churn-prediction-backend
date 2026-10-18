from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ART = Path(__file__).parent / "artifacts"
THRESHOLD = 0.4  # same as val_preds in train_mlflow.py

app = FastAPI(title="Churn Prediction API (Logistic Regression)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://churn-prediction-frontend-kappa.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

preprocessor = joblib.load(ART / "preprocessor.pkl")
params = np.load(ART / "logreg.npz")
W, B = params["w"], float(params["b"][0])


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    prob = float(1 / (1 + np.exp(-(X[0] @ W + B))))
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": bool(prob >= THRESHOLD),
    }