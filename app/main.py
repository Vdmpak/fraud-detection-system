from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

model = joblib.load(MODEL_DIR / "fraud_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
config = joblib.load(MODEL_DIR / "model_config.joblib")

THRESHOLD = round(config["threshold"], 2)
FEATURES = config["features"]


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "fraud_detection",
    "user": "postgres",
}


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud probability prediction service",
    version="1.0.0",
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

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


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data["Amount_log"] = np.log1p(data["Amount"])

    seconds_per_day = 24 * 60 * 60
    hour = (data["Time"] % seconds_per_day) / 3600

    data["Time_sin"] = np.sin(2 * np.pi * hour / 24)
    data["Time_cos"] = np.cos(2 * np.pi * hour / 24)

    return data


# --------------------------------------------------
# Database helper
# --------------------------------------------------

def save_prediction(fraud_probability: float, is_fraud: bool):
    connection = psycopg2.connect(**DB_CONFIG)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                fraud_probability,
                is_fraud
            )
            VALUES (%s, %s)
            RETURNING id;
            """,
            (fraud_probability, is_fraud),
        )

        prediction_id = cursor.fetchone()[0]

        connection.commit()

        return prediction_id

    finally:
        cursor.close()
        connection.close()


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "threshold": THRESHOLD,
    }


# --------------------------------------------------
# Fraud prediction
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert request to DataFrame
    data = pd.DataFrame([transaction.model_dump()])

    # Feature engineering
    data = engineer_features(data)

    # Ensure correct feature order
    data = data[FEATURES]

    # Scale features
    data_scaled = scaler.transform(data)

    # Fraud probability
    fraud_probability = float(
        model.predict_proba(data_scaled)[0, 1]
    )

    # Apply threshold
    is_fraud = fraud_probability >= THRESHOLD

    # Store prediction
    prediction_id = save_prediction(
        fraud_probability,
        is_fraud
    )

    return {
        "prediction_id": prediction_id,
        "fraud_probability": round(fraud_probability, 6),
        "is_fraud": is_fraud,
        "threshold": THRESHOLD,
    }
