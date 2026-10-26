import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def transaction_to_payload(transaction):
    return {
        "Time": float(transaction["Time"]),
        "V1": float(transaction["V1"]),
        "V2": float(transaction["V2"]),
        "V3": float(transaction["V3"]),
        "V4": float(transaction["V4"]),
        "V5": float(transaction["V5"]),
        "V6": float(transaction["V6"]),
        "V7": float(transaction["V7"]),
        "V8": float(transaction["V8"]),
        "V9": float(transaction["V9"]),
        "V10": float(transaction["V10"]),
        "V11": float(transaction["V11"]),
        "V12": float(transaction["V12"]),
        "V13": float(transaction["V13"]),
        "V14": float(transaction["V14"]),
        "V15": float(transaction["V15"]),
        "V16": float(transaction["V16"]),
        "V17": float(transaction["V17"]),
        "V18": float(transaction["V18"]),
        "V19": float(transaction["V19"]),
        "V20": float(transaction["V20"]),
        "V21": float(transaction["V21"]),
        "V22": float(transaction["V22"]),
        "V23": float(transaction["V23"]),
        "V24": float(transaction["V24"]),
        "V25": float(transaction["V25"]),
        "V26": float(transaction["V26"]),
        "V27": float(transaction["V27"]),
        "V28": float(transaction["V28"]),
        "Amount": float(transaction["Amount"]),
    }


def get_valid_transaction():
    df = pd.read_csv("data/creditcard.csv")
    return transaction_to_payload(df.iloc[0])


def get_fraud_transaction():
    df = pd.read_csv("data/creditcard.csv")
    fraud = df[df["Class"] == 1].iloc[0]
    return transaction_to_payload(fraud)


def mock_save_prediction(fraud_probability, is_fraud):
    return 999


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "Random Forest"
    assert data["threshold"] == 0.35


def test_valid_prediction(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_prediction",
        mock_save_prediction
    )

    payload = get_valid_transaction()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction_id"] == 999
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "threshold" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert isinstance(data["is_fraud"], bool)
    assert data["threshold"] == 0.35


def test_fraud_transaction(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_prediction",
        mock_save_prediction
    )

    payload = get_fraud_transaction()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["is_fraud"] is True
    assert data["fraud_probability"] >= 0.35


def test_missing_required_field(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_prediction",
        mock_save_prediction
    )

    payload = get_valid_transaction()

    del payload["V14"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_invalid_amount(monkeypatch):
    monkeypatch.setattr(
        "app.main.save_prediction",
        mock_save_prediction
    )

    payload = get_valid_transaction()

    payload["Amount"] = "not-a-number"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
