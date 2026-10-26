# Credit Card Fraud Detection System

A machine learning fraud detection service that classifies credit card transactions in real time using a Random Forest model, FastAPI, and PostgreSQL.

## Overview

This project builds an end-to-end fraud detection pipeline using the Credit Card Fraud Detection dataset containing **284,807 transactions**, including **492 fraudulent transactions**.

The system:

- Engineers transaction-level features
- Handles severe class imbalance using class-weighted training
- Trains and evaluates a Random Forest classifier
- Selects a decision threshold using validation data
- Exposes the model through a FastAPI REST API
- Stores prediction results in PostgreSQL
- Includes automated API tests with pytest

## Architecture

```text
Credit Card Dataset
        |
        v
     Pandas
        |
        v
Feature Engineering
        |
        v
Scikit-learn Random Forest
        |
        v
   Saved Model
        |
        v
     FastAPI
        |
        v
 Fraud Probability
        |
        v
   PostgreSQL
```

## Dataset

The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Dataset statistics:

| Metric                  |   Value |
| ----------------------- | ------: |
| Transactions            | 284,807 |
| Fraudulent transactions |     492 |
| Fraud rate              | 0.1727% |
| Input features          |      30 |
| Engineered features     |      33 |

The dataset is intentionally excluded from this repository because of its size.

Download `creditcard.csv` from Kaggle and place it at:

```text
data/creditcard.csv
```

## Machine Learning

### Feature Engineering

Three additional features were created:

- `Amount_log` — log-transformed transaction amount
- `Time_sin` — cyclical representation of transaction time
- `Time_cos` — cyclical representation of transaction time

This resulted in **33 features** used by the final model.

The preprocessing pipeline includes feature scaling using a saved scikit-learn scaler so that the API applies the same transformations used during training.

### Models

Two models were evaluated:

- Logistic Regression baseline
- Random Forest

The final Random Forest configuration used:

- 200 trees
- Balanced class weighting
- Decision threshold: `0.35`

The threshold was selected using validation data rather than directly optimizing on the test set.

## Model Performance

Final Random Forest performance on the held-out test set:

| Metric          | Result |
| --------------- | -----: |
| Precision       | 86.46% |
| Recall          | 84.69% |
| F1 Score        | 85.57% |
| ROC-AUC         | 96.04% |
| False positives |     13 |
| False negatives |     15 |
| True positives  |     83 |
| True negatives  | 56,851 |

The test set contained **98 actual fraudulent transactions**, of which the model detected **83**.

### Logistic Regression Comparison

| Model               |  Precision | Recall |         F1 | ROC-AUC |
| ------------------- | ---------: | -----: | ---------: | ------: |
| Logistic Regression |     51.52% | 86.73% |     64.64% |  97.45% |
| Random Forest       | **86.46%** | 84.69% | **85.57%** |  96.04% |

Random Forest was selected because it provided a substantially better precision/F1 trade-off while maintaining strong recall.

## API

The application provides two endpoints.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "Random Forest",
  "threshold": 0.35
}
```

### Fraud Prediction

```http
POST /predict
```

The endpoint accepts transaction features and returns a fraud probability and classification.

Example response:

```json
{
  "prediction_id": 1,
  "fraud_probability": 0.685,
  "is_fraud": true,
  "threshold": 0.35
}
```

Interactive API documentation is available through FastAPI at:

```text
http://127.0.0.1:8000/docs
```

## PostgreSQL

Prediction results are stored in PostgreSQL.

The `predictions` table contains:

```text
id
fraud_probability
is_fraud
created_at
```

This allows prediction results to be persisted and queried after inference.

## Testing

The project uses pytest for automated API testing.

The test suite covers:

- Health endpoint
- Valid prediction requests
- Fraud transaction classification
- Missing required fields
- Invalid input types

Current test result:

```text
5 passed
```

## Performance

A local benchmark of 100 HTTP requests to `/predict` produced:

| Metric          |   Result |
| --------------- | -------: |
| Mean latency    | 19.02 ms |
| Median latency  | 16.91 ms |
| P95 latency     | 28.46 ms |
| Requests tested |      100 |

The benchmark was performed locally during development and is not intended as a production performance guarantee.

The underlying Random Forest batch inference benchmark processed the 56,962-record test set in approximately **0.089 seconds**.

## Project Structure

```text
fraud-detection-system/
├── app/
│   ├── __init__.py
│   └── main.py
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.joblib
│   ├── model_config.joblib
│   └── scaler.joblib
├── notebooks/
│   └── 01_dataset_exploration.ipynb
├── tests/
│   └── test_api.py
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── venv/
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd fraud-detection-system
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `creditcard.csv` from Kaggle and place it in:

```text
data/creditcard.csv
```

### 5. Configure PostgreSQL

Create the database:

```sql
CREATE DATABASE fraud_detection;
```

Then create the predictions table:

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    fraud_probability DOUBLE PRECISION NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Run tests

```bash
pytest -v
```

## Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **FastAPI**
- **PostgreSQL**
- **pytest**
- **Joblib**
