from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import mlflow.sklearn
import mlflow
import time

app = FastAPI(title="Fraud Detection API")

mlflow.set_tracking_uri("http://localhost:5001")
model = mlflow.sklearn.load_model("models:/FraudDetector@champion")

# Prometheus metrics
REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Total prediction requests",
    ["prediction"]
)

LATENCY = Histogram(
    "fraud_latency_ms",
    "Prediction latency in milliseconds"
)

FRAUD_PROBABILITY = Histogram(
    "fraud_probability",
    "Distribution of fraud probabilities"
)

class TransactionRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "Fraud Detection API running"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(transaction: TransactionRequest):
    start = time.time()
    prediction = model.predict([transaction.features])
    probability = model.predict_proba([transaction.features])[0][1]
    latency_ms = round((time.time() - start) * 1000, 2)

    label = "fraud" if prediction[0] == 1 else "legitimate"

    REQUEST_COUNT.labels(prediction=label).inc()
    LATENCY.observe(latency_ms)
    FRAUD_PROBABILITY.observe(float(probability))

    return {
        "prediction": label,
        "probability": round(float(probability), 6),
        "latency_ms": latency_ms
    }