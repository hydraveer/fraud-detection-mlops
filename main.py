from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import mlflow
import time

app = FastAPI(title="Fraud Detection API")

mlflow.set_tracking_uri("http://localhost:5001")
model = mlflow.sklearn.load_model("models:/FraudDetector@champion")

class TransactionRequest(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "Fraud Detection API running"}

@app.post("/predict")
def predict(transaction: TransactionRequest):
    start = time.time()
    prediction = model.predict([transaction.features])
    probability = model.predict_proba([transaction.features])[0][1]
    latency_ms = round((time.time() - start) * 1000, 2)

    return {
        "prediction": "fraud" if prediction[0] == 1 else "legitimate",
        "probability": round(float(probability), 6),
        "latency_ms": latency_ms
    }

