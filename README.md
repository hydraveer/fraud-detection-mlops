# Fraud Detection MLOps

[![CI](https://github.com/hydraveer/fraud-detection-mlops/actions/workflows/ci.yaml/badge.svg)](https://github.com/hydraveer/fraud-detection-mlops/actions/workflows/ci.yaml)
![Python](https://img.shields.io/badge/python-3.11-blue)

An end-to-end MLOps system for credit card fraud detection. It covers the full model lifecycle: training with experiment tracking, model registry, real-time serving, production monitoring, data drift detection, and automated Champion/Challenger promotion.

## Overview

Fraud detection is a highly imbalanced problem: only **0.172%** of transactions are fraudulent. This project goes beyond training a classifier and builds the infrastructure around it:

- **Reproducible training.** Every run logs its parameters, metrics and model artifact to MLflow and registers the model in the Model Registry.
- **Safe deployments.** The API always serves the model tagged `@champion`. A newly trained model replaces it only when it scores a strictly higher F1.
- **Drift-aware retraining.** Evidently compares incoming data against reference data, and retraining runs only when dataset drift is detected.
- **Observability.** Every prediction updates Prometheus metrics (request counts, latency, fraud-probability distribution) that Grafana can chart.

## Architecture

```
                    ┌─────────────────────────── TRAINING ───────────────────────────┐
                    │                                                                 │
  creditcard.csv ──►│  DataLoader       Preprocessor            Trainer               │
  (CSV/Parquet/     │  (Factory)   ──►  (Pipeline)         ──►  (Factory)        ──┐  │
   JSON)            │                   Validate → Drop NA      RandomForest /     │  │
                    │                   → Scale → Drop cols     GradientBoosting   │  │
                    └──────────────────────────────────────────────────────────────┼──┘
                                                                                   │ log params,
                                                                                   │ metrics, model
                                                                                   ▼
  ┌──────────── RETRAINING ────────────┐                        ┌───────────────────────────┐
  │                                    │   drift detected       │      MLflow Server         │
  │  Evidently DriftDetector           │ ─────────────────────► │  Experiment tracking       │
  │  reference vs. current data        │   train challenger     │  Model Registry            │
  │                                    │                        │  FraudDetector@champion    │
  │  Champion/Challenger comparison    │ ◄───────────────────── │                            │
  │  promote if F1 strictly better     │   champion metrics     └─────────────┬──────────────┘
  └────────────────────────────────────┘                                      │ load @champion
                                                                              ▼
  ┌──────────── MONITORING ────────────┐   scrape /metrics      ┌───────────────────────────┐
  │  Prometheus  ──►  Grafana          │ ◄───────────────────── │   FastAPI  (port 8000)    │
  │  (every 15s)      dashboards       │                        │   POST /predict           │
  └────────────────────────────────────┘                        │   GET  /metrics           │
                                                                └───────────────────────────┘
                                                                              ▲
                                                                              │ JSON
                                                                           Client
```

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Modeling | scikit-learn | RandomForest and GradientBoosting classifiers |
| Data | pandas, NumPy | Loading and preprocessing |
| Experiment tracking | MLflow | Run tracking, Model Registry, alias-based promotion |
| Serving | FastAPI, Uvicorn, Pydantic | REST inference API with request validation |
| Monitoring | Prometheus, Grafana | Metrics scraping and dashboards |
| Drift detection | Evidently | Statistical data drift reports |
| Containerization | Docker, Docker Compose | Reproducible services |
| CI/CD | GitHub Actions | Automated testing on every push and PR |
| Testing | pytest | Unit tests for preprocessing, training and the pipeline |

## Design Patterns

### Factory Pattern: `DataLoader` and `Trainer`

Callers ask a factory for an object and never instantiate concrete classes themselves. To add a new file format or model, write one class and add one entry to the registry dict. Existing code does not change (Open/Closed Principle).

```python
from src.data_loader import get_loader
from src.trainer import get_trainer

loader  = get_loader("data/creditcard.csv")          # → CSVLoader (also .parquet, .json)
trainer = get_trainer("gradient_boosting",           # → GradientBoostingTrainer
                      n_estimators=50, max_depth=4)
```

Every loader implements the `DataLoader` interface (`validate`, `load`). Every trainer implements the `BaseTrainer` interface (`train`, `evaluate`, `get_params`), so the pipeline handles all models the same way. An unsupported format or model name raises a clear `ValueError` that lists the supported options.

### Pipeline Pattern: `DataPreprocessor`

Preprocessing is a chain of small, single-purpose steps. Each step implements `PreprocessingStep.run(df) -> df`, which makes the steps easy to test on their own, reorder, and reuse.

```python
from src.preprocessor import (
    DataPreprocessor, ValidateSchema, DropMissingValues, ScaleFeatures, DropColumns,
)

preprocessor = DataPreprocessor(steps=[
    ValidateSchema(required_columns=["Time", "Amount", "Class"]),
    DropMissingValues(),
    ScaleFeatures(columns=["Amount", "Time"]),
    DropColumns(columns=["Time"]),
])
df = preprocessor.run(df)
```

## Project Structure

```
fraud-detection-mlops/
├── src/
│   ├── data_loader.py      # Factory pattern: CSV / Parquet / JSON loaders
│   ├── preprocessor.py     # Pipeline pattern: chainable cleaning steps
│   ├── trainer.py          # Factory pattern: RandomForest / GradientBoosting trainers
│   ├── pipeline.py         # End-to-end training orchestration with MLflow
│   ├── retrain.py          # Champion/Challenger promotion + drift-triggered retraining
│   └── drift_detector.py   # Evidently drift reports
├── tests/                  # pytest suite (runs on synthetic data, no CSV needed)
├── main.py                 # FastAPI app: /predict and /metrics endpoints
├── Dockerfile              # API container image
├── docker-compose.yml      # MLflow server + API
├── prometheus.yml          # Prometheus scrape config
├── reports/                # Generated Evidently HTML drift reports
├── .github/workflows/
│   └── ci.yaml             # GitHub Actions CI
└── requirements.txt
```

## Quick Start

**Prerequisites:** Python 3.11, Docker, and the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**1. Clone and install**

```bash
git clone https://github.com/hydraveer/fraud-detection-mlops.git
cd fraud-detection-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -c constraints.txt
```

**2. Add the dataset**

Download `creditcard.csv` from Kaggle and place it at `data/creditcard.csv`.

**3. Start the MLflow server**

```bash
docker compose up -d mlflow        # UI at http://localhost:5001
```

**4. Train and register a model**

```bash
python -m src.pipeline
```

This logs the run to the `fraud-detection` experiment and registers a new version of `FraudDetector`. Settings such as model type, hyperparameters and row count are in `PipelineConfig` in `src/pipeline.py`.

**5. Mark the first model as champion**

The API serves `FraudDetector@champion`, so the first model version needs the alias:

```bash
python -c "import mlflow; mlflow.set_tracking_uri('http://localhost:5001'); \
mlflow.MlflowClient().set_registered_model_alias('FraudDetector', 'champion', '1')"
```

**6. Start the API**

```bash
docker compose up -d --build fraud-api    # http://localhost:8000/docs
```

**7. Start monitoring (optional)**

Set the target in `prometheus.yml` to your host's address, for example `host.docker.internal:8000`. Then run:

```bash
docker run -d --name prometheus -p 9090:9090 \
  -v "$PWD/prometheus.yml:/etc/prometheus/prometheus.yml" prom/prometheus
docker run -d --name grafana -p 3000:3000 grafana/grafana
```

In Grafana at `http://localhost:3000`, add Prometheus as a data source (`http://host.docker.internal:9090`) and build panels from the metrics below.

**8. Run drift-triggered retraining**

```bash
python src/retrain.py
```

**9. Run the tests**

```bash
PYTHONPATH=. pytest tests/ -v
```

## API Documentation

Interactive Swagger docs are available at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Classify a transaction |
| `GET` | `/metrics` | Prometheus metrics (text exposition format) |

### `POST /predict`

`features` is an ordered list of **29 floats**: `V1`–`V28` followed by `Amount`, matching the training columns after `Time` is dropped.

**Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [-1.3598, -0.0728, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987,
                  0.3638,  0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704,
                  0.2080,  0.0258,  0.4040,  0.2514, -0.0183,  0.2778, -0.1105, 0.0669,
                  0.1285, -0.1891,  0.1336, -0.0211, 0.2450]
  }'
```

**Response `200 OK`**

```json
{
  "prediction": "legitimate",
  "probability": 0.012345,
  "latency_ms": 3.21
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | string | `"fraud"` or `"legitimate"` |
| `probability` | float | Model-estimated probability of fraud (0–1) |
| `latency_ms` | float | Model inference time in milliseconds |

A malformed body, for example non-numeric features, returns `422 Unprocessable Entity` with Pydantic validation details.

### `GET /`

```json
{ "status": "Fraud Detection API running" }
```

## Monitoring Metrics

Prometheus scrapes `/metrics` every 15 seconds.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `fraud_requests_total` | Counter | `prediction` = `fraud` \| `legitimate` | Total prediction requests by outcome |
| `fraud_latency_ms` | Histogram | — | Model inference latency in milliseconds |
| `fraud_probability` | Histogram | — | Distribution of predicted fraud probabilities |

**Example PromQL queries**

```promql
# Fraud rate over the last 5 minutes
sum(rate(fraud_requests_total{prediction="fraud"}[5m])) / sum(rate(fraud_requests_total[5m]))

# p95 inference latency
histogram_quantile(0.95, rate(fraud_latency_ms_bucket[5m]))

# Mean predicted fraud probability (a shift can indicate concept drift)
rate(fraud_probability_sum[5m]) / rate(fraud_probability_count[5m])
```

## Drift Detection

A model trained on last quarter's transactions can degrade when spending patterns or fraud tactics change. `DriftDetector` (`src/drift_detector.py`) uses Evidently to compare a **reference** dataset (the data the model was trained on) with a **current** dataset (recent production data):

1. Evidently's `DataDriftPreset` runs a per-column statistical test, for example Kolmogorov–Smirnov for numerical features, to decide whether each feature's distribution has shifted.
2. `DatasetDriftMetric` flags **dataset drift** when the share of drifted columns crosses the threshold (default 50%).
3. An HTML report showing per-feature distributions is saved to `reports/drift_report.html`.
4. The detector returns `drift_detected`, `drift_share` and `report_path`.

### Champion/Challenger retraining flow (`src/retrain.py`)

```
Fetch champion metrics ──► Run drift check ──► No drift?  ──► Exit (keep champion)
                                    │
                                    ▼ drift detected
                         Train challenger (GradientBoosting)
                                    │
                                    ▼
                  Challenger F1 > Champion F1 ? ──► Yes ──► Move @champion alias to new version
                                    │
                                    └────────────► No  ──► Keep current champion
```

Promotion is an MLflow alias change, so the API picks up the new champion on its next restart with no code change. Rolling back means moving the alias back to the previous version.

F1 is the promotion metric because accuracy is misleading at a 0.172% fraud rate: a model that predicts "legitimate" for every transaction is 99.83% accurate. F1 balances precision (avoiding blocked legitimate customers) and recall (catching fraud).

## CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yaml`) runs on every push and pull request to `main`:

| Step | Action |
|---|---|
| Checkout | `actions/checkout@v3` |
| Set up Python | Python 3.11 |
| Install | scikit-learn, pandas, numpy, pytest |
| Test | `pytest tests/ -v` with `PYTHONPATH=.` |

The tests use **synthetic data**, so CI does not need the 150 MB Kaggle dataset or a running MLflow server. They cover the preprocessing steps, both trainers through the factory, and the end-to-end pipeline logic.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the Machine Learning Group, ULB

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraudulent | 492 (**0.172%**) |
| Period | 2 days, September 2013, European cardholders |
| Features | `V1`–`V28` (PCA-transformed for confidentiality), `Time`, `Amount` |
| Target | `Class` (1 = fraud, 0 = legitimate) |

To handle the class imbalance, the pipeline uses a stratified train/test split, `class_weight="balanced"` for RandomForest, and F1 / ROC-AUC / recall as evaluation metrics instead of accuracy.

> The dataset is not committed to this repository. Download it from Kaggle into `data/`.

## Roadmap

- Add Prometheus and Grafana services to `docker-compose.yml` with a provisioned dashboard
- Save the fitted scaler with the model so the API accepts raw `Amount` values
- Build the Docker image in CI and push it to a registry
- Schedule `retrain.py` to run automatically, for example with a cron-triggered GitHub Actions workflow
- Hot-reload the champion model in the API without a restart
