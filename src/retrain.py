"""Champion/Challenger retraining pipeline for FraudDetector."""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running as a script (python src/retrain.py) as well as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import mlflow.sklearn

from src.pipeline import PipelineConfig, TrainingPipeline

MLFLOW_TRACKING_URI = "http://localhost:5001"
MODEL_REGISTRY_NAME = "FraudDetector"

def get_champion_metrics() -> dict[str, float]:
    """Get current champion's metrics from MLflow."""
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, "champion")
    run = client.get_run(champion.run_id)
    return {
        "f1": float(run.data.metrics.get("f1", 0.0)),
        "roc_auc": float(run.data.metrics.get("roc_auc", 0.0)),
        "recall": float(run.data.metrics.get("recall", 0.0)),
    }

def promote_if_better(
    new_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    new_version: str,
) -> bool:
    """Promote new model if F1 strictly better than champion."""
    client = mlflow.MlflowClient()
    print(f"\nChampion F1:  {champion_metrics['f1']:.4f}")
    print(f"Challenger F1: {new_metrics['f1']:.4f}")
    if new_metrics["f1"] > champion_metrics["f1"]:
        client.set_registered_model_alias(
            MODEL_REGISTRY_NAME, "champion", new_version
        )
        print(f"New champion: version {new_version} promoted!")
        return True
    else:
        print("Champion retained. Challenger did not beat champion.")
        return False

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Step 1: Get champion metrics
    print("=== Getting Champion Metrics ===")
    champion_metrics = get_champion_metrics()
    print(f"Champion F1:     {champion_metrics['f1']:.4f}")
    print(f"Champion ROC AUC: {champion_metrics['roc_auc']:.4f}")
    print(f"Champion Recall:  {champion_metrics['recall']:.4f}")

    # Step 2: Train challenger with different config
    print("\n=== Training Challenger ===")
    config = PipelineConfig()
    config.model_type = "gradient_boosting"
    config.model_params = {
        "n_estimators": 50,
        "max_depth": 4,
    }


    pipepline = TrainingPipeline(config)
    new_metrics = pipepline.run()


    # Step 3: Get new version number
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
    latest_version = max(versions, key=lambda v: int(v.version))

    # Step 4: Compare and promote
    print("\n=== Champion vs Challenger ===")
    promoted = promote_if_better(new_metrics, champion_metrics, latest_version.version)

    print("\n=== Retraining Complete ===")
    print(f"Promoted: {promoted}")
