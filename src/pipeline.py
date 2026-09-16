"""End-to-end training pipeline with MLflow tracking."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import get_loader
from src.preprocessor import (
    DataPreprocessor,
    DropColumns,
    DropMissingValues,
    ScaleFeatures,
    ValidateSchema,
)
from src.trainer import get_trainer


# ─── Pipeline Config ───────────────────────────────────────────────────────────

class PipelineConfig:
    """All pipeline settings in one place."""

    # Data
    data_path: str = "data/creditcard.csv"
    nrows: int | None = 10000  # None = load all rows
    random_seed: int = 42

    # Preprocessing
    required_columns: list[str] = ["Time", "Amount", "Class"]
    scale_columns: list[str] = ["Amount", "Time"]
    drop_columns: list[str] = ["Time"]
    target_column: str = "Class"

    # Training
    model_type: str = "random_forest"
    model_params: dict = {
        "n_estimators": 10,
        "max_depth": 3,
    }
    test_size: float = 0.2
    random_state: int = 42

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "fraud-detection"
    model_registry_name: str = "FraudDetector"


# ─── Pipeline ──────────────────────────────────────────────────────────────────

class TrainingPipeline:
    """Runs full training pipeline end to end."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> dict[str, float]:
        cfg = self.config

        # Setup MLflow
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment)

        with mlflow.start_run() as run:
            print(f"\nMLflow Run ID: {run.info.run_id}")

            # Step 1: Load data
            print("\n--- Step 1: Loading Data ---")
            loader = get_loader(cfg.data_path)
            df = loader.load(cfg.data_path, nrows=cfg.nrows)
            df = df.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)

            # Step 2: Preprocess
            print("\n--- Step 2: Preprocessing ---")
            preprocessor = DataPreprocessor(steps=[
                ValidateSchema(required_columns=cfg.required_columns),
                DropMissingValues(),
                ScaleFeatures(columns=cfg.scale_columns),
                DropColumns(columns=cfg.drop_columns),
            ])
            df = preprocessor.run(df)

            # Step 3: Split
            print("\n--- Step 3: Splitting Data ---")
            X = df.drop(cfg.target_column, axis=1)
            y = df[cfg.target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=cfg.test_size, 
                random_state=cfg.random_state,
                stratify=y,
            )
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            print(f"Fraud in train: {y_train.sum()}, Fraud in test: {y_test.sum()}")

            # Step 4: Train
            print("\n--- Step 4: Training ---")
            trainer = get_trainer(cfg.model_type, **cfg.model_params)
            model = trainer.train(X_train, y_train)

            # Step 5: Evaluate
            print("\n--- Step 5: Evaluating ---")
            metrics = trainer.evaluate(model, X_test, y_test)
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # Step 6: Log to MLflow
            print("\n--- Step 6: Logging to MLflow ---")
            mlflow.log_params(trainer.get_params())
            mlflow.log_params({
                "data_path": cfg.data_path,
                "nrows": cfg.nrows,
                "test_size": cfg.test_size,
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "fraud-detector")
            print("Logged to MLflow successfully")

            # Step 7: Register model
            print("\n--- Step 7: Registering Model ---")
            model_uri = f"runs:/{run.info.run_id}/fraud-detector"
            registered = mlflow.register_model(model_uri, cfg.model_registry_name)
            print(f"Model registered: {cfg.model_registry_name} v{registered.version}")

            return metrics


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = TrainingPipeline(config)
    metrics = pipeline.run()

    print("\n=== Pipeline Complete ===")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")