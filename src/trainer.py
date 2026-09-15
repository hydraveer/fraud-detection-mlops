"""Model training with factory pattern."""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ─── Abstract Interface ────────────────────────────────────────────────────────

class BaseTrainer(ABC):
    """Base interface all trainers must implement."""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        """Train model and return trained model object."""
        ...

    @abstractmethod
    def evaluate(self, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate model and return metrics dict."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return model parameters for MLflow logging."""
        ...

# ─── Concrete Trainers ─────────────────────────────────────────────────────────

class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer."""
    def __init__(self, n_estimators: int = 100, max_depth: int = 10) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def get_params(self) -> dict:
        return {
            "model_type": "random_forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": "balanced",
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        print(f"Training RandomForest (n_estimators={self.n_estimators}, max_depth={self.max_depth})...")
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        print("RandomForest training completed.")
        return model

    def evaluate(self, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability for the positive class
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }


class GradientBoostingTrainer(BaseTrainer):
    """Gradient Boosting trainer."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 5) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def get_params(self) -> dict:
        return {
            "model_type": "gradient_boosting",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> object:
        print(f"Training GradientBoosting (n_estimators={self.n_estimators}, max_depth={self.max_depth})...")
        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        print("Training complete.")
        return model

    def evaluate(self, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

# ─── Factory ───────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {
    "random_forest": RandomForestTrainer,
    "gradient_boosting": GradientBoostingTrainer,
}

def get_trainer(model_type: str, **kwargs) -> BaseTrainer:
    """Factory method — returns correct trainer based on model type."""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}\n"
            f"Supported models: {list(SUPPORTED_MODELS.keys())}"
        )
    trainer_class = SUPPORTED_MODELS[model_type]
    return trainer_class(**kwargs)