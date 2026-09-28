import pandas as pd
import numpy as np
from src.trainer import get_trainer

def make_sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10), columns=[f"V{i}" for i in range(10)])
    y = pd.Series([0] * 90 + [1] * 10)
    return X, y

def test_random_forest_trainer():
    X, y = make_sample_data()
    trainer = get_trainer("random_forest", n_estimators=10, max_depth=3)
    model = trainer.train(X, y)
    metrics = trainer.evaluate(model, X, y)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert metrics["accuracy"] > 0.5

def test_unsupported_model():
    try:
        get_trainer("xgboost")
        assert False
    except ValueError:
        assert True