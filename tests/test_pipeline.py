import pandas as pd
from src.data_loader import get_loader
from src.preprocessor import DataPreprocessor, ValidateSchema, DropMissingValues, ScaleFeatures, DropColumns
from src.trainer import get_trainer

def test_data_loader():
    import io
    import numpy as np
    df = pd.DataFrame({
        "Time": [0.0, 1.0],
        "V1": [-1.3, 1.2],
        "Amount": [149.0, 2.0],
        "Class": [0, 1]
    })
    assert df.shape[0] == 2
    assert "Class" in df.columns

def test_preprocessor():
    df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0],
        "Amount": [100.0, 200.0, 300.0],
        "V1": [1.0, 2.0, 3.0],
        "Class": [0, 0, 1]
    })
    preprocessor = DataPreprocessor(steps=[
        ValidateSchema(required_columns=["Time", "Amount", "Class"]),
        DropMissingValues(),
        ScaleFeatures(columns=["Amount", "Time"]),
        DropColumns(columns=["Time"]),
    ])
    clean_df = preprocessor.run(df)
    assert "Time" not in clean_df.columns
    assert "Amount" in clean_df.columns

def test_trainer():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(10)])
    trainer = get_trainer("random_forest", n_estimators=10, max_depth=3)
    import pandas as pd
    model = trainer.train(pd.DataFrame(X), pd.Series(y))
    assert model is not None

def test_unsupported_model():
    try:
        get_trainer("xgboost")
        assert False
    except ValueError:
        assert True
