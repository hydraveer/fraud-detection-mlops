import pandas as pd
from src.data_loader import get_loader
from src.preprocessor import DataPreprocessor, ValidateSchema, DropMissingValues, ScaleFeatures, DropColumns

def make_sample_df():
    return pd.DataFrame({
        "Time": [0.0, 1.0, 2.0, 3.0, 4.0],
        "Amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "V1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Class": [0, 0, 0, 0, 1]
    })

def test_preprocessor_pipeline():
    df = make_sample_df()
    preprocessor = DataPreprocessor(steps=[
        ValidateSchema(required_columns=["Time", "Amount", "Class"]),
        DropMissingValues(),
        ScaleFeatures(columns=["Amount", "Time"]),
        DropColumns(columns=["Time"]),
    ])
    clean_df = preprocessor.run(df)
    assert "Time" not in clean_df.columns
    assert "Amount" in clean_df.columns
    assert clean_df.shape[0] == 5

def test_validate_schema_fails():
    df = pd.DataFrame({"wrong_col": [1, 2]})
    try:
        ValidateSchema(required_columns=["Time"]).run(df)
        assert False
    except ValueError:
        assert True