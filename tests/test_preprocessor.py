from src.data_loader import get_loader
from src.preprocessor import (
    DataPreprocessor,
    DropMissingValues,
    ScaleFeatures,
    DropColumns,
    ValidateSchema,
)

# Load only 10000 rows directly
import pandas as pd
df = pd.read_csv("data/creditcard.csv", nrows=10000)
print("Loaded:", df.shape)

# Build pipeline
preprocessor = DataPreprocessor(steps=[
    ValidateSchema(required_columns=["Time", "Amount", "Class"]),
    DropMissingValues(),
    ScaleFeatures(columns=["Amount", "Time"]),
    DropColumns(columns=["Time"]),
])

# Run pipeline
clean_df = preprocessor.run(df)

print("Clean data shape:", clean_df.shape)
print("Columns:", clean_df.columns.tolist())