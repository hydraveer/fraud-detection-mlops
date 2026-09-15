"""Data preprocessing pipeline for fraud detection."""

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─── Abstract Step ─────────────────────────────────────────────────────────────
class PreprocessingStep(ABC):
    """Base interface all preprocessing steps must implement."""
    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply this step to the dataframe and return result."""
        ...

# ─── Concrete Steps ────────────────────────────────────────────────────────────

class DropMissingValues(PreprocessingStep):
    """Drop rows with missing values."""
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Dropping rows with missing values...")
        before = len(df)
        df = df.dropna()
        after = len(df)
        dropped = before - after
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values.")
        else:
            print("No missing values found.")
        return df

class ScaleFeatures(PreprocessingStep):
    """Scale specified columns using StandardScaler."""
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.scaler = StandardScaler()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in DataFrame.")
        df = df.copy()
        df[self.columns] = self.scaler.fit_transform(df[self.columns])
        print(f"ScaleFeatures: scaled columns {self.columns}")
        return df

class DropColumns(PreprocessingStep):
    """Drop columns that are not needed for training."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.columns if c in df.columns]
        df = df.drop(columns=existing)
        print(f"DropColumns: dropped {existing}")
        return df

class ValidateSchema(PreprocessingStep):
    """Validate that required columns exist in the dataframe."""

    def __init__(self, required_columns: list[str]) -> None:
        self.required_columns = required_columns

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Schema validation failed. Missing columns: {missing}\n"
                f"Expected: {self.required_columns}"
            )
        print(f"ValidateSchema: all required columns present")
        return df

# ─── Pipeline ──────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """Chains preprocessing steps sequentially."""

    def __init__(self, steps: list[PreprocessingStep]) -> None:
        self.steps = steps

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"\nStarting preprocessing pipeline ({len(self.steps)} steps)...")
        for i, step in enumerate(self.steps, 1):
            print(f"Step {i}/{len(self.steps)}: {step.__class__.__name__}")
            df = step.run(df)
        print(f"Pipeline complete. Final shape: {df.shape}\n")
        return df