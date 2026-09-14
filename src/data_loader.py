"""
Data loading with factory pattern.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import os
import pandas as pd

# ─── Abstract Interface ────────────────────────────────────────────────────────
class DataLoader(ABC):
    """
    Abstract interface for data loaders.
    """
    @abstractmethod # type: ignore
    def load(self, path: str) -> pd.DataFrame:
        """Load data from path and return DataFrame."""
        ...

    @abstractmethod # type: ignore
    def validate(self, path: str) -> None:
        """Validate file exists and is correct format. Raise ValueError if not."""
        ...

# ─── Concrete Loaders ──────────────────────────────────────────────────────────


class CSVLoader(DataLoader):
    """
    Concrete implementation of DataLoader for CSV files.
    """
    Extension = ".csv"

    def validate(self, path: str) -> None:
        """
        Validate that the file exists and is a CSV file.
        """
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
        if not path.endswith(self.Extension):
            raise FileNotFoundError(f"File not found: {path}")

    def load(self, path: str) -> pd.DataFrame:
        """
        Load data from a CSV file and return a DataFrame.
        """
        self.validate(path)
        print(f"Loading data from {path}...")
        df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df


class ParquetLoader(DataLoader):
    """
    Concrete implementation of DataLoader for Parquet files.
    """
    Extension = ".parquet"

    def validate(self, path: str) -> None:
        """
        Validate that the file exists and is a Parquet file.
        """
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
        if not path.endswith(self.Extension):
            raise FileNotFoundError(f"File not found: {path}")

    def load(self, path: str) -> pd.DataFrame:
        """
        Load data from a Parquet file and return a DataFrame.
        """
        self.validate(path)
        print(f"Loading data from {path}...")
        df = pd.read_parquet(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df


class JSONLoader(DataLoader):
    """
    Concrete implementation of DataLoader for JSON files.
    """
    Extension = ".json"

    def validate(self, path: str) -> None:
        """
        Validate that the file exists and is a JSON file.
        """
        if not os.path.exists(path):
            raise ValueError(f"File {path} does not exist.")
        if not path.endswith(self.Extension):
            raise FileNotFoundError(f"File not found: {path}")

    def load(self, path: str) -> pd.DataFrame:
        """
        Load data from a JSON file and return a DataFrame.
        """
        self.validate(path)
        print(f"Loading data from {path}...")
        df = pd.read_json(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df

# ─── Factory ───────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".csv": CSVLoader,
    ".parquet": ParquetLoader,
    ".json": JSONLoader,
}

UNSUPPORTED_MESSAGE = """
Unsupported file format: {ext}

Supported formats:
  .csv     → CSV files
  .parquet → Parquet files
  .json    → JSON files

If your file is:
  .zip  → extract it first: unzip your_file.zip
  .xlsx → convert to CSV: pd.read_excel(...).to_csv(...)
  .tsv  → rename to .csv and set sep='\\t'
"""

def get_loader(path: str) -> DataLoader:
    """Factory method — returns correct loader based on file extension."""
    ext = os.path.splitext(path)[-1].lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(UNSUPPORTED_MESSAGE.format(ext=ext))

    loader_class = SUPPORTED_FORMATS[ext]
    return loader_class()