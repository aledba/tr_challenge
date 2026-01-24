"""
Data loading utilities for TR Data Challenge.
Loads JSON Lines format into pandas DataFrame.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class DataLoader:
    """
    Loads JSON Lines data into a pandas DataFrame.
    50MB is trivial for modern machines - just load it all.
    """
    
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        self._df: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """Loads the JSON Lines file into a DataFrame."""
        if self._df is None:
            self._df = pd.read_json(self.file_path, lines=True)
        return self._df
    
    @property
    def df(self) -> pd.DataFrame:
        """Cached DataFrame property."""
        return self.load()
    
    def peek(self, n: int = 5) -> pd.DataFrame:
        """Returns first n rows."""
        return self.df.head(n)
    
    def get_schema(self) -> dict:
        """Returns column names and dtypes."""
        return self.df.dtypes.to_dict()
    
    def get_column_names(self) -> list[str]:
        """Returns list of column names."""
        return list(self.df.columns)
