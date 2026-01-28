"""Tests for data loading functionality."""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader


@pytest.fixture
def data_path():
    """Path to test data file."""
    return Path(__file__).parent.parent / 'data' / 'TRDataChallenge2023.txt'


@pytest.fixture
def loader(data_path):
    """DataLoader instance."""
    return DataLoader(data_path)


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_data_file_exists(self, data_path):
        """Verify data file exists."""
        assert data_path.exists(), f"Data file not found: {data_path}"
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
    
    def test_load_returns_dataframe(self, loader):
        """Test load() returns a DataFrame."""
        df = loader.load()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "DataFrame is empty"
    
    def test_dataframe_has_required_columns(self, loader):
        """Test DataFrame has expected columns."""
        df = loader.load()
        required_columns = ['documentId', 'postures', 'sections']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_postures_are_lists(self, loader):
        """Test postures column contains lists."""
        df = loader.load()
        # Check first non-empty posture
        for postures in df['postures']:
            assert isinstance(postures, list), f"Postures should be list, got {type(postures)}"
            break
    
    def test_document_count(self, loader):
        """Test we have expected number of documents."""
        df = loader.load()
        # Should have ~18K documents based on challenge description
        assert len(df) > 10000, f"Expected >10K documents, got {len(df)}"
        assert len(df) < 25000, f"Expected <25K documents, got {len(df)}"
    
    def test_get_schema(self, loader):
        """Test get_schema returns column types."""
        _ = loader.load()
        schema = loader.get_schema()
        assert isinstance(schema, dict)
        assert len(schema) > 0
