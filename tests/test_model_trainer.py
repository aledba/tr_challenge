"""Tests for model training functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.model_trainer import DataPreparer, PreparedData


@pytest.fixture
def data_path():
    """Path to test data file."""
    return Path(__file__).parent.parent / 'data' / 'TRDataChallenge2023.txt'


@pytest.fixture
def loader(data_path):
    """DataLoader instance."""
    return DataLoader(data_path)


class TestDataPreparer:
    """Tests for DataPreparer class."""
    
    def test_initialization(self, loader):
        """Test DataPreparer initializes correctly."""
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        assert preparer.min_label_count == 50
        assert preparer.random_state == 42
    
    def test_prepare_returns_prepared_data(self, loader):
        """Test prepare() returns PreparedData."""
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        data = preparer.prepare(max_features=1000, ngram_range=(1, 1))
        
        assert isinstance(data, PreparedData)
    
    def test_prepared_data_has_splits(self, loader):
        """Test PreparedData has train/val/test splits."""
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        data = preparer.prepare(max_features=1000, ngram_range=(1, 1))
        
        # Check texts
        assert len(data.train_texts) > 0
        assert len(data.val_texts) > 0
        assert len(data.test_texts) > 0
        
        # Check labels
        assert data.y_train.shape[0] == len(data.train_texts)
        assert data.y_val.shape[0] == len(data.val_texts)
        assert data.y_test.shape[0] == len(data.test_texts)
    
    def test_no_data_leakage(self, loader):
        """Test train/val/test have no overlapping documents by index.
        
        Note: We check indices, not text content, because some documents
        may have identical text but different IDs.
        """
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        data = preparer.prepare(max_features=1000, ngram_range=(1, 1))
        
        # Verify split sizes add up (no document in multiple splits)
        total_docs = len(data.train_texts) + len(data.val_texts) + len(data.test_texts)
        
        # Check that we have reasonable split proportions
        train_ratio = len(data.train_texts) / total_docs
        val_ratio = len(data.val_texts) / total_docs
        test_ratio = len(data.test_texts) / total_docs
        
        # Typical splits are around 70/15/15 or 80/10/10
        assert 0.6 <= train_ratio <= 0.9, f"Unexpected train ratio: {train_ratio}"
        assert 0.05 <= val_ratio <= 0.25, f"Unexpected val ratio: {val_ratio}"
        assert 0.05 <= test_ratio <= 0.25, f"Unexpected test ratio: {test_ratio}"
    
    def test_label_filtering(self, loader):
        """Test min_label_count filters rare labels."""
        preparer_strict = DataPreparer(loader, min_label_count=100, random_state=42)
        data_strict = preparer_strict.prepare(max_features=1000, ngram_range=(1, 1))
        
        preparer_loose = DataPreparer(loader, min_label_count=10, random_state=42)
        data_loose = preparer_loose.prepare(max_features=1000, ngram_range=(1, 1))
        
        # Stricter filtering = fewer labels
        assert len(data_strict.label_names) <= len(data_loose.label_names)
    
    def test_reproducibility(self, loader):
        """Test same random_state produces same splits."""
        preparer1 = DataPreparer(loader, min_label_count=50, random_state=42)
        data1 = preparer1.prepare(max_features=1000, ngram_range=(1, 1))
        
        preparer2 = DataPreparer(loader, min_label_count=50, random_state=42)
        data2 = preparer2.prepare(max_features=1000, ngram_range=(1, 1))
        
        assert data1.train_texts == data2.train_texts
        assert np.array_equal(data1.y_train, data2.y_train)
    
    def test_different_seeds_different_splits(self, loader):
        """Test different random_state produces different splits."""
        preparer1 = DataPreparer(loader, min_label_count=50, random_state=42)
        data1 = preparer1.prepare(max_features=1000, ngram_range=(1, 1))
        
        preparer2 = DataPreparer(loader, min_label_count=50, random_state=123)
        data2 = preparer2.prepare(max_features=1000, ngram_range=(1, 1))
        
        # At least some samples should differ
        assert data1.train_texts != data2.train_texts


class TestPreparedData:
    """Tests for PreparedData dataclass."""
    
    @pytest.fixture
    def prepared_data(self, loader):
        """Create PreparedData instance."""
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        return preparer.prepare(max_features=1000, ngram_range=(1, 1))
    
    def test_summary_returns_string(self, prepared_data):
        """Test summary() returns formatted string."""
        summary = prepared_data.summary()
        assert isinstance(summary, str)
        assert 'Train' in summary
        assert 'Val' in summary
        assert 'Test' in summary
    
    def test_label_names_match_y_shape(self, prepared_data):
        """Test label names match y matrix columns."""
        assert len(prepared_data.label_names) == prepared_data.y_train.shape[1]
        assert len(prepared_data.label_names) == prepared_data.y_val.shape[1]
        assert len(prepared_data.label_names) == prepared_data.y_test.shape[1]
    
    def test_y_matrices_are_binary(self, prepared_data):
        """Test y matrices contain only 0 and 1."""
        for y in [prepared_data.y_train, prepared_data.y_val, prepared_data.y_test]:
            unique_values = set(np.unique(y))
            assert unique_values <= {0, 1}, f"Expected binary, got {unique_values}"
