"""Tests for model evaluation functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_evaluator import MultiLabelEvaluator, EvaluationResults


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""
    
    def test_summary_returns_string(self):
        """Test summary() returns formatted string."""
        results = EvaluationResults(
            f1_micro=0.75,
            f1_macro=0.65,
            f1_weighted=0.70,
            f1_samples=0.68,
            precision_micro=0.80,
            precision_macro=0.75,
            recall_micro=0.70,
            recall_macro=0.65,
            hamming_loss=0.05,
            exact_match_ratio=0.45,
            per_class_metrics=pd.DataFrame({
                'label': ['A', 'B'],
                'f1': [0.8, 0.6],
                'precision': [0.85, 0.65],
                'recall': [0.75, 0.55],
                'support': [100, 50]
            }),
            label_names=['A', 'B']
        )
        summary = results.summary()
        assert isinstance(summary, str)
        assert '0.75' in summary  # f1_micro
    
    def test_get_top_classes(self):
        """Test getting top performing classes."""
        results = EvaluationResults(
            f1_micro=0.75,
            f1_macro=0.65,
            f1_weighted=0.70,
            f1_samples=0.68,
            precision_micro=0.80,
            precision_macro=0.75,
            recall_micro=0.70,
            recall_macro=0.65,
            hamming_loss=0.05,
            exact_match_ratio=0.45,
            per_class_metrics=pd.DataFrame({
                'label': ['A', 'B', 'C'],
                'f1': [0.9, 0.7, 0.5],
                'precision': [0.85, 0.65, 0.55],
                'recall': [0.95, 0.75, 0.45],
                'support': [100, 50, 30]
            }),
            label_names=['A', 'B', 'C']
        )
        top = results.get_top_classes(2, 'f1')
        assert len(top) == 2
        assert top.iloc[0]['label'] == 'A'
    
    def test_get_bottom_classes(self):
        """Test getting bottom performing classes."""
        results = EvaluationResults(
            f1_micro=0.75,
            f1_macro=0.65,
            f1_weighted=0.70,
            f1_samples=0.68,
            precision_micro=0.80,
            precision_macro=0.75,
            recall_micro=0.70,
            recall_macro=0.65,
            hamming_loss=0.05,
            exact_match_ratio=0.45,
            per_class_metrics=pd.DataFrame({
                'label': ['A', 'B', 'C'],
                'f1': [0.9, 0.7, 0.5],
                'precision': [0.85, 0.65, 0.55],
                'recall': [0.95, 0.75, 0.45],
                'support': [100, 50, 30]
            }),
            label_names=['A', 'B', 'C']
        )
        bottom = results.get_bottom_classes(2, 'f1')
        assert len(bottom) == 2
        assert bottom.iloc[0]['label'] == 'C'


class TestMultiLabelEvaluator:
    """Tests for MultiLabelEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator with sample labels."""
        return MultiLabelEvaluator(['ClassA', 'ClassB', 'ClassC'])
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction data."""
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        y_pred = np.array([
            [1, 0, 1],  # Perfect
            [0, 1, 1],  # 1 FP
            [1, 0, 0],  # 1 FN
            [0, 0, 1],  # Perfect
        ])
        return y_true, y_pred
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes with labels."""
        assert evaluator.label_names == ['ClassA', 'ClassB', 'ClassC']
    
    def test_evaluate_returns_results(self, evaluator, sample_data):
        """Test evaluate returns EvaluationResults."""
        y_true, y_pred = sample_data
        results = evaluator.evaluate(y_true, y_pred)
        
        assert isinstance(results, EvaluationResults)
        assert 0 <= results.f1_micro <= 1
        assert 0 <= results.f1_macro <= 1
    
    def test_evaluate_perfect_predictions(self, evaluator):
        """Test evaluation with perfect predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[1, 0, 1], [0, 1, 0]])
        
        results = evaluator.evaluate(y_true, y_pred)
        assert results.f1_micro == 1.0
        assert results.precision_micro == 1.0
        assert results.recall_micro == 1.0
    
    def test_evaluate_all_wrong(self, evaluator):
        """Test evaluation with all wrong predictions."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0, 1, 1], [1, 0, 1]])
        
        results = evaluator.evaluate(y_true, y_pred)
        assert results.f1_micro == 0.0
    
    def test_per_class_metrics_shape(self, evaluator, sample_data):
        """Test per-class metrics have correct shape."""
        y_true, y_pred = sample_data
        results = evaluator.evaluate(y_true, y_pred)
        
        assert len(results.per_class_metrics) == 3
        assert 'label' in results.per_class_metrics.columns
        assert 'f1' in results.per_class_metrics.columns
    
    def test_feasibility_analysis(self, evaluator, sample_data):
        """Test feasibility analysis method."""
        y_true, y_pred = sample_data
        results = evaluator.evaluate(y_true, y_pred)
        
        feasibility = results.get_feasibility_analysis(
            human_kappa_low=0.5,
            human_kappa_high=0.9
        )
        
        assert isinstance(feasibility, pd.DataFrame)
        assert 'automation_feasible' in feasibility.columns
        assert 'high_confidence' in feasibility.columns
        assert 'needs_review' in feasibility.columns
