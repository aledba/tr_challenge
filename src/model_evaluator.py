"""
Model evaluation utilities for TR Data Challenge.
Computes multi-label metrics and per-class performance analysis.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    hamming_loss, accuracy_score, classification_report,
    cohen_kappa_score
)


@dataclass
class EvaluationResults:
    """Container for multi-label evaluation metrics."""
    
    # Overall metrics
    f1_micro: float
    f1_macro: float
    f1_weighted: float
    f1_samples: float
    
    precision_micro: float
    precision_macro: float
    recall_micro: float
    recall_macro: float
    
    hamming_loss: float
    exact_match_ratio: float  # "Subset accuracy"
    
    # Per-class results
    per_class_metrics: pd.DataFrame
    
    # Label info
    label_names: list[str]
    
    def summary(self) -> str:
        """Returns a formatted summary string."""
        lines = [
            "=" * 60,
            "MULTI-LABEL EVALUATION RESULTS",
            "=" * 60,
            "",
            "Overall Metrics:",
            f"  F1 Micro:          {self.f1_micro:.4f}",
            f"  F1 Macro:          {self.f1_macro:.4f}",
            f"  F1 Weighted:       {self.f1_weighted:.4f}",
            f"  F1 Samples:        {self.f1_samples:.4f}",
            "",
            f"  Precision (micro): {self.precision_micro:.4f}",
            f"  Precision (macro): {self.precision_macro:.4f}",
            f"  Recall (micro):    {self.recall_micro:.4f}",
            f"  Recall (macro):    {self.recall_macro:.4f}",
            "",
            f"  Hamming Loss:      {self.hamming_loss:.4f}",
            f"  Exact Match Ratio: {self.exact_match_ratio:.4f}",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def get_top_classes(self, n: int = 10, metric: str = 'f1') -> pd.DataFrame:
        """Returns top n classes by specified metric."""
        return self.per_class_metrics.nlargest(n, metric)
    
    def get_bottom_classes(self, n: int = 10, metric: str = 'f1') -> pd.DataFrame:
        """Returns bottom n classes by specified metric."""
        return self.per_class_metrics.nsmallest(n, metric)
    
    def get_feasibility_analysis(
        self,
        human_kappa_low: float = 0.63,
        human_kappa_high: float = 0.93,
    ) -> pd.DataFrame:
        """
        Analyzes which postures are feasible for automation.
        
        Compares model performance to human annotator agreement range.
        """
        df = self.per_class_metrics.copy()
        
        # F1 approximates Kappa for balanced classes
        # Use it as proxy for automation feasibility
        df['automation_feasible'] = df['f1'] >= human_kappa_low
        df['high_confidence'] = df['f1'] >= human_kappa_high
        df['needs_review'] = (df['f1'] >= human_kappa_low * 0.8) & (df['f1'] < human_kappa_low)
        
        return df[['label', 'f1', 'support', 'automation_feasible', 'high_confidence', 'needs_review']]


class MultiLabelEvaluator:
    """
    Evaluates multi-label classification models.
    
    Computes both overall and per-class metrics for
    detailed performance analysis.
    """
    
    def __init__(self, label_names: list[str]):
        self.label_names = label_names
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationResults:
        """
        Computes comprehensive multi-label metrics.
        
        Args:
            y_true: Ground truth binary labels (n_samples, n_labels)
            y_pred: Predicted binary labels (n_samples, n_labels)
        
        Returns:
            EvaluationResults with overall and per-class metrics
        """
        # Overall metrics
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        h_loss = hamming_loss(y_true, y_pred)
        
        # Exact match ratio (all labels must match)
        exact_match = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        per_class = self._compute_per_class_metrics(y_true, y_pred)
        
        return EvaluationResults(
            f1_micro=f1_micro,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            f1_samples=f1_samples,
            precision_micro=precision_micro,
            precision_macro=precision_macro,
            recall_micro=recall_micro,
            recall_macro=recall_macro,
            hamming_loss=h_loss,
            exact_match_ratio=exact_match,
            per_class_metrics=per_class,
            label_names=self.label_names,
        )
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Computes metrics for each label individually."""
        records = []
        
        for i, label in enumerate(self.label_names):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            support = int(y_true_i.sum())
            pred_count = int(y_pred_i.sum())
            
            # Skip labels with no support
            if support == 0:
                records.append({
                    'label': label,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'support': support,
                    'predicted_count': pred_count,
                })
                continue
            
            precision = precision_score(y_true_i, y_pred_i, zero_division=0)
            recall = recall_score(y_true_i, y_pred_i, zero_division=0)
            f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
            
            records.append({
                'label': label,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'predicted_count': pred_count,
            })
        
        df = pd.DataFrame(records)
        return df.sort_values('support', ascending=False).reset_index(drop=True)


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """
    Analyzes how different probability thresholds affect metrics.
    
    Useful for finding optimal threshold per class.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    records = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        records.append({
            'threshold': thresh,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
        })
    
    return pd.DataFrame(records)


def create_classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    """
    Creates a DataFrame from sklearn's classification report.
    Useful for detailed per-class analysis.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(report).T
    df = df.reset_index().rename(columns={'index': 'label'})
    
    # Filter out aggregate rows for per-class analysis
    per_class = df[~df['label'].isin(['micro avg', 'macro avg', 'weighted avg', 'samples avg'])]
    
    return per_class
