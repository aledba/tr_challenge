"""
Model training utilities for TR Data Challenge.
Handles data preparation and model training for multi-label classification.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from .data_loader import DataLoader


@dataclass
class PreparedData:
    """Container for prepared training/validation/test data."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    
    # For interpretability
    train_texts: list[str] = field(default_factory=list)
    val_texts: list[str] = field(default_factory=list)
    test_texts: list[str] = field(default_factory=list)
    
    # Label info
    label_names: list[str] = field(default_factory=list)
    mlb: Optional[MultiLabelBinarizer] = None
    vectorizer: Optional[TfidfVectorizer] = None
    
    def summary(self) -> str:
        """Returns a formatted summary string."""
        n_train = self.X_train.shape[0]
        n_val = self.X_val.shape[0]
        n_test = self.X_test.shape[0]
        n_features = self.X_train.shape[1] if len(self.X_train.shape) > 1 else 'N/A'
        
        lines = [
            "=" * 50,
            "PREPARED DATA SUMMARY",
            "=" * 50,
            f"Train samples:     {n_train:,}",
            f"Validation samples:{n_val:,}",
            f"Test samples:      {n_test:,}",
            f"Number of labels:  {len(self.label_names)}",
            f"Feature dimension: {n_features}",
            "=" * 50,
        ]
        return "\n".join(lines)


class TextExtractor:
    """
    Extracts flat text from nested document structure.
    Handles the sections[].paragraphs[] format.
    """
    
    @staticmethod
    def extract_text(sections: list) -> str:
        """Flattens nested sections/paragraphs into single text."""
        if not isinstance(sections, list):
            return str(sections)
        
        all_text = []
        for section in sections:
            if isinstance(section, dict):
                # Handle section with paragraphs
                if 'paragraphs' in section:
                    for para in section['paragraphs']:
                        all_text.append(str(para))
                # Handle other dict structures
                elif 'text' in section:
                    all_text.append(str(section['text']))
            elif isinstance(section, str):
                all_text.append(section)
        
        return " ".join(all_text)
    
    @staticmethod
    def extract_all(df: pd.DataFrame, col: str = 'sections') -> list[str]:
        """Extracts text from all documents."""
        return [TextExtractor.extract_text(row) for row in df[col]]


class DataPreparer:
    """
    Prepares data for multi-label classification.
    
    - Extracts flat text from nested structure
    - Encodes multi-labels as binary matrix
    - Splits into train/val/test
    - Vectorizes text with TF-IDF
    """
    
    def __init__(
        self,
        loader: DataLoader,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        min_label_count: Optional[int] = None,
    ):
        self.loader = loader
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.min_label_count = min_label_count
        
        self._mlb: Optional[MultiLabelBinarizer] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
    
    def _get_viable_labels(self, df: pd.DataFrame, posture_col: str) -> set[str]:
        """Returns labels with sufficient samples for training."""
        if self.min_label_count is None:
            # Use all labels
            all_labels = set()
            for labels in df[posture_col]:
                all_labels.update(labels)
            return all_labels
        
        # Count each label
        from collections import Counter
        label_counts = Counter()
        for labels in df[posture_col]:
            label_counts.update(labels)
        
        return {label for label, count in label_counts.items() 
                if count >= self.min_label_count}
    
    def prepare(
        self,
        text_col: str = 'sections',
        posture_col: str = 'postures',
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
    ) -> PreparedData:
        """
        Full data preparation pipeline.
        
        Returns PreparedData with TF-IDF features and binary labels.
        """
        df = self.loader.df.copy()
        
        # Extract flat text
        texts = TextExtractor.extract_all(df, text_col)
        labels = df[posture_col].tolist()
        
        # Filter to viable labels if specified
        if self.min_label_count is not None:
            viable = self._get_viable_labels(df, posture_col)
            labels = [[l for l in doc_labels if l in viable] for doc_labels in labels]
            # Remove docs with no viable labels
            mask = [len(l) > 0 for l in labels]
            texts = [t for t, m in zip(texts, mask) if m]
            labels = [l for l, m in zip(labels, mask) if m]
        
        # Encode labels
        self._mlb = MultiLabelBinarizer()
        y = self._mlb.fit_transform(labels)
        
        # Train/val/test split (stratified is tricky for multi-label, use random)
        X_temp, X_test, y_temp, y_test, texts_temp, texts_test = train_test_split(
            texts, y, texts,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Adjust val_size for remaining data
        adjusted_val_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, texts_train, texts_val = train_test_split(
            X_temp, y_temp, texts_temp,
            test_size=adjusted_val_size,
            random_state=self.random_state
        )
        
        # Vectorize with TF-IDF
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
        )
        
        X_train_vec = self._vectorizer.fit_transform(X_train)
        X_val_vec = self._vectorizer.transform(X_val)
        X_test_vec = self._vectorizer.transform(X_test)
        
        return PreparedData(
            X_train=X_train_vec,
            X_val=X_val_vec,
            X_test=X_test_vec,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_texts=texts_train,
            val_texts=texts_val,
            test_texts=texts_test,
            label_names=list(self._mlb.classes_),
            mlb=self._mlb,
            vectorizer=self._vectorizer,
        )


class BaselineTrainer:
    """
    TF-IDF + Logistic Regression baseline for multi-label classification.
    Uses OneVsRestClassifier for multi-label support.
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter
        self.model: Optional[OneVsRestClassifier] = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaselineTrainer':
        """Trains the baseline model."""
        base_clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',
            class_weight='balanced',
            n_jobs=-1,
        )
        self.model = OneVsRestClassifier(base_clf, n_jobs=-1)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability scores."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
