"""
Data analysis utilities for TR Data Challenge.
Computes statistics from pandas DataFrame.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .data_loader import DataLoader


@dataclass
class DatasetStatistics:
    """Container for dataset-level statistics."""
    num_documents: int
    num_postures: int
    total_paragraphs: int
    posture_distribution: pd.Series
    
    # Text statistics
    avg_paragraphs_per_doc: float
    avg_words_per_doc: float
    min_words: int
    max_words: int
    median_words: float
    
    # Schema info
    schema: dict
    
    def summary(self) -> str:
        """Returns a formatted summary string."""
        lines = [
            "=" * 60,
            "DATASET STATISTICS",
            "=" * 60,
            f"Number of documents:     {self.num_documents:,}",
            f"Number of postures:      {self.num_postures}",
            f"Total paragraphs:        {self.total_paragraphs:,}",
            "",
            "Text Statistics:",
            f"  Avg paragraphs/doc:    {self.avg_paragraphs_per_doc:.2f}",
            f"  Avg words/doc:         {self.avg_words_per_doc:.2f}",
            f"  Min words:             {self.min_words:,}",
            f"  Max words:             {self.max_words:,}",
            f"  Median words:          {self.median_words:.0f}",
            "",
            f"Top 10 Postures:",
        ]
        
        for posture, count in self.posture_distribution.head(10).items():
            pct = (count / self.num_documents) * 100
            lines.append(f"  {str(posture)[:45]:<45} {count:>6} ({pct:>5.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class DatasetAnalyzer:
    """
    Analyzes document datasets using pandas.
    Works directly with DataFrames for simplicity.
    """
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._stats: Optional[DatasetStatistics] = None
        
        # Will be set after inspecting data
        self._posture_col: Optional[str] = None
        self._paragraphs_col: Optional[str] = None
        self._text_col: Optional[str] = None
    
    def _detect_columns(self, df: pd.DataFrame) -> None:
        """Auto-detect which columns contain posture, paragraphs, text."""
        df_cols = list(df.columns)
        
        # Find posture column (check plural forms too)
        for candidate in ['postures', 'posture', 'labels', 'label', 'category', 'procedural_posture']:
            matches = [c for c in df_cols if c.lower() == candidate]
            if matches:
                self._posture_col = matches[0]
                break
        
        # Find paragraphs/sections column  
        for candidate in ['sections', 'paragraphs', 'paras', 'text']:
            matches = [c for c in df_cols if c.lower() == candidate]
            if matches:
                self._paragraphs_col = matches[0]
                break
    
    def compute_statistics(self, use_cache: bool = True) -> DatasetStatistics:
        """Computes comprehensive dataset statistics."""
        if use_cache and self._stats is not None:
            return self._stats
        
        df = self.loader.df
        self._detect_columns(df)
        
        num_documents = len(df)
        
        # Posture distribution - handle list column (multi-label)
        if self._posture_col:
            posture_col = df[self._posture_col]
            # Check if it's a list column
            first_val = posture_col.iloc[0]
            if isinstance(first_val, list):
                # Explode to count each posture
                exploded = posture_col.explode()
                posture_dist = exploded.value_counts()
                num_postures = exploded.nunique()
            else:
                posture_dist = posture_col.value_counts()
                num_postures = posture_col.nunique()
        else:
            posture_dist = pd.Series(dtype=int)
            num_postures = 0
        
        # Paragraph counts - handle nested sections structure
        if self._paragraphs_col:
            para_col = df[self._paragraphs_col]
            
            def count_paragraphs(sections):
                """Count paragraphs from sections structure."""
                if not isinstance(sections, list):
                    return 1
                total = 0
                for section in sections:
                    if isinstance(section, dict) and 'paragraphs' in section:
                        total += len(section['paragraphs'])
                    elif isinstance(section, str):
                        total += 1
                return total if total > 0 else len(sections)
            
            def count_words_in_sections(sections):
                """Count words from sections structure."""
                if not isinstance(sections, list):
                    return len(str(sections).split())
                total = 0
                for section in sections:
                    if isinstance(section, dict) and 'paragraphs' in section:
                        for para in section['paragraphs']:
                            total += len(str(para).split())
                    elif isinstance(section, str):
                        total += len(section.split())
                return total
            
            para_counts = para_col.apply(count_paragraphs)
            total_paragraphs = para_counts.sum()
            avg_paragraphs = para_counts.mean()
            word_counts = para_col.apply(count_words_in_sections)
        else:
            total_paragraphs = num_documents
            avg_paragraphs = 1.0
            word_counts = pd.Series([0] * num_documents)
        
        self._stats = DatasetStatistics(
            num_documents=num_documents,
            num_postures=num_postures,
            total_paragraphs=int(total_paragraphs),
            posture_distribution=posture_dist,
            avg_paragraphs_per_doc=float(avg_paragraphs),
            avg_words_per_doc=float(word_counts.mean()),
            min_words=int(word_counts.min()),
            max_words=int(word_counts.max()),
            median_words=float(word_counts.median()),
            schema=self.loader.get_schema()
        )
        
        return self._stats
    
    def get_posture_distribution(self) -> pd.Series:
        """Returns posture label counts as a Series."""
        stats = self.compute_statistics()
        return stats.posture_distribution
    
    def get_class_imbalance_ratio(self) -> float:
        """Computes imbalance ratio (max/min class count)."""
        stats = self.compute_statistics()
        counts = stats.posture_distribution.values
        if len(counts) == 0 or min(counts) == 0:
            return float('inf')
        return max(counts) / min(counts)
    
    def get_word_counts(self) -> pd.Series:
        """Returns word counts per document."""
        df = self.loader.df
        self._detect_columns(df)
        
        if self._paragraphs_col:
            para_col = df[self._paragraphs_col]
            
            def count_words_in_sections(sections):
                """Count words from sections structure."""
                if not isinstance(sections, list):
                    return len(str(sections).split())
                total = 0
                for section in sections:
                    if isinstance(section, dict) and 'paragraphs' in section:
                        for para in section['paragraphs']:
                            total += len(str(para).split())
                    elif isinstance(section, str):
                        total += len(section.split())
                return total
            
            return para_col.apply(count_words_in_sections)
        
        return pd.Series([0] * len(df))
    
    def get_posture_taxonomy(self) -> 'PostureTaxonomy':
        """Creates a PostureTaxonomy for analyzing posture label distribution."""
        stats = self.compute_statistics()
        return PostureTaxonomy(all_postures=stats.posture_distribution)


@dataclass
class PostureTaxonomy:
    """
    Organizes posture labels by frequency tiers for modeling decisions.
    
    Helps decide which postures to keep, merge, or filter based on
    sample counts for training viability.
    """
    all_postures: pd.Series  # posture -> count mapping
    
    # Frequency thresholds
    COMMON_THRESHOLD: int = 100
    MODERATE_THRESHOLD: int = 10
    
    @property
    def common(self) -> list[str]:
        """Postures with >= 100 occurrences (viable for ML)."""
        return list(self.all_postures[self.all_postures >= self.COMMON_THRESHOLD].index)
    
    @property
    def moderate(self) -> list[str]:
        """Postures with 10-99 occurrences (might need oversampling)."""
        mask = (self.all_postures >= self.MODERATE_THRESHOLD) & (self.all_postures < self.COMMON_THRESHOLD)
        return list(self.all_postures[mask].index)
    
    @property
    def rare(self) -> list[str]:
        """Postures with < 10 occurrences (candidates for 'Other' bucket)."""
        return list(self.all_postures[self.all_postures < self.MODERATE_THRESHOLD].index)
    
    @property
    def singletons(self) -> list[str]:
        """Postures appearing exactly once."""
        return list(self.all_postures[self.all_postures == 1].index)
    
    def summary(self) -> dict:
        """Returns summary statistics."""
        return {
            'total_unique': len(self.all_postures),
            'common_count': len(self.common),
            'moderate_count': len(self.moderate),
            'rare_count': len(self.rare),
            'singleton_count': len(self.singletons),
            'total_assignments': int(self.all_postures.sum()),
            'common_coverage': float(self.all_postures[self.common].sum() / self.all_postures.sum()),
        }
    
    def get_modeling_subset(self, min_samples: int = 50) -> list[str]:
        """Returns postures with enough samples for reliable ML training."""
        return list(self.all_postures[self.all_postures >= min_samples].index)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Returns full taxonomy as DataFrame for analysis."""
        df = self.all_postures.reset_index()
        df.columns = ['posture', 'count']
        df['tier'] = df['count'].apply(
            lambda x: 'common' if x >= self.COMMON_THRESHOLD 
            else 'moderate' if x >= self.MODERATE_THRESHOLD 
            else 'rare'
        )
        return df.sort_values('count', ascending=False)
