"""
Legal Transformer Training Module for TR Data Challenge.

This module provides transformer-based models for multi-label classification
of legal procedural postures, with support for long documents via summarization.

Classes:
    - LegalSummarizer: Summarizes long documents using Legal-LED
    - LegalLongformerTrainer: Fine-tunes Legal-Longformer for classification
    - HybridLegalClassifier: Orchestrates summarization + classification pipeline

Design Patterns:
    - Strategy Pattern: Long document handling (truncate vs summarize)
    - Template Method: Training loop with customizable hooks
    - Factory Method: Device and model initialization
"""

import json
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from tqdm.auto import tqdm

# Type hints for lazy imports (only used by type checker, not at runtime)
if TYPE_CHECKING:
    from transformers import (
        AutoTokenizer as _AutoTokenizerType,
        AutoModel as _AutoModelType,
        AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLMType,
        LongformerForSequenceClassification as _LongformerType,
    )

# Lazy imports for transformers (heavy)
_transformers_imported: bool = False
_AutoTokenizer: Any = None
_AutoModel: Any = None
_AutoModelForSeq2SeqLM: Any = None
_LongformerForSequenceClassification: Any = None
_get_linear_schedule_with_warmup: Any = None


def _import_transformers():
    """Lazy import of transformers to speed up module load."""
    global _transformers_imported, _AutoTokenizer, _AutoModel
    global _AutoModelForSeq2SeqLM, _LongformerForSequenceClassification
    global _get_linear_schedule_with_warmup
    
    if not _transformers_imported:
        from transformers import (
            AutoTokenizer,
            AutoModel,
            AutoModelForSeq2SeqLM,
            LongformerForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        _AutoTokenizer = AutoTokenizer
        _AutoModel = AutoModel
        _AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
        _LongformerForSequenceClassification = LongformerForSequenceClassification
        _get_linear_schedule_with_warmup = get_linear_schedule_with_warmup
        _transformers_imported = True


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Device Management
# =============================================================================

class DeviceManager:
    """
    Manages compute device selection with auto-detection.
    
    Priority: CUDA > MPS > CPU
    """
    
    @staticmethod
    def get_device(preference: str = 'auto') -> torch.device:
        """
        Get the best available device.
        
        Args:
            preference: 'auto', 'cuda', 'mps', or 'cpu'
            
        Returns:
            torch.device instance
        """
        if preference == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using Apple MPS (Metal Performance Shaders)")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU")
        else:
            device = torch.device(preference)
            logger.info(f"Using requested device: {device}")
        
        return device
    
    @staticmethod
    def get_device_info() -> dict:
        """Returns detailed device information."""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        return info


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5  # 1e-5 is good middle ground for fine-tuning
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    checkpoint_dir: Optional[str] = None  # Directory to save checkpoints after each epoch
    use_pos_weight: bool = True  # Use class weights to handle imbalance
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class TrainingHistory:
    """Records training progress for analysis."""
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_f1_micro: list[float] = field(default_factory=list)
    val_f1_macro: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_f1: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_micro': self.val_f1_micro,
            'val_f1_macro': self.val_f1_macro,
            'best_epoch': self.best_epoch,
            'best_f1': self.best_f1,
        }
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        with open(path, 'r') as f:
            data = json.load(f)
        history = cls()
        for key, value in data.items():
            setattr(history, key, value)
        return history


# =============================================================================
# Summarization Component
# =============================================================================

class SummaryCache:
    """
    Disk-based cache for document summaries.
    
    Uses content hashing to avoid recomputing summaries for identical documents.
    """
    
    def __init__(self, cache_dir: str = 'outputs/summaries'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, str] = {}
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate short hash for text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.json"
    
    def get(self, text: str) -> Optional[str]:
        """Retrieve cached summary if exists."""
        text_hash = self._hash_text(text)
        
        # Check memory cache first
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                summary = data['summary']
                self._memory_cache[text_hash] = summary
                return summary
        
        return None
    
    def put(self, text: str, summary: str) -> None:
        """Store summary in cache."""
        text_hash = self._hash_text(text)
        
        # Memory cache
        self._memory_cache[text_hash] = summary
        
        # Disk cache
        cache_path = self._get_cache_path(text_hash)
        with open(cache_path, 'w') as f:
            json.dump({
                'hash': text_hash,
                'original_length': len(text),
                'summary_length': len(summary),
                'summary': summary,
            }, f, indent=2)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob('*.json'))
        return {
            'cached_summaries': len(cache_files),
            'memory_cache_size': len(self._memory_cache),
            'cache_dir': str(self.cache_dir),
        }


class LegalSummarizer:
    """
    Summarizes legal documents using Legal-LED model.
    
    The LED (Longformer Encoder-Decoder) model handles up to 16,384 tokens,
    making it suitable for long legal documents.
    
    Args:
        model_name: HuggingFace model identifier
        max_input_length: Maximum input tokens (default: 16384)
        summary_max_length: Maximum summary tokens (default: 1024)
        summary_min_length: Minimum summary tokens (default: 256)
        cache_dir: Directory for caching summaries
        device: Compute device ('auto', 'cuda', 'mps', 'cpu')
    """
    
    DEFAULT_MODEL = 'nsi319/legal-led-base-16384'
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_input_length: int = 16384,
        summary_max_length: int = 1024,
        summary_min_length: int = 256,
        cache_dir: Optional[str] = 'outputs/summaries',
        device: str = 'auto',
    ):
        _import_transformers()
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.summary_max_length = summary_max_length
        self.summary_min_length = summary_min_length
        self.device = DeviceManager.get_device(device)
        
        # Initialize cache
        self.cache = SummaryCache(cache_dir) if cache_dir else None
        
        # Lazy model loading
        self._tokenizer = None
        self._model = None
    
    def _load_model(self) -> None:
        """Load model and tokenizer on first use."""
        if self._model is None:
            logger.info(f"Loading summarization model: {self.model_name}")
            self._tokenizer = _AutoTokenizer.from_pretrained(self.model_name)
            self._model = _AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Summarization model loaded successfully")
    
    @property
    def tokenizer(self):
        """Get tokenizer, loading model if needed."""
        self._load_model()
        assert self._tokenizer is not None, "Tokenizer failed to load"
        return self._tokenizer
    
    @property
    def model(self):
        """Get model, loading if needed."""
        self._load_model()
        assert self._model is not None, "Model failed to load"
        return self._model
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text without loading full model."""
        if self._tokenizer is None:
            self._tokenizer = _AutoTokenizer.from_pretrained(self.model_name)
        assert self._tokenizer is not None, "Tokenizer failed to load"
        return len(self._tokenizer.encode(text, add_special_tokens=False))
    
    def summarize(self, text: str, use_cache: bool = True) -> str:
        """
        Summarize a single document.
        
        Args:
            text: Document text to summarize
            use_cache: Whether to use disk cache
            
        Returns:
            Summary string
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        self._load_model()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                min_length=self.summary_min_length,
                max_length=self.summary_max_length,
            )
        
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        # Cache result
        if use_cache and self.cache:
            self.cache.put(text, summary)
        
        return summary
    
    def summarize_batch(
        self,
        texts: list[str],
        use_cache: bool = True,
        show_progress: bool = True,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Summarize multiple documents with TRUE batched GPU inference for efficiency.
        
        Separates cache hits from cache misses, performs batched tokenization and
        generation on GPU for all cache misses, then combines results maintaining
        input order.
        
        Args:
            texts: List of document texts
            use_cache: Whether to use disk cache
            show_progress: Whether to show progress bar
            batch_size: Number of documents to process in parallel (default 8)
            
        Returns:
            List of summary strings in same order as input texts
        """
        self._load_model()
        all_summaries: list[Optional[str]] = [None] * len(texts)  # Preserve order
        texts_needing_inference: list[tuple[int, str]] = []  # (original_index, text) tuples
        
        # Phase 1: Check cache and separate cache misses
        for idx, text in enumerate(texts):
            if use_cache and self.cache:
                cached = self.cache.get(text)
                if cached is not None:
                    all_summaries[idx] = cached
                    continue
            
            texts_needing_inference.append((idx, text))
        
        # Phase 2: Process cache misses in batches with true GPU batching
        # NOTE: MPS has INT_MAX tensor dimension limit - LED's 16K context with large batches
        # exceeds this (batch_size=8 fails), but smaller batches work fine
        if str(self.device).startswith('mps'):
            effective_batch_size = 3  # MPS can handle small batches (8 fails, 2-3 works)
            logger.info(f"Using batch_size=3 for MPS (INT_MAX tensor limit with large batches)")
        else:
            effective_batch_size = batch_size
        
        if texts_needing_inference:
            num_batches = (len(texts_needing_inference) + effective_batch_size - 1) // effective_batch_size
            pbar = tqdm(total=num_batches, desc="Summarizing", disable=not show_progress)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * effective_batch_size
                end_idx = min(start_idx + effective_batch_size, len(texts_needing_inference))
                batch_items = texts_needing_inference[start_idx:end_idx]
                
                # Extract indices and texts from batch items
                original_indices = [item[0] for item in batch_items]
                batch_texts = [item[1] for item in batch_items]
                
                # Batch tokenization: tokenize all texts together
                batch_inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    max_length=self.max_input_length,
                    truncation=True,
                    padding='max_length',
                ).to(self.device)
                
                # Batch generation: single model.generate() call for entire batch
                with torch.no_grad():
                    summary_ids = self.model.generate(
                        batch_inputs['input_ids'],
                        attention_mask=batch_inputs['attention_mask'],
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        min_length=self.summary_min_length,
                        max_length=self.summary_max_length,
                    )
                
                # Batch decoding: decode all summaries together
                summaries = self.tokenizer.batch_decode(
                    summary_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                
                # Cache results and store in correct positions
                for orig_idx, text, summary in zip(original_indices, batch_texts, summaries):
                    all_summaries[orig_idx] = summary
                    if use_cache and self.cache:
                        self.cache.put(text, summary)
                
                pbar.update(1)
            
            pbar.close()
        
        # All positions should be filled at this point
        return all_summaries  # type: ignore[return-value]


# =============================================================================
# Classification Dataset
# =============================================================================

class LegalTextDataset(Dataset):
    """
    PyTorch Dataset for legal text classification.
    
    Handles tokenization and multi-label encoding.
    """
    
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 4096,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32),
        }


# =============================================================================
# Long Document Strategy (Strategy Pattern)
# =============================================================================

class LongDocumentStrategy(ABC):
    """Abstract base class for handling documents exceeding max_length."""
    
    @abstractmethod
    def process(self, text: str, token_count: int, max_length: int) -> str:
        """Process a long document."""
        pass


class TruncateStrategy(LongDocumentStrategy):
    """Simply truncate to max_length (handled by tokenizer)."""
    
    def process(self, text: str, token_count: int, max_length: int) -> str:
        return text  # Tokenizer handles truncation
        # Note: pass not needed, return statement is present


class HeadTailStrategy(LongDocumentStrategy):
    """Keep first and last portions of the document."""
    
    def __init__(self, head_ratio: float = 0.7):
        self.head_ratio = head_ratio
    
    def process(self, text: str, token_count: int, max_length: int) -> str:
        # Approximate character positions
        total_chars = len(text)
        target_chars = int(total_chars * max_length / token_count)
        
        head_chars = int(target_chars * self.head_ratio)
        tail_chars = target_chars - head_chars
        
        head = text[:head_chars]
        tail = text[-tail_chars:] if tail_chars > 0 else ""
        
        return head + " [...] " + tail


class SummarizeStrategy(LongDocumentStrategy):
    """Summarize long documents before classification."""
    
    def __init__(self, summarizer: LegalSummarizer):
        self.summarizer = summarizer
    
    def process(self, text: str, token_count: int, max_length: int) -> str:
        return self.summarizer.summarize(text)


# =============================================================================
# Trainer Base Class (Template Method Pattern)
# =============================================================================

class BaseTransformerTrainer(ABC):
    """
    Abstract base class for transformer-based trainers.
    
    Implements Template Method pattern for training loop,
    allowing subclasses to customize specific steps.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        max_length: int,
        device: str = 'auto',
    ):
        _import_transformers()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = DeviceManager.get_device(device)
        
        self._tokenizer = None
        self._model = None
        self._is_trained = False
        self._loss_fn = None  # Will be set in train() if use_pos_weight=True
        self.history = TrainingHistory()
    
    @abstractmethod
    def _create_model(self):
        """Create the model architecture. To be implemented by subclasses."""
        pass
    
    def _load_tokenizer(self):
        """Load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = _AutoTokenizer.from_pretrained(self.model_name)
        assert self._tokenizer is not None, "Tokenizer failed to load"
        return self._tokenizer
    
    @property
    def tokenizer(self):
        """Get tokenizer, loading if needed."""
        return self._load_tokenizer()
    
    @property
    def model(self) -> nn.Module:
        """Get model, creating if needed."""
        if self._model is None:
            self._model = self._create_model()
            assert self._model is not None, "Model failed to initialize"
            self._model.to(self.device)
        return self._model
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=True))
    
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        return [self.count_tokens(t) for t in texts]
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute evaluation metrics."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        return {
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        }
    
    def _training_step(self, batch: dict, model: nn.Module) -> torch.Tensor:
        """Single training step. Returns loss."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass without labels to get logits
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Compute loss with pos_weight if available
        if self._loss_fn is not None:
            loss = self._loss_fn(outputs.logits, labels)
        else:
            # Fallback: re-run with labels to use model's built-in loss
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
        
        return loss
    
    def _validation_step(
        self,
        val_loader: TorchDataLoader,
        model: nn.Module,
    ) -> tuple[float, dict]:
        """Validate model and return loss and metrics."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                # Compute loss with pos_weight if available
                if self._loss_fn is not None:
                    loss = self._loss_fn(outputs.logits, labels)
                    total_loss += loss.item()
                else:
                    # Fallback: re-run with labels
                    outputs_with_labels = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += outputs_with_labels.loss.item()
                
                # Get predictions
                logits = outputs.logits
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_labels)
        metrics = self._compute_metrics(y_true, y_pred)
        
        model.train()
        return avg_loss, metrics
    
    def train(
        self,
        train_texts: list[str],
        y_train: np.ndarray,
        val_texts: list[str],
        y_val: np.ndarray,
        config: Optional[TrainingConfig] = None,
        callback: Optional[Callable[[int, dict], None]] = None,
    ) -> 'BaseTransformerTrainer':
        """
        Train the model.
        
        Args:
            train_texts: Training documents
            y_train: Training labels (binary matrix)
            val_texts: Validation documents
            y_val: Validation labels
            config: Training configuration
            callback: Optional callback(epoch, metrics) for logging
            
        Returns:
            self for chaining
        """
        if config is None:
            config = TrainingConfig()
        
        # Setup loss function with pos_weight for class imbalance
        if config.use_pos_weight:
            # Calculate pos_weight: ratio of negatives to positives per class
            # Higher weight for rare positive classes
            pos_count = y_train.sum(axis=0)  # positives per class
            neg_count = y_train.shape[0] - pos_count  # negatives per class
            # Avoid division by zero
            pos_count = np.maximum(pos_count, 1)
            pos_weight = neg_count / pos_count
            # Cap weights to avoid extreme values
            pos_weight = np.clip(pos_weight, 1.0, 50.0)
            pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(self.device)
            self._loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            logger.info(f"Using pos_weight: min={pos_weight.min():.2f}, max={pos_weight.max():.2f}, mean={pos_weight.mean():.2f}")
        else:
            self._loss_fn = None
        
        # Create datasets
        train_dataset = LegalTextDataset(
            train_texts, y_train, self.tokenizer, self.max_length
        )
        val_dataset = LegalTextDataset(
            val_texts, y_val, self.tokenizer, self.max_length
        )
        
        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # MPS compatibility
        )
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        scheduler = _get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training loop
        best_f1 = 0.0
        patience_counter = 0
        best_state = None
        
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"Effective batch size: {config.effective_batch_size}")
        
        self.model.train()
        
        for epoch in range(config.num_epochs):
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            )
            
            for step, batch in enumerate(progress):
                loss_raw = self._training_step(batch, self.model)
                loss_scaled = loss_raw / config.gradient_accumulation_steps
                loss_scaled.backward()
                
                epoch_loss += loss_raw.item()  # Track actual loss, not scaled
                
                # Gradient accumulation
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                progress.set_postfix({'loss': f'{loss_raw.item():.4f}'})  # Show actual loss
            
            # Validation
            val_loss, val_metrics = self._validation_step(val_loader, self.model)
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Record history
            self.history.train_losses.append(avg_train_loss)
            self.history.val_losses.append(val_loss)
            self.history.val_f1_micro.append(val_metrics['f1_micro'])
            self.history.val_f1_macro.append(val_metrics['f1_macro'])
            
            logger.info(
                f"Epoch {epoch + 1}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_f1_micro={val_metrics['f1_micro']:.4f}, "
                f"val_f1_macro={val_metrics['f1_macro']:.4f}"
            )
            
            # Callback
            if callback:
                callback(epoch + 1, {
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    **val_metrics,
                })
            
            # Save checkpoint after each epoch
            if config.checkpoint_dir:
                ckpt_dir = Path(config.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_f1_micro': val_metrics['f1_micro'],
                }, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")
            
            # Early stopping
            if val_metrics['f1_micro'] > best_f1:
                best_f1 = val_metrics['f1_micro']
                self.history.best_f1 = best_f1
                self.history.best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            logger.info(f"Restored best model from epoch {self.history.best_epoch}")
        
        self._is_trained = True
        return self
    
    def predict_proba(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """
        Predict probabilities for texts.
        
        Args:
            texts: Documents to classify
            batch_size: Inference batch size
            
        Returns:
            Probability matrix (n_samples, n_labels)
        """
        # Get model reference (property ensures it's initialized)
        model = self.model
        model.eval()
        all_probs = []
        
        dataset = LegalTextDataset(
            texts,
            np.zeros((len(texts), self.num_labels)),  # Dummy labels
            self.tokenizer,
            self.max_length,
        )
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Use forward() explicitly - clearer than __call__ and type-checker friendly
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_probs.append(probs)
        
        return np.vstack(all_probs)
    
    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for texts.
        
        Args:
            texts: Documents to classify
            threshold: Classification threshold
            
        Returns:
            Binary prediction matrix (n_samples, n_labels)
        """
        proba = self.predict_proba(texts)
        return (proba >= threshold).astype(int)
    
    def save(self, path: str) -> None:
        """Save model weights and training history."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
        }, save_path)
        
        # Save history alongside
        history_path = save_path.with_suffix('.history.json')
        self.history.save(str(history_path))
        
        logger.info(f"Model saved to {save_path}")
    
    def load(self, path: str) -> 'BaseTransformerTrainer':
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self._is_trained = True
        
        # Load history if exists
        history_path = Path(path).with_suffix('.history.json')
        if history_path.exists():
            self.history = TrainingHistory.load(str(history_path))
        
        logger.info(f"Model loaded from {path}")
        return self


# =============================================================================
# Legal Longformer Trainer
# =============================================================================

class LegalLongformerTrainer(BaseTransformerTrainer):
    """
    Fine-tunes Legal-Longformer for multi-label classification.
    
    Legal-Longformer supports 4,096 tokens, covering most legal documents
    without truncation.
    
    Args:
        num_labels: Number of classification labels
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length (default: 4096)
        device: Compute device
    """
    
    DEFAULT_MODEL = 'lexlms/legal-longformer-base'
    
    def __init__(
        self,
        num_labels: int,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 4096,
        device: str = 'auto',
    ):
        super().__init__(model_name, num_labels, max_length, device)
    
    def _create_model(self):
        """Create Longformer classification model."""
        logger.info(f"Loading classification model: {self.model_name}")
        
        model = _LongformerForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="multi_label_classification",
        )
        
        logger.info(f"Model loaded with {self.num_labels} labels")
        return model


# =============================================================================
# Hybrid Classifier (Facade Pattern)
# =============================================================================

class HybridLegalClassifier:
    """
    Hybrid classifier combining summarization and classification.
    
    Routes documents based on length:
    - Short documents (≤max_length): Direct classification
    - Long documents (>max_length): Summarize first, then classify
    
    Args:
        classifier_model: HuggingFace model for classification
        summarizer_model: HuggingFace model for summarization
        num_labels: Number of classification labels
        max_length: Classifier's maximum sequence length
        summary_max_length: Target summary length
        cache_dir: Directory for caching summaries
        device: Compute device
    """
    
    def __init__(
        self,
        num_labels: int,
        classifier_model: str = LegalLongformerTrainer.DEFAULT_MODEL,
        summarizer_model: str = LegalSummarizer.DEFAULT_MODEL,
        max_length: int = 4096,
        summary_max_length: int = 1024,
        cache_dir: str = 'outputs/summaries',
        device: str = 'auto',
    ):
        self.num_labels = num_labels
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.device_name = device
        
        # Initialize components
        self.summarizer = LegalSummarizer(
            model_name=summarizer_model,
            summary_max_length=summary_max_length,
            cache_dir=cache_dir,
            device=device,
        )
        
        self.classifier = LegalLongformerTrainer(
            num_labels=num_labels,
            model_name=classifier_model,
            max_length=max_length,
            device=device,
        )
        
        # Statistics
        self._stats = {
            'total_processed': 0,
            'direct_classified': 0,
            'summarized_first': 0,
        }
    
    @property
    def tokenizer(self):
        return self.classifier.tokenizer
    
    @property
    def history(self) -> TrainingHistory:
        return self.classifier.history
    
    def analyze_lengths(self, texts: list[str]) -> pd.DataFrame:
        """
        Analyze document lengths and routing decisions.
        
        Returns DataFrame with token counts and routing info.
        """
        token_counts = self.classifier.count_tokens_batch(texts)
        
        df = pd.DataFrame({
            'doc_idx': range(len(texts)),
            'token_count': token_counts,
            'exceeds_limit': [tc > self.max_length for tc in token_counts],
            'route': ['summarize' if tc > self.max_length else 'direct' 
                     for tc in token_counts],
        })
        
        return df
    
    def get_length_stats(self, texts: list[str]) -> dict:
        """Get summary statistics about document lengths."""
        analysis = self.analyze_lengths(texts)
        
        return {
            'total_docs': len(texts),
            'direct_route': (analysis['route'] == 'direct').sum(),
            'summarize_route': (analysis['route'] == 'summarize').sum(),
            'direct_pct': (analysis['route'] == 'direct').mean() * 100,
            'summarize_pct': (analysis['route'] == 'summarize').mean() * 100,
            'token_mean': analysis['token_count'].mean(),
            'token_median': analysis['token_count'].median(),
            'token_max': analysis['token_count'].max(),
            'max_length_threshold': self.max_length,
        }
    
    def preprocess_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[str]:
        """
        Preprocess texts: summarize long documents, keep short ones.
        
        Args:
            texts: Input documents
            show_progress: Whether to show progress bar
            
        Returns:
            Processed texts (some summarized, some original)
        """
        analysis = self.analyze_lengths(texts)
        processed = []
            
        long_indices = analysis[analysis['exceeds_limit']]['doc_idx'].tolist()
        
        # Summarize long documents with batching for efficiency
        if long_indices:
            long_texts = [texts[i] for i in long_indices]
            
            logger.info(f"Summarizing {len(long_texts)} long documents (batch_size=8)...")
            summaries = self.summarizer.summarize_batch(
                long_texts,
                show_progress=show_progress,
                batch_size=8,  # Process 8 docs at a time for better GPU utilization
            )
            
            # Create lookup
            summary_map = dict(zip(long_indices, summaries))
        else:
            summary_map = {}
        
        # Build processed list
        for i, text in enumerate(texts):
            if i in summary_map:
                processed.append(summary_map[i])
                self._stats['summarized_first'] += 1
            else:
                processed.append(text)
                self._stats['direct_classified'] += 1
            self._stats['total_processed'] += 1
        
        return processed
    
    def train(
        self,
        train_texts: list[str],
        y_train: np.ndarray,
        val_texts: list[str],
        y_val: np.ndarray,
        config: Optional[TrainingConfig] = None,
        preprocess: bool = True,
        callback: Optional[Callable[[int, dict], None]] = None,
    ) -> 'HybridLegalClassifier':
        """
        Train the hybrid classifier.
        
        Args:
            train_texts: Training documents
            y_train: Training labels
            val_texts: Validation documents
            y_val: Validation labels
            config: Training configuration
            preprocess: Whether to summarize long docs first
            callback: Optional epoch callback
            
        Returns:
            self for chaining
        """
        # Preprocess if requested
        if preprocess:
            logger.info("Preprocessing training texts...")
            train_texts = self.preprocess_texts(train_texts)
            logger.info("Preprocessing validation texts...")
            val_texts = self.preprocess_texts(val_texts)
        
        # Train classifier
        self.classifier.train(
            train_texts=train_texts,
            y_train=y_train,
            val_texts=val_texts,
            y_val=y_val,
            config=config,
            callback=callback,
        )
        
        return self
    
    def predict_proba(
        self,
        texts: list[str],
        preprocess: bool = True,
        batch_size: int = 8,
    ) -> np.ndarray:
        """Predict probabilities."""
        if preprocess:
            texts = self.preprocess_texts(texts, show_progress=True)
        return self.classifier.predict_proba(texts, batch_size=batch_size)
    
    def predict(
        self,
        texts: list[str],
        preprocess: bool = True,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict_proba(texts, preprocess=preprocess)
        return (proba >= threshold).astype(int)
    
    def get_processing_stats(self) -> dict:
        """Get processing statistics."""
        return {
            **self._stats,
            'cache_stats': self.summarizer.cache.get_stats() if self.summarizer.cache else {},
        }
    
    def save(self, path: str) -> None:
        """Save classifier model."""
        self.classifier.save(path)
    
    def load(self, path: str) -> 'HybridLegalClassifier':
        """Load classifier model."""
        self.classifier.load(path)
        return self
