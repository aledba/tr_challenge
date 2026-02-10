"""Tests for BERT trainer functionality."""

import pytest
import numpy as np
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_trainer import (
    DeviceManager,
    TrainingConfig,
    TrainingHistory,
    LegalLongformerTrainer,
    HybridLegalClassifier,
    LegalTextDataset,
)


class TestDeviceManager:
    """Tests for DeviceManager."""
    
    def test_get_device_auto(self):
        """Test auto device detection."""
        device = DeviceManager.get_device('auto')
        assert isinstance(device, torch.device)
    
    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = DeviceManager.get_device('cpu')
        assert device.type == 'cpu'
    
    def test_get_device_info(self):
        """Test device info returns dict."""
        info = DeviceManager.get_device_info()
        assert isinstance(info, dict)
        assert 'cuda_available' in info
        assert 'mps_available' in info


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        assert config.batch_size == 4
        assert config.num_epochs == 5
        assert config.learning_rate == 1e-5
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 16
    
    def test_checkpoint_dir_optional(self):
        """Test checkpoint_dir is Optional[str]."""
        config = TrainingConfig()
        assert config.checkpoint_dir is None
        
        config2 = TrainingConfig(checkpoint_dir='/tmp/checkpoints')
        assert config2.checkpoint_dir == '/tmp/checkpoints'


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""
    
    def test_default_history(self):
        """Test default history values."""
        history = TrainingHistory()
        assert history.train_losses == []
        assert history.val_losses == []
        assert history.best_epoch == 0
        assert history.best_f1 == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        history = TrainingHistory()
        history.train_losses = [0.5, 0.4, 0.3]
        history.val_losses = [0.6, 0.5, 0.4]
        history.best_epoch = 3
        history.best_f1 = 0.75
        
        d = history.to_dict()
        assert isinstance(d, dict)
        assert d['best_epoch'] == 3
        assert d['best_f1'] == 0.75
    
    def test_save_load_roundtrip(self, tmp_path):
        """Test save and load functionality."""
        history = TrainingHistory()
        history.train_losses = [0.5, 0.4]
        history.val_f1_micro = [0.6, 0.7]
        history.best_epoch = 2
        history.best_f1 = 0.7
        
        path = tmp_path / 'history.json'
        history.save(str(path))
        
        loaded = TrainingHistory.load(str(path))
        assert loaded.train_losses == [0.5, 0.4]
        assert loaded.best_epoch == 2


class TestLegalTextDataset:
    """Tests for LegalTextDataset."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained('bert-base-uncased')
    
    def test_dataset_length(self, mock_tokenizer):
        """Test dataset length matches input."""
        texts = ["This is a test.", "Another test document."]
        labels = np.array([[1, 0], [0, 1]])
        
        dataset = LegalTextDataset(texts, labels, mock_tokenizer, max_length=128)
        assert len(dataset) == 2
    
    def test_dataset_getitem(self, mock_tokenizer):
        """Test dataset returns correct structure."""
        texts = ["This is a test document."]
        labels = np.array([[1, 0, 1]])
        
        dataset = LegalTextDataset(texts, labels, mock_tokenizer, max_length=128)
        item = dataset[0]
        
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert item['labels'].shape == (3,)


class TestLegalLongformerTrainer:
    """Tests for LegalLongformerTrainer."""
    
    def test_initialization(self):
        """Test trainer initializes without loading model."""
        trainer = LegalLongformerTrainer(
            num_labels=10,
            device='cpu',
        )
        assert trainer.num_labels == 10
        assert trainer.max_length == 4096
        assert trainer._model is None  # Lazy loading
    
    def test_tokenizer_property(self):
        """Test tokenizer loads on demand."""
        trainer = LegalLongformerTrainer(
            num_labels=5,
            device='cpu',
        )
        tokenizer = trainer.tokenizer
        assert tokenizer is not None
        # Second access should return same instance
        assert trainer.tokenizer is tokenizer
    
    def test_count_tokens(self):
        """Test token counting."""
        trainer = LegalLongformerTrainer(
            num_labels=5,
            device='cpu',
        )
        count = trainer.count_tokens("This is a test sentence.")
        assert isinstance(count, int)
        assert count > 0


class TestHybridLegalClassifier:
    """Tests for HybridLegalClassifier."""
    
    def test_initialization(self, tmp_path):
        """Test hybrid classifier initialization."""
        classifier = HybridLegalClassifier(
            num_labels=10,
            cache_dir=str(tmp_path / 'cache'),
            device='cpu',
        )
        assert classifier.num_labels == 10
        assert classifier.summarizer is not None
        assert classifier.classifier is not None
    
    def test_processing_stats(self, tmp_path):
        """Test processing stats are tracked."""
        classifier = HybridLegalClassifier(
            num_labels=5,
            cache_dir=str(tmp_path / 'cache'),
            device='cpu',
        )
        stats = classifier.get_processing_stats()
        assert 'total_processed' in stats
        assert 'direct_classified' in stats
        assert 'summarized_first' in stats


@pytest.mark.skipif(
    not Path('outputs/legal_longformer_best.pt').exists(),
    reason="Trained model not available"
)
class TestModelLoading:
    """Tests for loading trained models."""
    
    def test_load_trained_model(self):
        """Test loading the trained model checkpoint."""
        from src.data_loader import DataLoader
        from src.model_trainer import DataPreparer
        
        # Load data to get label count
        loader = DataLoader('data/TRDataChallenge2023.txt')
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        data = preparer.prepare(max_features=10000, ngram_range=(1, 2))
        
        # Initialize and load model
        classifier = HybridLegalClassifier(
            num_labels=len(data.label_names),
            cache_dir='outputs/summaries',
            device='cpu',  # Use CPU for testing
        )
        classifier.load('outputs/legal_longformer_best.pt')
        
        # Verify model is loaded
        assert classifier.classifier._is_trained
    
    def test_model_predict_single(self):
        """Test model can make predictions."""
        from src.data_loader import DataLoader
        from src.model_trainer import DataPreparer
        
        loader = DataLoader('data/TRDataChallenge2023.txt')
        preparer = DataPreparer(loader, min_label_count=50, random_state=42)
        data = preparer.prepare(max_features=10000, ngram_range=(1, 2))
        
        classifier = HybridLegalClassifier(
            num_labels=len(data.label_names),
            cache_dir='outputs/summaries',
            device='cpu',
        )
        classifier.load('outputs/legal_longformer_best.pt')
        
        # Test prediction on a single short text
        test_text = "The court grants summary judgment for defendant."
        proba = classifier.predict_proba([test_text], preprocess=False, batch_size=1)
        
        assert proba.shape == (1, len(data.label_names))
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0
