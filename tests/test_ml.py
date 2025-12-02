"""
Tests for ML functionality.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from obsideo.client import Client
from obsideo import ml


class TestMLOperations:
    """Test ML save/load operations."""
    
    @pytest.fixture
    def temp_root(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def client(self, temp_root):
        """Create a client with temporary storage."""
        return Client(root=temp_root)
    
    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint for testing."""
        return {
            'model_state_dict': {'layer1.weight': [1, 2, 3]},
            'optimizer_state_dict': {'param_groups': []},
            'epoch': 10,
            'loss': 0.123
        }
    
    def test_save_load_checkpoint_pickle(self, client, sample_checkpoint):
        """Test saving and loading checkpoint with pickle."""
        # Save checkpoint
        version = ml.save_checkpoint(
            sample_checkpoint,
            name="test/model_checkpoint",
            client=client,
            framework="pickle"
        )
        
        assert version.name == "test/model_checkpoint"
        assert version.metadata["framework"] == "pickle"
        assert "model_state_dict" in version.metadata["checkpoint_keys"]
        
        # Load checkpoint
        loaded_checkpoint = ml.load_checkpoint(
            name="test/model_checkpoint",
            client=client,
            framework="pickle"
        )
        
        assert loaded_checkpoint == sample_checkpoint
    
    @patch('obsideo.ml.HAS_TORCH', True)
    @patch('obsideo.ml.torch')
    def test_save_load_checkpoint_torch(self, mock_torch, client, sample_checkpoint):
        """Test saving and loading checkpoint with PyTorch."""
        # Mock torch.save and torch.load
        mock_torch.save = MagicMock()
        mock_torch.load = MagicMock(return_value=sample_checkpoint)
        
        # Save checkpoint
        version = ml.save_checkpoint(
            sample_checkpoint,
            name="test/model_checkpoint",
            client=client,
            framework="torch"
        )
        
        assert version.metadata["framework"] == "torch"
        mock_torch.save.assert_called_once()
        
        # Load checkpoint
        loaded_checkpoint = ml.load_checkpoint(
            name="test/model_checkpoint",
            client=client,
            framework="torch"
        )
        
        mock_torch.load.assert_called_once()
        assert loaded_checkpoint == sample_checkpoint
    
    def test_save_checkpoint_invalid_data(self, client):
        """Test saving invalid checkpoint data raises error."""
        with pytest.raises(ValueError, match="Checkpoint data must be a dictionary"):
            ml.save_checkpoint(
                "not a dict",
                name="test/checkpoint",
                client=client
            )
    
    def test_save_checkpoint_with_metadata(self, client, sample_checkpoint):
        """Test saving checkpoint with custom metadata."""
        metadata = {"experiment": "baseline", "dataset": "cifar10"}
        
        version = ml.save_checkpoint(
            sample_checkpoint,
            name="test/checkpoint",
            client=client,
            metadata=metadata,
            framework="pickle"
        )
        
        assert version.metadata["framework"] == "pickle"
        assert version.metadata["experiment"] == "baseline"
        assert version.metadata["dataset"] == "cifar10"
    
    def test_load_checkpoint_auto_detect_framework(self, client, sample_checkpoint):
        """Test auto-detecting framework from metadata."""
        # Save with explicit framework
        ml.save_checkpoint(
            sample_checkpoint,
            name="test/checkpoint",
            client=client,
            framework="pickle"
        )
        
        # Load without specifying framework (should auto-detect)
        loaded = ml.load_checkpoint(name="test/checkpoint", client=client)
        assert loaded == sample_checkpoint
    
    def test_save_load_model_pickle(self, client):
        """Test saving and loading model with pickle."""
        # Simple model object
        model = {"type": "simple_model", "weights": [1, 2, 3]}
        
        version = ml.save_model(
            model,
            name="test/simple_model",
            client=client,
            framework="pickle"
        )
        
        assert version.metadata["framework"] == "pickle"
        assert version.metadata["model_type"] == "dict"
        
        loaded_model = ml.load_model(name="test/simple_model", client=client)
        assert loaded_model == model
    
    @patch('obsideo.ml.HAS_TORCH', True)
    @patch('obsideo.ml.torch')
    def test_save_load_model_torch(self, mock_torch, client):
        """Test saving and loading model with PyTorch."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "MyModel"
        mock_torch.save = MagicMock()
        mock_torch.load = MagicMock(return_value=mock_model)
        
        version = ml.save_model(
            mock_model,
            name="test/pytorch_model",
            client=client,
            framework="torch"
        )
        
        assert version.metadata["framework"] == "torch"
        assert version.metadata["model_type"] == "MyModel"
        mock_torch.save.assert_called_once()
        
        loaded_model = ml.load_model(name="test/pytorch_model", client=client)
        mock_torch.load.assert_called_once()
        assert loaded_model == mock_model
    
    def test_load_nonexistent_checkpoint(self, client):
        """Test loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            ml.load_checkpoint(name="nonexistent/checkpoint", client=client)
    
    def test_load_nonexistent_model(self, client):
        """Test loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            ml.load_model(name="nonexistent/model", client=client)
    
    def test_unsupported_framework(self, client, sample_checkpoint):
        """Test unsupported framework raises error."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            ml.save_checkpoint(
                sample_checkpoint,
                name="test/checkpoint",
                client=client,
                framework="unsupported"
            )
    
    def test_torch_without_import(self, client, sample_checkpoint):
        """Test torch operations without PyTorch installed."""
        with patch('obsideo.ml.HAS_TORCH', False):
            with pytest.raises(ImportError, match="PyTorch is required"):
                ml.save_checkpoint(
                    sample_checkpoint,
                    name="test/checkpoint",
                    client=client,
                    framework="torch"
                )
    
    def test_load_checkpoint_specific_version(self, client, sample_checkpoint):
        """Test loading a specific version of a checkpoint."""
        # Save multiple versions
        v1 = ml.save_checkpoint(
            sample_checkpoint,
            name="test/checkpoint",
            client=client,
            framework="pickle"
        )
        
        modified_checkpoint = sample_checkpoint.copy()
        modified_checkpoint['epoch'] = 20
        v2 = ml.save_checkpoint(
            modified_checkpoint,
            name="test/checkpoint",
            client=client,
            framework="pickle"
        )
        
        # Load specific versions
        loaded_v1 = ml.load_checkpoint(name="test/checkpoint", client=client, version=1)
        loaded_v2 = ml.load_checkpoint(name="test/checkpoint", client=client, version=2)
        
        assert loaded_v1['epoch'] == 10
        assert loaded_v2['epoch'] == 20


if __name__ == "__main__":
    pytest.main([__file__])