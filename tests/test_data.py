"""
Tests for data science functionality.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from obsideo.client import Client
from obsideo import data


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
class TestDataFrameOperations:
    """Test DataFrame save/load operations."""
    
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
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
    
    def test_save_load_dataframe_parquet(self, client, sample_df):
        """Test saving and loading DataFrame as parquet."""
        # Save DataFrame
        version = data.save_dataframe(
            sample_df, 
            name="test/sales_data", 
            client=client,
            format="parquet"
        )
        
        assert version.name == "test/sales_data"
        assert version.metadata["format"] == "parquet"
        assert version.metadata["shape"] == [3, 3]
        
        # Load DataFrame
        loaded_df = data.load_dataframe(name="test/sales_data", client=client)
        
        # Should be identical
        pd.testing.assert_frame_equal(sample_df, loaded_df)
    
    def test_save_load_dataframe_csv(self, client, sample_df):
        """Test saving and loading DataFrame as CSV."""
        # Save DataFrame
        version = data.save_dataframe(
            sample_df, 
            name="test/sales_data", 
            client=client,
            format="csv"
        )
        
        assert version.metadata["format"] == "csv"
        
        # Load DataFrame
        loaded_df = data.load_dataframe(name="test/sales_data", client=client)
        
        # Should be identical
        pd.testing.assert_frame_equal(sample_df, loaded_df)
    
    def test_save_load_dataframe_json(self, client, sample_df):
        """Test saving and loading DataFrame as JSON."""
        # Save DataFrame
        version = data.save_dataframe(
            sample_df, 
            name="test/sales_data", 
            client=client,
            format="json"
        )
        
        assert version.metadata["format"] == "json"
        
        # Load DataFrame
        loaded_df = data.load_dataframe(name="test/sales_data", client=client)
        
        # Should be identical (JSON may change column types)
        pd.testing.assert_frame_equal(sample_df, loaded_df, check_dtype=False)
    
    def test_save_dataframe_with_metadata(self, client, sample_df):
        """Test saving DataFrame with custom metadata."""
        metadata = {"source": "sales_system", "date": "2025-12-01"}
        
        version = data.save_dataframe(
            sample_df, 
            name="test/sales_data", 
            client=client,
            metadata=metadata
        )
        
        # Should include both format metadata and custom metadata
        assert version.metadata["format"] == "parquet"  # default
        assert version.metadata["shape"] == [3, 3]
        assert version.metadata["source"] == "sales_system"
        assert version.metadata["date"] == "2025-12-01"
    
    def test_load_dataframe_specific_version(self, client, sample_df):
        """Test loading a specific version of a DataFrame."""
        # Save multiple versions
        v1 = data.save_dataframe(sample_df, name="test/data", client=client)
        
        modified_df = sample_df.copy()
        modified_df.loc[0, 'A'] = 999
        v2 = data.save_dataframe(modified_df, name="test/data", client=client)
        
        # Load specific version
        loaded_v1 = data.load_dataframe(name="test/data", client=client, version=1)
        loaded_v2 = data.load_dataframe(name="test/data", client=client, version=2)
        
        pd.testing.assert_frame_equal(sample_df, loaded_v1)
        pd.testing.assert_frame_equal(modified_df, loaded_v2)
    
    def test_save_load_dataset(self, client, sample_df):
        """Test saving and loading multi-part datasets."""
        # Create splits
        train_df = sample_df.iloc[:2]
        test_df = sample_df.iloc[2:]
        
        datasets = {
            "train": train_df,
            "test": test_df
        }
        
        # Save dataset
        versions = data.save_dataset(
            datasets,
            name="test/mnist",
            client=client,
            metadata={"experiment": "baseline"}
        )
        
        assert "train" in versions
        assert "test" in versions
        assert versions["train"].name == "test/mnist/train"
        assert versions["test"].name == "test/mnist/test"
        
        # Load dataset
        loaded_dataset = data.load_dataset(name="test/mnist", client=client)
        
        assert "train" in loaded_dataset
        assert "test" in loaded_dataset
        pd.testing.assert_frame_equal(train_df, loaded_dataset["train"])
        pd.testing.assert_frame_equal(test_df, loaded_dataset["test"])
    
    def test_load_nonexistent_dataframe(self, client):
        """Test loading nonexistent DataFrame raises error."""
        with pytest.raises(FileNotFoundError):
            data.load_dataframe(name="nonexistent/data", client=client)
    
    def test_unsupported_format(self, client, sample_df):
        """Test unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            data.save_dataframe(
                sample_df, 
                name="test/data", 
                client=client,
                format="unsupported"
            )


class TestDataWithoutPandas:
    """Test data operations when pandas is not available."""
    
    @pytest.fixture
    def temp_root(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def client(self, temp_root):
        """Create a client with temporary storage."""
        return Client(root=temp_root)
    
    def test_save_dataframe_without_pandas(self, client):
        """Test that save_dataframe raises ImportError without pandas."""
        with patch.dict('sys.modules', {'pandas': None}):
            with patch('obsideo.data.HAS_PANDAS', False):
                with pytest.raises(ImportError, match="pandas is required"):
                    data.save_dataframe(None, name="test", client=client)
    
    def test_load_dataframe_without_pandas(self, client):
        """Test that load_dataframe raises ImportError without pandas."""
        with patch.dict('sys.modules', {'pandas': None}):
            with patch('obsideo.data.HAS_PANDAS', False):
                with pytest.raises(ImportError, match="pandas is required"):
                    data.load_dataframe(name="test", client=client)


if __name__ == "__main__":
    pytest.main([__file__])