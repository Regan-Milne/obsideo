"""
Core functionality tests for Obsideo local storage.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from obsideo.client import Client
from obsideo.store import ArtifactVersion, ChecksumMismatchError
from obsideo.hashing import blake3_file, blake3_bytes


class TestClient:
    """Test the core Client functionality."""
    
    @pytest.fixture
    def temp_root(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def client(self, temp_root):
        """Create a client with temporary storage."""
        return Client(root=temp_root)
    
    def test_client_init_default_root(self):
        """Test client initialization with default root directory."""
        with patch.dict('os.environ', {}, clear=True):
            client = Client()
            expected_root = Path.home() / ".obsideo"
            assert client.root == expected_root
    
    def test_client_init_env_var(self):
        """Test client initialization with OBSIDEO_HOME environment variable."""
        test_home = "/tmp/test_obsideo"
        with patch.dict('os.environ', {'OBSIDEO_HOME': test_home}):
            client = Client()
            assert client.root == Path(test_home)
    
    def test_client_init_explicit_root(self, temp_root):
        """Test client initialization with explicit root."""
        client = Client(root=temp_root)
        assert client.root == temp_root
    
    def test_put_creates_new_version(self, client):
        """Test that put creates a new version for each call."""
        test_data = b"hello world"
        
        # First put
        v1 = client.put(test_data, name="test/artifact")
        assert v1.name == "test/artifact"
        assert v1.version == 1
        assert v1.size_bytes == len(test_data)
        
        # Second put
        v2 = client.put(test_data, name="test/artifact")
        assert v2.name == "test/artifact"
        assert v2.version == 2
        assert v2.size_bytes == len(test_data)
        
        # Same hash since same content
        assert v1.hash == v2.hash
    
    def test_put_file_path(self, client, temp_root):
        """Test putting a file by path."""
        # Create a test file
        test_file = temp_root / "test.txt"
        test_content = "hello world"
        test_file.write_text(test_content)
        
        version = client.put(test_file, name="test/file")
        assert version.name == "test/file"
        assert version.version == 1
        assert version.size_bytes == len(test_content.encode())
    
    def test_put_nonexistent_file(self, client, temp_root):
        """Test putting a nonexistent file raises error."""
        nonexistent = temp_root / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            client.put(nonexistent, name="test/missing")
    
    def test_get_latest(self, client):
        """Test getting the latest version of an artifact."""
        test_data = b"hello world"
        
        # Put multiple versions
        v1 = client.put(test_data, name="test/artifact")
        v2 = client.put(b"updated data", name="test/artifact")
        
        # Get latest (should be v2)
        retrieved_path = client.get("test/artifact")
        content = retrieved_path.read_bytes()
        assert content == b"updated data"
        
        # Clean up
        retrieved_path.unlink(missing_ok=True)
    
    def test_get_specific_version(self, client):
        """Test getting a specific version of an artifact."""
        test_data = b"hello world"
        
        # Put multiple versions
        v1 = client.put(test_data, name="test/artifact")
        v2 = client.put(b"updated data", name="test/artifact")
        
        # Get v1 specifically
        retrieved_path = client.get("test/artifact", version=1)
        content = retrieved_path.read_bytes()
        assert content == test_data
        
        # Clean up
        retrieved_path.unlink(missing_ok=True)
    
    def test_get_nonexistent(self, client):
        """Test getting a nonexistent artifact raises error."""
        with pytest.raises(FileNotFoundError):
            client.get("nonexistent/artifact")
    
    def test_get_with_destination(self, client, temp_root):
        """Test getting an artifact to a specific destination."""
        test_data = b"hello world"
        client.put(test_data, name="test/artifact")
        
        dst = temp_root / "output.txt"
        retrieved_path = client.get("test/artifact", dst=dst)
        
        assert retrieved_path == dst
        assert dst.read_bytes() == test_data
    
    def test_list_versions_order(self, client):
        """Test that list_versions returns versions in descending order."""
        test_data = b"hello world"
        
        # Put multiple versions
        v1 = client.put(test_data, name="test/artifact")
        v2 = client.put(b"version 2", name="test/artifact")
        v3 = client.put(b"version 3", name="test/artifact")
        
        versions = client.list_versions("test/artifact")
        assert len(versions) == 3
        assert versions[0].version == 3  # Latest first
        assert versions[1].version == 2
        assert versions[2].version == 1
    
    def test_checksum_verification(self, client, temp_root):
        """Test checksum verification on retrieval."""
        test_data = b"hello world"
        version = client.put(test_data, name="test/artifact")
        
        # Verify passes normally
        assert client.verify("test/artifact") is True
        
        # Corrupt the blob file
        blob_file = client.root / "artifacts" / version.hash[:2] / version.hash
        blob_file.write_bytes(b"corrupted data")
        
        # Verification should fail
        assert client.verify("test/artifact") is False
        
        # Getting with verification should raise error
        with pytest.raises(ChecksumMismatchError):
            client.get("test/artifact", verify=True)
    
    def test_checksum_mismatch_raises(self, client, temp_root):
        """Test that corrupted blobs raise ChecksumMismatchError."""
        test_data = b"hello world"
        version = client.put(test_data, name="test/artifact")
        
        # Corrupt the stored blob
        blob_file = client.root / "artifacts" / version.hash[:2] / version.hash
        blob_file.write_bytes(b"corrupted")
        
        # Should raise error when getting with verification
        with pytest.raises(ChecksumMismatchError):
            client.get("test/artifact", verify=True)
    
    def test_metadata_storage(self, client):
        """Test that metadata is stored and retrieved correctly."""
        test_data = b"hello world"
        metadata = {"author": "test", "purpose": "testing"}
        
        version = client.put(test_data, name="test/artifact", metadata=metadata)
        assert version.metadata == metadata
        
        # Retrieve and check metadata
        info = client.get_artifact_info("test/artifact")
        assert info.metadata == metadata
    
    def test_stats(self, client):
        """Test storage statistics."""
        test_data = b"hello world"
        
        # Empty stats
        stats = client.stats()
        assert stats["total_artifacts"] == 0
        assert stats["total_versions"] == 0
        assert stats["total_size_bytes"] == 0
        
        # Add some data
        client.put(test_data, name="test/artifact1")
        client.put(b"different data", name="test/artifact2")
        client.put(test_data, name="test/artifact1")  # Same content, new version
        
        stats = client.stats()
        assert stats["total_artifacts"] == 2
        assert stats["total_versions"] == 3
        # Should only count unique blobs once
        expected_size = len(test_data) + len(b"different data")
        assert stats["total_size_bytes"] == expected_size


class TestHashing:
    """Test the hashing utilities."""
    
    def test_blake3_bytes(self):
        """Test BLAKE3 hashing of bytes."""
        test_data = b"hello world"
        digest = blake3_bytes(test_data)
        assert isinstance(digest, str)
        assert len(digest) == 64  # BLAKE3 produces 32-byte hash = 64 hex chars
        
        # Same input should produce same hash
        assert blake3_bytes(test_data) == digest
        
        # Different input should produce different hash
        assert blake3_bytes(b"different") != digest
    
    def test_blake3_file(self, tmp_path):
        """Test BLAKE3 hashing of files."""
        test_file = tmp_path / "test.txt"
        test_content = "hello world"
        test_file.write_text(test_content)
        
        digest = blake3_file(test_file)
        assert isinstance(digest, str)
        assert len(digest) == 64
        
        # Should match bytes hash of same content
        expected = blake3_bytes(test_content.encode())
        assert digest == expected


if __name__ == "__main__":
    pytest.main([__file__])