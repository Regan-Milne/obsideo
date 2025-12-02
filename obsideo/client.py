"""
Core client for local content-addressed storage.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from .store import Store, ArtifactVersion, ChecksumMismatchError
from .hashing import blake3_file, blake3_bytes, store_blob, verify_blob, blob_path


class Client:
    """
    Main client for local content-addressed artifact storage.
    
    This client provides methods for storing and retrieving artifacts using
    BLAKE3 content-addressed storage with SQLite metadata.
    
    Args:
        root: Root directory for storage (default: ~/.obsideo or OBSIDEO_HOME)
    
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> client.put("data.csv", name="acme/datasets/sales")
        >>> path = client.get("acme/datasets/sales")
    """
    
    def __init__(self, root: Optional[Union[str, Path]] = None):
        if root is None:
            # Use OBSIDEO_HOME env var or default to ~/.obsideo
            home_dir = os.environ.get("OBSIDEO_HOME")
            if home_dir:
                self.root = Path(home_dir)
            else:
                self.root = Path.home() / ".obsideo"
        else:
            self.root = Path(root)
        
        # Ensure root directory exists
        self.root.mkdir(parents=True, exist_ok=True)
        
        # Initialize store with db.sqlite3
        db_path = self.root / "db.sqlite3"
        self.store = Store(db_path)
    
    @classmethod
    def from_env(cls) -> "Client":
        """
        Create a Client using environment configuration.
        
        Uses OBSIDEO_HOME environment variable or defaults to ~/.obsideo.
        """
        return cls()
    
    def put(
        self,
        src: Union[str, Path, bytes],
        *,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactVersion:
        """
        Store an artifact in content-addressed storage.
        
        Args:
            src: Source file path or bytes to store
            name: Logical name for the artifact (e.g., "acme/models/resnet")
            metadata: Optional metadata dictionary
            
        Returns:
            ArtifactVersion with assigned version number and hash
        """
        if isinstance(src, bytes):
            # Handle bytes input
            digest = blake3_bytes(src)
            size_bytes = len(src)
            
            # Store the blob
            store_blob(self.root, src, digest)
        else:
            # Handle file path input
            src_path = Path(src)
            if not src_path.exists():
                raise FileNotFoundError(f"Source file not found: {src_path}")
            
            digest = blake3_file(src_path)
            size_bytes = src_path.stat().st_size
            
            # Store the blob
            store_blob(self.root, src_path, digest)
        
        # Store metadata
        return self.store.put_artifact(name, digest, size_bytes, metadata)
    
    def get(
        self,
        name: str,
        version: Optional[int] = None,
        dst: Optional[Union[str, Path]] = None,
        verify: bool = True,
    ) -> Path:
        """
        Retrieve an artifact from storage.
        
        Args:
            name: Logical name of the artifact
            version: Version number (None for latest)
            dst: Destination path (None for temporary file)
            verify: Whether to verify checksum
            
        Returns:
            Path to the retrieved file
            
        Raises:
            FileNotFoundError: If artifact doesn't exist
            ChecksumMismatchError: If verification fails
        """
        # Get artifact metadata
        artifact = self.store.get_artifact(name, version)
        if artifact is None:
            raise FileNotFoundError(f"Artifact not found: {name}" + 
                                   (f" version {version}" if version else ""))
        
        # Get blob path
        blob_file = blob_path(self.root, artifact.hash)
        if not blob_file.exists():
            raise FileNotFoundError(f"Blob file missing: {blob_file}")
        
        # Verify checksum if requested
        if verify:
            if not verify_blob(self.root, artifact.hash):
                raise ChecksumMismatchError(
                    f"Checksum mismatch for {name} v{artifact.version}"
                )
        
        # Copy to destination
        if dst is None:
            # Create temporary file with original extension if possible
            suffix = ""
            if "." in name:
                suffix = "." + name.split(".")[-1]
            
            # Create temp file that won't be automatically deleted
            fd, dst = tempfile.mkstemp(suffix=suffix, prefix="obsideo_")
            os.close(fd)  # Close the file descriptor
            dst = Path(dst)
        else:
            dst = Path(dst)
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the blob to destination
        shutil.copy2(blob_file, dst)
        
        return dst
    
    def list_versions(self, name: str) -> List[ArtifactVersion]:
        """
        List all versions of an artifact.
        
        Args:
            name: Logical name of the artifact
            
        Returns:
            List of ArtifactVersion objects, ordered by version DESC
        """
        return self.store.list_versions(name)
    
    def verify(self, name: str, version: Optional[int] = None) -> bool:
        """
        Verify that an artifact's stored blob matches its hash.
        
        Args:
            name: Logical name of the artifact
            version: Version number (None for latest)
            
        Returns:
            True if verification passes
        """
        try:
            artifact = self.store.get_artifact(name, version)
            if artifact is None:
                return False
            
            return verify_blob(self.root, artifact.hash)
        except Exception:
            return False
    
    def list_artifacts(self) -> List[str]:
        """
        List all artifact names in the store.
        
        Returns:
            List of unique artifact names
        """
        return self.store.list_artifacts()
    
    def get_artifact_info(self, name: str, version: Optional[int] = None) -> Optional[ArtifactVersion]:
        """
        Get information about an artifact without retrieving the content.
        
        Args:
            name: Logical name of the artifact
            version: Version number (None for latest)
            
        Returns:
            ArtifactVersion object or None if not found
        """
        return self.store.get_artifact(name, version)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage stats
        """
        artifacts = self.store.list_artifacts()
        total_artifacts = len(artifacts)
        
        total_versions = 0
        total_size = 0
        
        for artifact_name in artifacts:
            versions = self.store.list_versions(artifact_name)
            total_versions += len(versions)
            # Count unique blobs to avoid double-counting
            unique_hashes = set(v.hash for v in versions)
            for version in versions:
                if version.hash in unique_hashes:
                    total_size += version.size_bytes
                    unique_hashes.discard(version.hash)
        
        return {
            "root_path": str(self.root),
            "total_artifacts": total_artifacts,
            "total_versions": total_versions,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }