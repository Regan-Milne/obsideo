"""
BLAKE3 content-addressed storage utilities.
"""

import os
import shutil
from pathlib import Path
from typing import Union

import blake3


def blake3_file(path: Union[str, Path]) -> str:
    """
    Compute BLAKE3 hash of a file.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        Hex digest of the BLAKE3 hash
    """
    path = Path(path)
    hasher = blake3.blake3()
    
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def blake3_bytes(data: bytes) -> str:
    """
    Compute BLAKE3 hash of bytes.
    
    Args:
        data: Bytes to hash
        
    Returns:
        Hex digest of the BLAKE3 hash
    """
    hasher = blake3.blake3()
    hasher.update(data)
    return hasher.hexdigest()


def blob_path(root: Path, digest: str) -> Path:
    """
    Get the filesystem path for a blob with the given digest.
    
    Args:
        root: Root directory (e.g., ~/.obsideo)
        digest: BLAKE3 hex digest
        
    Returns:
        Path to the blob file (e.g., ~/.obsideo/artifacts/3f/3f8c9d...)
    """
    return root / "artifacts" / digest[:2] / digest


def store_blob(root: Path, src: Union[str, Path, bytes], digest: str) -> Path:
    """
    Store a blob in content-addressed storage.
    
    Args:
        root: Root directory (e.g., ~/.obsideo)
        src: Source file path or bytes
        digest: BLAKE3 hex digest
        
    Returns:
        Path where the blob was stored
    """
    blob_file = blob_path(root, digest)
    
    # Create parent directories if they don't exist
    blob_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Don't overwrite if blob already exists
    if blob_file.exists():
        return blob_file
    
    if isinstance(src, bytes):
        # Write bytes directly
        with open(blob_file, "wb") as f:
            f.write(src)
    else:
        # Copy file
        shutil.copy2(src, blob_file)
    
    return blob_file


def verify_blob(root: Path, digest: str) -> bool:
    """
    Verify that a stored blob matches its expected hash.
    
    Args:
        root: Root directory (e.g., ~/.obsideo)
        digest: Expected BLAKE3 hex digest
        
    Returns:
        True if the blob exists and matches the digest
    """
    blob_file = blob_path(root, digest)
    
    if not blob_file.exists():
        return False
    
    actual_digest = blake3_file(blob_file)
    return actual_digest == digest