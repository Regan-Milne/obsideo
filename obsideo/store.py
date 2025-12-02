"""
SQLite metadata store for artifacts.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class ArtifactVersion:
    """Represents a versioned artifact in the store."""
    name: str
    version: int
    hash: str
    size_bytes: int
    created_at: str
    metadata: Dict[str, Any]


class ChecksumMismatchError(Exception):
    """Raised when a blob's checksum doesn't match the expected value."""
    pass


class Store:
    """SQLite-backed metadata store for content-addressed artifacts."""
    
    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize the store with a SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with required tables."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    UNIQUE(name, version)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_name ON artifacts(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(hash)")
            
            conn.commit()
    
    def get_next_version(self, name: str) -> int:
        """
        Get the next version number for an artifact.
        
        Args:
            name: Artifact name
            
        Returns:
            Next version number (1 for new artifacts)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifacts WHERE name = ?",
                (name,)
            )
            return cursor.fetchone()[0]
    
    def put_artifact(
        self,
        name: str,
        hash: str,
        size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactVersion:
        """
        Store metadata for an artifact.
        
        Args:
            name: Logical name of the artifact
            hash: BLAKE3 hash of the content
            size_bytes: Size of the content in bytes
            metadata: Optional metadata dictionary
            
        Returns:
            ArtifactVersion with the assigned version number
        """
        version = self.get_next_version(name)
        created_at = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO artifacts (name, version, hash, size_bytes, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, version, hash, size_bytes, created_at, metadata_json)
            )
            conn.commit()
        
        return ArtifactVersion(
            name=name,
            version=version,
            hash=hash,
            size_bytes=size_bytes,
            created_at=created_at,
            metadata=metadata or {}
        )
    
    def get_artifact(self, name: str, version: Optional[int] = None) -> Optional[ArtifactVersion]:
        """
        Get artifact metadata by name and version.
        
        Args:
            name: Artifact name
            version: Version number (None for latest)
            
        Returns:
            ArtifactVersion if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            if version is None:
                # Get latest version
                cursor = conn.execute(
                    """
                    SELECT name, version, hash, size_bytes, created_at, metadata_json
                    FROM artifacts 
                    WHERE name = ? 
                    ORDER BY version DESC 
                    LIMIT 1
                    """,
                    (name,)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT name, version, hash, size_bytes, created_at, metadata_json
                    FROM artifacts 
                    WHERE name = ? AND version = ?
                    """,
                    (name, version)
                )
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            name, version, hash, size_bytes, created_at, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            return ArtifactVersion(
                name=name,
                version=version,
                hash=hash,
                size_bytes=size_bytes,
                created_at=created_at,
                metadata=metadata
            )
    
    def list_versions(self, name: str) -> List[ArtifactVersion]:
        """
        List all versions of an artifact.
        
        Args:
            name: Artifact name
            
        Returns:
            List of ArtifactVersion objects, ordered by version DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT name, version, hash, size_bytes, created_at, metadata_json
                FROM artifacts 
                WHERE name = ? 
                ORDER BY version DESC
                """,
                (name,)
            )
            
            versions = []
            for row in cursor.fetchall():
                name, version, hash, size_bytes, created_at, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                versions.append(ArtifactVersion(
                    name=name,
                    version=version,
                    hash=hash,
                    size_bytes=size_bytes,
                    created_at=created_at,
                    metadata=metadata
                ))
            
            return versions
    
    def list_artifacts(self) -> List[str]:
        """
        List all artifact names in the store.
        
        Returns:
            List of unique artifact names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT name FROM artifacts ORDER BY name")
            return [row[0] for row in cursor.fetchall()]