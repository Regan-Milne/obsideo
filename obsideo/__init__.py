"""
Obsideo - Local content-addressed storage for machine learning artifacts.

A Python-first SDK that provides local content-addressed storage using BLAKE3 
hashing and SQLite metadata for ML datasets, checkpoints, and artifacts.

Example usage:
    >>> import obsideo as obs
    >>> client = obs.Client.from_env()
    >>> client.put("data.csv", name="acme/datasets/sales")
    >>> path = client.get("acme/datasets/sales")
    >>> df = obs.data.load_dataframe(name="acme/datasets/sales", client=client)
"""

__version__ = "0.1.0"
__author__ = "OSD Labs"
__email__ = "info@osdlabs.io"

from .client import Client
from .store import ArtifactVersion, ChecksumMismatchError
from . import data
from . import ml

__all__ = ["Client", "ArtifactVersion", "ChecksumMismatchError", "data", "ml"]