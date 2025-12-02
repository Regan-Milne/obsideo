"""
Data science utilities for working with DataFrames and datasets.
"""

import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import Client
    import pandas as pd

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def save_dataframe(
    df: "pd.DataFrame", 
    *, 
    name: str,
    client: "Client",
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "parquet"
) -> "ArtifactVersion":
    """
    Save a pandas DataFrame to local storage.
    
    Args:
        df: DataFrame to save
        name: Logical name for the artifact (e.g., "acme/datasets/sales")
        client: Client instance
        metadata: Optional metadata dictionary
        format: File format - "parquet" (default), "csv", or "json"
        
    Returns:
        ArtifactVersion with assigned version number
        
    Raises:
        ImportError: If pandas is not installed
        ValueError: If unsupported format specified
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for save_dataframe. Install with: pip install 'obsideo[data]'")
    
    # Create temporary file with appropriate extension
    if format == "parquet":
        suffix = ".parquet"
    elif format == "csv":
        suffix = ".csv"
    elif format == "json":
        suffix = ".json"
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'json'")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Save DataFrame in requested format
        if format == "parquet":
            df.to_parquet(temp_path, index=False)
        elif format == "csv":
            df.to_csv(temp_path, index=False)
        elif format == "json":
            df.to_json(temp_path, orient="records", indent=2)
        
        # Add format info to metadata
        full_metadata = {"format": format, "shape": list(df.shape)}
        if metadata:
            full_metadata.update(metadata)
        
        # Store with client
        return client.put(temp_path, name=name, metadata=full_metadata)
    
    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


def load_dataframe(
    *,
    name: str,
    client: "Client",
    version: Optional[int] = None,
    **kwargs
) -> "pd.DataFrame":
    """
    Load a DataFrame from local storage.
    
    Args:
        name: Logical name of the artifact
        client: Client instance
        version: Version number (None for latest)
        **kwargs: Additional arguments passed to pandas loading function
        
    Returns:
        pandas DataFrame containing the loaded data
        
    Raises:
        ImportError: If pandas is not installed
        FileNotFoundError: If artifact doesn't exist
        
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> df = obs.data.load_dataframe(name="acme/datasets/sales", client=client)
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for load_dataframe. Install with: pip install 'obsideo[data]'")
    
    # Get the artifact info to determine format
    artifact_info = client.get_artifact_info(name, version)
    if artifact_info is None:
        raise FileNotFoundError(f"Artifact not found: {name}" + 
                               (f" version {version}" if version else ""))
    
    # Get the file
    file_path = client.get(name, version)
    
    try:
        # Determine format from metadata or file extension
        format_type = "parquet"  # default
        if artifact_info.metadata:
            format_type = artifact_info.metadata.get("format", "parquet")
        
        # If no format in metadata, guess from name
        if format_type == "parquet" and "." in name:
            ext = name.split(".")[-1].lower()
            if ext in ["csv", "json", "parquet"]:
                format_type = ext
        
        # Load with appropriate function
        if format_type == "parquet":
            return pd.read_parquet(file_path, **kwargs)
        elif format_type == "csv":
            return pd.read_csv(file_path, **kwargs)
        elif format_type == "json":
            return pd.read_json(file_path, **kwargs)
        else:
            # Try to guess format from file extension
            if str(file_path).endswith('.csv'):
                return pd.read_csv(file_path, **kwargs)
            elif str(file_path).endswith('.json'):
                return pd.read_json(file_path, **kwargs)
            else:
                return pd.read_parquet(file_path, **kwargs)
    
    finally:
        # Clean up temp file
        file_path.unlink(missing_ok=True)


def save_dataset(
    datasets: Dict[str, "pd.DataFrame"],
    *,
    name: str,
    client: "Client", 
    metadata: Optional[Dict[str, Any]] = None,
    format: str = "parquet"
) -> Dict[str, "ArtifactVersion"]:
    """
    Save a multi-part dataset (e.g., train/val/test splits).
    
    Args:
        datasets: Dictionary of split name -> DataFrame
        name: Base name for the dataset (e.g., "acme/datasets/mnist")
        client: Client instance
        metadata: Optional metadata dictionary
        format: File format for each split
        
    Returns:
        Dictionary of split name -> ArtifactVersion
        
    Example:
        >>> datasets = {
        ...     "train": train_df,
        ...     "val": val_df,
        ...     "test": test_df
        ... }
        >>> versions = obs.data.save_dataset(
        ...     datasets, name="acme/datasets/mnist", client=client
        ... )
    """
    versions = {}
    
    for split_name, df in datasets.items():
        split_artifact_name = f"{name}/{split_name}"
        split_metadata = {"split": split_name, "dataset_name": name}
        if metadata:
            split_metadata.update(metadata)
        
        version = save_dataframe(
            df, 
            name=split_artifact_name, 
            client=client,
            metadata=split_metadata,
            format=format
        )
        versions[split_name] = version
    
    return versions


def load_dataset(
    *,
    name: str,
    client: "Client",
    version: Optional[int] = None,
    **kwargs
) -> Dict[str, "pd.DataFrame"]:
    """
    Load a multi-part dataset.
    
    Args:
        name: Base name of the dataset
        client: Client instance
        version: Version number for all splits (None for latest)
        **kwargs: Additional arguments passed to pandas loading functions
        
    Returns:
        Dictionary of split name -> DataFrame
        
    Example:
        >>> dataset = obs.data.load_dataset(name="acme/datasets/mnist", client=client)
        >>> train_df = dataset["train"]
        >>> val_df = dataset["val"]
    """
    # Find all artifacts that start with the dataset name
    all_artifacts = client.list_artifacts()
    dataset_artifacts = [
        art for art in all_artifacts 
        if art.startswith(f"{name}/")
    ]
    
    if not dataset_artifacts:
        raise FileNotFoundError(f"No dataset splits found for: {name}")
    
    # Load each split
    dataset = {}
    for artifact_name in dataset_artifacts:
        # Extract split name (everything after the last /)
        split_name = artifact_name.split("/")[-1]
        
        df = load_dataframe(
            name=artifact_name,
            client=client, 
            version=version,
            **kwargs
        )
        dataset[split_name] = df
    
    return dataset