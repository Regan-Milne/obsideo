"""
Machine learning utilities for saving and loading checkpoints and models.
"""

import pickle
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .client import Client

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def save_checkpoint(
    checkpoint: dict, 
    *, 
    name: str,
    client: "Client",
    metadata: Optional[Dict[str, Any]] = None,
    framework: str = "torch"
) -> "ArtifactVersion":
    """
    Save a training checkpoint to local storage.
    
    Args:
        checkpoint: Checkpoint data to save (model state, optimizer state, etc.)
        name: Logical name for the checkpoint (e.g., "acme/models/resnet_epoch_10")
        client: Client instance
        metadata: Optional metadata dictionary
        framework: Framework to use - "torch" (default) or "pickle"
        
    Returns:
        ArtifactVersion with assigned version number
        
    Raises:
        ValueError: If invalid checkpoint data or unsupported framework
        ImportError: If PyTorch is not installed and framework is "torch"
        
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> checkpoint = {
        ...     'model_state_dict': model.state_dict(),
        ...     'optimizer_state_dict': optimizer.state_dict(),
        ...     'epoch': 10,
        ...     'loss': 0.123
        ... }
        >>> obs.ml.save_checkpoint(
        ...     checkpoint, name="acme/models/resnet_epoch_10", client=client
        ... )
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint data must be a dictionary")
    
    if framework == "torch" and not HAS_TORCH:
        raise ImportError("PyTorch is required for torch framework. Install with: pip install 'obsideo[ml]'")
    
    # Create temporary file with appropriate extension
    if framework == "torch":
        suffix = ".pt"
    elif framework == "pickle":
        suffix = ".pkl"
    else:
        raise ValueError(f"Unsupported framework: {framework}. Use 'torch' or 'pickle'")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Save checkpoint with appropriate method
        if framework == "torch":
            torch.save(checkpoint, temp_path)
        elif framework == "pickle":
            with open(temp_path, "wb") as f:
                pickle.dump(checkpoint, f)
        
        # Add framework info to metadata
        full_metadata = {"framework": framework, "checkpoint_keys": list(checkpoint.keys())}
        if metadata:
            full_metadata.update(metadata)
        
        # Store with client
        return client.put(temp_path, name=name, metadata=full_metadata)
    
    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


def load_checkpoint(
    *,
    name: str,
    client: "Client", 
    version: Optional[int] = None,
    framework: Optional[str] = None,
    **kwargs
) -> dict:
    """
    Load a checkpoint from local storage.
    
    Args:
        name: Logical name of the checkpoint
        client: Client instance
        version: Version number (None for latest)
        framework: Framework to use - if None, auto-detect from metadata
        **kwargs: Additional arguments passed to loading function
        
    Returns:
        Dictionary containing the loaded checkpoint data
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ImportError: If required framework is not installed
        
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> checkpoint = obs.ml.load_checkpoint(
        ...     name="acme/models/resnet_epoch_10", client=client
        ... )
        >>> model.load_state_dict(checkpoint['model_state_dict'])
    """
    # Get the artifact info to determine framework
    artifact_info = client.get_artifact_info(name, version)
    if artifact_info is None:
        raise FileNotFoundError(f"Checkpoint not found: {name}" + 
                               (f" version {version}" if version else ""))
    
    # Determine framework from metadata or parameter
    if framework is None and artifact_info.metadata:
        framework = artifact_info.metadata.get("framework", "torch")
    elif framework is None:
        framework = "torch"  # default
    
    if framework == "torch" and not HAS_TORCH:
        raise ImportError("PyTorch is required for torch framework. Install with: pip install 'obsideo[ml]'")
    
    # Get the file
    file_path = client.get(name, version)
    
    try:
        # Load with appropriate method
        if framework == "torch":
            return torch.load(file_path, **kwargs)
        elif framework == "pickle":
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    finally:
        # Clean up temp file
        file_path.unlink(missing_ok=True)


def save_model(
    model: Any,
    *,
    name: str, 
    client: "Client",
    metadata: Optional[Dict[str, Any]] = None,
    framework: str = "torch"
) -> "ArtifactVersion":
    """
    Save a complete model to local storage.
    
    Args:
        model: Model object to save
        name: Logical name for the model (e.g., "acme/models/resnet_final")
        client: Client instance
        metadata: Optional metadata dictionary
        framework: Framework to use - "torch" (default), "pickle", or "sklearn"
        
    Returns:
        ArtifactVersion with assigned version number
        
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> obs.ml.save_model(
        ...     model, name="acme/models/resnet_final", client=client,
        ...     metadata={"accuracy": 0.95, "dataset": "imagenet"}
        ... )
    """
    if framework == "torch" and not HAS_TORCH:
        raise ImportError("PyTorch is required for torch framework. Install with: pip install 'obsideo[ml]'")
    
    # Create temporary file with appropriate extension
    if framework == "torch":
        suffix = ".pt"
    elif framework in ["pickle", "sklearn"]:
        suffix = ".pkl"
    else:
        raise ValueError(f"Unsupported framework: {framework}. Use 'torch', 'pickle', or 'sklearn'")
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Save model with appropriate method
        if framework == "torch":
            torch.save(model, temp_path)
        elif framework in ["pickle", "sklearn"]:
            with open(temp_path, "wb") as f:
                pickle.dump(model, f)
        
        # Add framework info to metadata
        full_metadata = {"framework": framework, "model_type": type(model).__name__}
        if metadata:
            full_metadata.update(metadata)
        
        # Store with client
        return client.put(temp_path, name=name, metadata=full_metadata)
    
    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


def load_model(
    *,
    name: str,
    client: "Client",
    version: Optional[int] = None,
    framework: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Load a complete model from local storage.
    
    Args:
        name: Logical name of the model
        client: Client instance
        version: Version number (None for latest)
        framework: Framework to use - if None, auto-detect from metadata
        **kwargs: Additional arguments passed to loading function
        
    Returns:
        Loaded model object
        
    Example:
        >>> import obsideo as obs
        >>> client = obs.Client.from_env()
        >>> model = obs.ml.load_model(name="acme/models/resnet_final", client=client)
    """
    # Get the artifact info to determine framework
    artifact_info = client.get_artifact_info(name, version)
    if artifact_info is None:
        raise FileNotFoundError(f"Model not found: {name}" + 
                               (f" version {version}" if version else ""))
    
    # Determine framework from metadata or parameter
    if framework is None and artifact_info.metadata:
        framework = artifact_info.metadata.get("framework", "torch")
    elif framework is None:
        framework = "torch"  # default
    
    if framework == "torch" and not HAS_TORCH:
        raise ImportError("PyTorch is required for torch framework. Install with: pip install 'obsideo[ml]'")
    
    # Get the file
    file_path = client.get(name, version)
    
    try:
        # Load with appropriate method
        if framework == "torch":
            return torch.load(file_path, **kwargs)
        elif framework in ["pickle", "sklearn"]:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    finally:
        # Clean up temp file
        file_path.unlink(missing_ok=True)