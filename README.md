# Obsideo SDK

**Python SDK for OSD (Operation Save Data) decentralized storage protocol**

Obsideo is a Python-first SDK that lets machine learning teams store and retrieve datasets, checkpoints, and artifacts on the OSD storage network using simple Python calls.

## 🚀 Quick Start

### Installation

```bash
pip install obsideo
```

### Optional Dependencies

For data science workflows:
```bash
pip install "obsideo[data]"  # Adds pandas, numpy
```

For ML model management:
```bash
pip install "obsideo[ml]"    # Adds torch, tensorflow
```

For everything:
```bash
pip install "obsideo[all]"   # All optional dependencies
```

### Basic Usage

```python
import obsideo as obs

# Create client using environment variables
client = obs.Client.from_env()

# Upload a file
client.upload_file("data.csv", namespace="acme/datasets")

# Download a file
client.download_file("acme/datasets/data.csv", "local_data.csv")

# Load data directly into a DataFrame
df = obs.data.load_dataframe("acme/datasets/sales.csv")

# Save and load ML checkpoints
checkpoint = {
    'model_state_dict': model.state_dict(),
    'epoch': 10,
    'loss': 0.123
}
obs.ml.save_checkpoint(checkpoint, "acme/models/my_model_v1.pt")
```

## 📋 Configuration

Set your API credentials using environment variables:

```bash
export OSD_API_KEY="your-api-key-here"
export OSD_API_URL="https://api.osd.network"  # Optional
```

Or configure programmatically:

```python
import obsideo as obs

client = obs.Client(
    api_key="your-api-key",
    api_url="https://api.osd.network"
)
```

## 🔧 Core Features

### File Management
- **Upload/Download**: Simple file operations with the decentralized network
- **Namespacing**: Organize files using hierarchical namespaces
- **Metadata**: Attach custom metadata to files and datasets

### Data Science Integration
- **DataFrame Loading**: Load CSV, Parquet, JSON directly into pandas DataFrames
- **Multi-format Support**: Automatic format detection and loading
- **Structured Datasets**: Handle train/validation/test splits automatically

### ML Model Management  
- **Framework Support**: PyTorch, TensorFlow, scikit-learn, XGBoost, LightGBM
- **Checkpoints**: Save/load training checkpoints with metadata
- **Complete Models**: Save and load entire trained models
- **Version Control**: Track model versions and experiment metadata

## 📚 Examples

### Data Science Workflow

```python
import obsideo as obs

# Load a dataset for analysis
df = obs.data.load_dataframe("acme/datasets/sales_data.csv")
print(f"Dataset shape: {df.shape}")

# Load structured ML dataset
dataset = obs.data.load_dataset("acme/datasets/ml_experiment_1")
train_df = dataset["train"]
val_df = dataset["validation"]
test_df = dataset["test"]
```

### PyTorch Model Management

```python
import obsideo as obs
import torch

# Save checkpoint during training
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
obs.ml.save_checkpoint(checkpoint, "acme/models/resnet_epoch_10.pt")

# Load checkpoint to resume training
checkpoint = obs.ml.load_checkpoint("acme/models/resnet_epoch_10.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Save complete model for inference
obs.ml.save_model(model, "acme/models/resnet_final.pt", framework="pytorch")
```

### Scikit-learn Integration

```python
import obsideo as obs
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to OSD storage
obs.ml.save_model(
    model, 
    "acme/models/random_forest.pkl",
    framework="sklearn",
    metadata={"accuracy": 0.95, "features": 10}
)

# Load for inference
model = obs.ml.load_model("acme/models/random_forest.pkl", framework="sklearn")
predictions = model.predict(X_test)
```

## 🏗️ Development Status

**Current Version: 0.1.0 (Pre-Alpha)**

This is an early version of the SDK with method signatures and documentation in place. The backend implementation is in development, so most methods currently raise `NotImplementedError`.

### What's Working
- ✅ Package installation and imports
- ✅ Client configuration and initialization  
- ✅ Complete API surface with type hints and documentation
- ✅ Examples and usage patterns

### Coming Soon
- 🔄 Backend API integration
- 🔄 Actual file upload/download functionality
- 🔄 Real data loading and ML model management
- 🔄 Authentication and security features

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Website**: https://osd.network
- **Documentation**: https://docs.osd.network/obsideo
- **GitHub**: https://github.com/osdlabs/obsideo
- **PyPI**: https://pypi.org/project/obsideo

## 🏢 About OSD Labs

OSD Labs is building the future of decentralized storage for machine learning. Our mission is to make ML data management simple, secure, and decentralized.

---

*Built with ❤️ by the OSD Labs team*