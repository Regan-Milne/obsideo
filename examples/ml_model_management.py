"""
Machine learning model management example using Obsideo SDK.

This example demonstrates how to save and load ML models and checkpoints
using Obsideo for different frameworks.
"""

import obsideo as obs

def pytorch_example():
    """Example using PyTorch models and checkpoints."""
    print("PyTorch Model Management Example")
    print("=" * 40)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleNet()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Create client
        client = obs.Client.from_env()
        
        # Example 1: Save a checkpoint
        print("Saving checkpoint...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 10,
            'loss': 0.123
        }
        
        try:
            obs.ml.save_checkpoint(
                client,
                checkpoint=checkpoint,
                run_id="llm-finetune-001",
                step=1000,
                namespace="acme/checkpoints"
            )
            print("Checkpoint saved successfully!")
        except NotImplementedError:
            print("Checkpoint saving not yet implemented - backend coming soon!")
        
        # Example 2: Load a checkpoint
        print("Loading checkpoint...")
        try:
            loaded_checkpoint = obs.ml.load_checkpoint(
                client,
                run_id="llm-finetune-001", 
                step=1000,
                namespace="acme/checkpoints"
            )
            model.load_state_dict(loaded_checkpoint['model_state_dict'])
            optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from epoch {loaded_checkpoint['epoch']}")
        except NotImplementedError:
            print("Checkpoint loading not yet implemented - backend coming soon!")
        
        # Example 3: Save complete model
        print("Saving complete model...")
        try:
            obs.ml.save_model(
                client,
                model,
                name="simple_net",
                namespace="acme/models",
                version="v1.0",
                metadata={"architecture": "SimpleNet", "input_size": 10}
            )
            print("Model saved successfully!")
        except NotImplementedError:
            print("Model saving not yet implemented - backend coming soon!")
        
        # Example 4: Load complete model
        print("Loading complete model...")
        try:
            loaded_model = obs.ml.load_model(
                client,
                name="simple_net",
                namespace="acme/models", 
                version="v1.0"
            )
            print("Model loaded successfully!")
        except NotImplementedError:
            print("Model loading not yet implemented - backend coming soon!")
            
    except ImportError:
        print("PyTorch not installed - install with: pip install 'obsideo[ml]'")


def sklearn_example():
    """Example using scikit-learn models."""
    print("\nScikit-learn Model Management Example")
    print("=" * 40)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create client and train a simple model
        client = obs.Client.from_env()
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        # Save the trained model
        print("Saving scikit-learn model...")
        try:
            obs.ml.save_model(
                client,
                model,
                name="random_forest", 
                namespace="acme/models",
                version="v1.0",
                metadata={"algorithm": "RandomForest", "accuracy": 0.95}
            )
            print("Model saved successfully!")
        except NotImplementedError:
            print("Model saving not yet implemented - backend coming soon!")
        
        # Load the model
        print("Loading scikit-learn model...")
        try:
            loaded_model = obs.ml.load_model(
                client,
                name="random_forest",
                namespace="acme/models",
                version="v1.0"
            )
            predictions = loaded_model.predict(X[:5])
            print(f"Predictions: {predictions}")
        except NotImplementedError:
            print("Model loading not yet implemented - backend coming soon!")
            
    except ImportError:
        print("scikit-learn not installed - install with: pip install scikit-learn")


def main():
    """Run all ML model management examples."""
    pytorch_example()
    sklearn_example()


if __name__ == "__main__":
    main()