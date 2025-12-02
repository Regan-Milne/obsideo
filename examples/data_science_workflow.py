"""
Data science workflow example using Obsideo SDK.

This example demonstrates how to use Obsideo for typical data science tasks:
loading datasets, working with DataFrames, and managing experiment results.
"""

import obsideo as obs

def data_science_workflow():
    """Example data science workflow using Obsideo."""
    
    # Create client
    client = obs.Client.from_env()
    
    # Example 1: Load a dataset directly into a DataFrame
    print("Loading dataset...")
    try:
        df = obs.data.load_dataframe(
            client,
            name="sales_data",
            namespace="acme/datasets",
            version="2025-12-01"
        )
        print(f"Loaded dataset with shape: {df.shape}")
        print(df.head())
    except NotImplementedError:
        print("DataFrame loading not yet implemented - backend coming soon!")
    except ImportError:
        print("pandas not installed - install with: pip install 'obsideo[data]'")
    
    # Example 2: Load a structured dataset (train/val/test)
    print("\nLoading structured dataset...")
    try:
        dataset = obs.data.load_dataset(
            client,
            name="ml_experiment_1",
            namespace="acme/datasets",
            version="latest"
        )
        print(f"Dataset contains: {list(dataset.keys())}")
        # Access different splits
        # train_df = dataset["train"]
        # val_df = dataset["validation"]
        # test_df = dataset["test"]
    except NotImplementedError:
        print("Dataset loading not yet implemented - backend coming soon!")
    
    # Example 3: Work with different data formats
    print("\nLoading different formats...")
    try:
        # Load Parquet file
        parquet_df = obs.data.load_dataframe(
            client,
            name="features",
            namespace="acme/datasets",
            version="latest"
        )
        
        # Load JSON Lines file  
        json_df = obs.data.load_dataframe(
            client,
            name="events", 
            namespace="acme/datasets",
            version="latest"
        )
        
        # Load Excel file
        excel_df = obs.data.load_dataframe(
            client,
            name="report",
            namespace="acme/datasets", 
            version="2025-12-01"
        )
        
        print("Successfully loaded multiple formats!")
        
    except NotImplementedError:
        print("Multi-format loading not yet implemented - backend coming soon!")
    except ImportError:
        print("Missing optional dependencies - install with: pip install 'obsideo[data]'")


if __name__ == "__main__":
    data_science_workflow()