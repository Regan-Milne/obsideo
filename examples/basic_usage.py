"""
Basic usage examples for the Obsideo SDK.

This example demonstrates the core workflows for uploading and downloading
files using the OSD decentralized storage network.
"""

import obsideo as obs

def main():
    # Create client using environment variables
    # Set OSD_API_KEY in your environment
    client = obs.Client.from_env()
    
    # Example 1: Upload a file
    print("Uploading file...")
    try:
        file_id = client.upload_file("data.csv", namespace="acme/datasets")
        print(f"File uploaded with ID: {file_id}")
    except NotImplementedError:
        print("Upload not yet implemented - backend coming soon!")
    except FileNotFoundError:
        print("data.csv not found - create a sample file first")
    
    # Example 2: Download a file
    print("\nDownloading file...")
    try:
        client.download_file("acme/datasets/data.csv", "downloaded_data.csv")
        print("File downloaded successfully!")
    except NotImplementedError:
        print("Download not yet implemented - backend coming soon!")
    
    # Example 3: List files in a namespace
    print("\nListing files...")
    try:
        files = client.list_files("acme/datasets")
        print(f"Found {len(files)} files:")
        for file in files:
            print(f"  - {file}")
    except NotImplementedError:
        print("List not yet implemented - backend coming soon!")
    
    # Example 4: Get file information
    print("\nGetting file info...")
    try:
        info = client.get_file_info("acme/datasets/data.csv")
        print(f"File info: {info}")
    except NotImplementedError:
        print("File info not yet implemented - backend coming soon!")


if __name__ == "__main__":
    main()