import chromadb
import os

storage_path = "memory_index/search_index"
client = chromadb.PersistentClient(path=storage_path)
try:
    print(f"Deleting collection omd_search from {storage_path}...")
    client.delete_collection("omd_search")
    print("Done.")
except Exception as e:
    print(f"Error: {e}")
