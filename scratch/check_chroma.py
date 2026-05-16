import chromadb
import os

def check_chroma():
    persist_dir = "/home/alexey/projects/omd/onmychat/memory_index/search_index"
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        col = client.get_collection("omd_search")
        res = col.get(where={"title": "Invoice-0A6BEAF1-0031.pdf"}, include=['metadatas'])
        print("Found items:")
        for i in range(len(res['ids'])):
            print(f"ID: {res['ids'][i]}, Meta: {res['metadatas'][i]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_chroma()
