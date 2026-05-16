import chromadb
import logging

def test_chroma():
    client = chromadb.PersistentClient(path="./test_chroma")
    col = client.get_or_create_collection("test")
    
    col.add(
        ids=["1", "2"],
        documents=["doc1", "doc2"],
        metadatas=[{"shared": ["bob", "alice"]}, {"shared": ["bob"]}]
    )
    
    print("Testing exact match...")
    res = col.query(query_texts=["doc"], where={"shared": "bob"})
    print(f"Found with 'bob': {res['ids']}")
    
    try:
        print("Testing $contains...")
        res = col.query(query_texts=["doc"], where={"shared": {"$contains": "bob"}})
        print(f"Found with $contains 'bob': {res['ids']}")
    except Exception as e:
        print(f"$contains failed: {e}")

if __name__ == "__main__":
    test_chroma()
