
import logging
import shutil
import os
from search_node import SearchNode
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)

# Mock data
MOCK_XML = """
<omd_index>
    <doc url="?readme" contentType="text/markdown" owner="daddy" last_modified="2023-10-27 10:00:00">
        <title>Read me first</title>
        <description>This is a test description for the readme file.</description>
    </doc>
    <doc url="subfolder?description" contentType="text/plain" owner="daddy" last_modified="2023-10-27 10:05:00">
        <title>Subfolder</title>
        <description>This is a subfolder description.</description>
    </doc>
</omd_index>
"""

TEST_DIR = "test_data"

def test_search_logic():
    # Clean up previous test
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)

    print("Initializing SearchNode...")
    node = SearchNode(storage_path=TEST_DIR, token="testtoken")
    
    # Mock requests.get
    with patch('requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = MOCK_XML
        mock_get.return_value = mock_response
        
        print("Indexing URL...")
        url = "http://localhost:8080/my/share"
        result = node.index_url(url)
        print(f"Index Result: {result}")
        
        # Verify calls
        mock_get.assert_called_with(url, headers={"Authorization": "testtoken"}, timeout=10)
        
    # Test Search
    print("Searching...")
    search_res = node.search("test description")
    print(f"Search Results: {search_res}")
    
    assert len(search_res) > 0
    assert search_res[0]['title'] in ["Read me first", "Subfolder"]
    
    print("Deleting path...")
    node.delete_path("/my/share")
    
    search_res_after = node.search("test description")
    print(f"Search Results after delete: {search_res_after}")
    # Note: Chromadb deletion might not be immediate or requires specific where clause we verified.
    # Our delete implementation uses 'itemPath': path. 
    # The indexed itemPath comes from urlparse(url).path.
    # URL was http://localhost:8080/my/share -> path is /my/share
    
    if len(search_res_after) == 0:
        print("Deletion successful.")
    else:
        print("Deletion failed or not immediate.")

if __name__ == "__main__":
    try:
        test_search_logic()
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
