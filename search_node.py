
import os
import logging
import requests
import chromadb
from chromadb.config import Settings
from lxml import etree
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from datetime import datetime
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

class SearchNode:
    def __init__(self, storage_path: str, model_name: str = "all-MiniLM-L6-v2", token: str = ""):
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(storage_path, "search_index"))
        self.collection = self.chroma_client.get_or_create_collection(name="omd_search")
        self.model = SentenceTransformer(model_name)
        self.token = token
        
    def _fetch_content(self, url: str) -> str:
        headers = {}
        if self.token:
            headers["Authorization"] = self.token
            
        try:
            logger.info(f"Fetching {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

    def _parse_omd_index(self, content: str, base_url: str):
        """
        Parses the <omd_index> XML response.
        Expected format:
        <omd_index>
            <doc url='...' contentType='...' owner='...' last_modified='...'>
                <title>...</title>
                <description>...</description>
            </doc>
            ...
        </omd_index>
        """
        items = []
        try:
            # Handle potential encoding issues or malformed XML
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(content.encode('utf-8'), parser=parser)
            
            if root.tag != 'omd_index':
                logger.warning("Root tag is not omd_index")
                return items

            for doc in root.findall('doc'):
                item = {
                    'url': doc.get('url'),
                    'contentType': doc.get('contentType'),
                    'owner': doc.get('owner'),
                    'last_modified': doc.get('last_modified'),
                    'title': '',
                    'description': '',
                    'snippet': ''
                }
                
                title_elem = doc.find('title')
                if title_elem is not None:
                    item['title'] = title_elem.text or ''
                    
                desc_elem = doc.find('description')
                if desc_elem is not None:
                    item['description'] = desc_elem.text or ''
                    
                # Construct absolute URL/Item Path logic
                # The 'url' attribute in XML might be relative or a query string (e.g., ?readme)
                # We need to map this back to the OnMyDisk path structure if possible,
                # or store it in a way that allows retrieval.
                # However, for the PeARS drop-in, expected search results usually contain 'path' or 'itemPath'.
                # Let's see what the crawler does.
                # The crawler was called with ?url=<OMD_URL>/<PATH>
                # The XML returned describes that PATH (and its children if it's a folder).
                
                # If the 'url' attr is like "?readme", it refers to the container path.
                # If it's a child listing (not fully visible in the snippet I saw, but implied),
                # we might need to handle recursion if the XML contained children. 
                # But the snippet showed <doc> tags being appended to responseXml.
                
                # For now, let's treat the 'snippet' as a combination of title and description
                item['snippet'] = f"{item['title']}\n{item['description']}"
                
                items.append(item)
                
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            
        return items

    def index_url(self, url: str):
        """
        Crawls the given URL and indexes the content.
        This roughly mimics the PeARS /indexer/from_crawl behavior.
        """
        content = self._fetch_content(url)
        if not content:
            return {"status": "error", "message": "Failed to fetch content"}
            
        # Parse the XML index
        parsed_items = self._parse_omd_index(content, url)
        
        # Prepare for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        # Extract the base path from the URL query param if possible, 
        # but here 'url' is the direct link to the resource being crawled.
        # The URL passed to from_crawl is usually "http://omd-server/path/to/resource"
        
        parsed_url = urlparse(url)
        # simplistic path mapping
        path = parsed_url.path
        
        for i, item in enumerate(parsed_items):
            # Generate a unique ID. 
            # If the XML 'url' is relative (e.g. "?readme"), we construct a unique ID based on the crawl URL.
            # If it's a site sub-page, it might have a full URL.
            
            doc_id = f"{url}#{i}"
            
            # Simple text representation for embedding
            text_to_embed = f"{item['title']} {item['description']}"
            
            ids.append(doc_id)
            documents.append(text_to_embed)
            
            # Create metadata
            # Ensure all values are strings/ints/floats for chroma
            meta = {
                "itemPath": path, # Store the crawl path as the item path for now. 
                                  # Refinement: if parsing children, iterate their names?
                                  # The XML analysis showed it returns information about the *requested resource*.
                                  # Does it return children? 
                                  # "if (! m_rootResource->isSharedRoot()) { Q_FOREACH(QString entryMame, localListing) ... }"
                                  # Yes, it iterates children and adds them to responseXml (though some parts seemed Cut off in snippet).
                                  # Assuming one <doc> per child or one <doc> for the folder + docs for children.
                                  # Wait, the snippet showed "responseXml += ..." for the folder itself or its children.
                                  # It seems to flatten them into a list of <doc> elements.
                
                "title": item['title'],
                "description": item['description'],
                "owner": item['owner'] or "",
                "snippet": item['snippet'],
                "contentType": item['contentType'] or ""
            }
            metadatas.append(meta)
            
        if documents:
            embeddings = self.model.encode(documents).tolist()
            
            # Upsert into Chroma
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            logger.info(f"Indexed {len(documents)} items from {url}")
            return {"status": "ok", "indexed": len(documents)}
        else:
            logger.warning(f"No items found to index for {url}")
            return {"status": "ok", "indexed": 0}

    def delete_path(self, path: str):
        """
        Removes items related to the path.
        PeARS usage: /api/urls/delete?path=...
        """
        # We need to find items where itemPath starts with path or equals path.
        # ChromaDB delete supports 'where' filter.
        try:
            # This deletes where itemPath == path exactly. 
            # To delete children recursively, we might need a regex or simply delete exact match 
            # if the crawler logic sends delete for every file.
            # However, usually delete is for a folder. Chroma 'where' operator doesn't support 'startswith' natively yet in all versions?
            # Actually, standard Chroma where clause is exact match.
            # We might need to query first or iterate.
            # For now, let's implement exact match deletion which covers files. 
            # For folders, this might be incomplete without 'startswith'.
            
            self.collection.delete(where={"itemPath": path})
            
            # Hack for recursive: fetching all might be too heavy? 
            # Ideally we'd store a 'parentPath' or look into substring matching if supported.
            
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error deleting path {path}: {e}")
            return {"status": "error", "message": str(e)}

    def move_path(self, src: str, target: str):
        """
        PeARS usage: /api/urls/move?src=...&target=...
        """
        # Retrieve all items with src path, update them to target path
        try:
            results = self.collection.get(where={"itemPath": src})
            if results and results['ids']:
                updates = []
                for meta in results['metadatas']:
                    meta['itemPath'] = target
                    updates.append(meta)
                
                self.collection.update(
                    ids=results['ids'],
                    metadatas=updates
                )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error moving path {src} to {target}: {e}")
            return {"status": "error", "message": str(e)}

    def search(self, query: str, limit: int = 10):
        """
        Semantic search.
        Returns entries compatible with main.js expectations.
        """
        try:
            query_embedding = self.model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit
            )
            
            # Map to expected JSON format
            # main.js expects: { key: { itemPath:..., snippet:..., title:..., description:... } } 
            # or array? code said "for (var searchItem in data)" -> object iteration or array iteration.
            
            out_results = []
            
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    meta = results['metadatas'][0][i]
                    item = {
                        "itemPath": meta.get("itemPath", ""),
                        "snippet": meta.get("snippet", ""),
                        "title": meta.get("title", ""),
                        "description": meta.get("description", "")
                    }
                    out_results.append(item)
                    
            return out_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
