
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
    def __init__(self, storage_path: str, model: SentenceTransformer = None, model_name: str = "all-MiniLM-L6-v2", token: str = ""):
        self.storage_path = storage_path
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(storage_path, "search_index"))
        self._collection = self.chroma_client.get_or_create_collection(name="omd_search")
        self.model = model or SentenceTransformer(model_name)
        self.token = token
        
    def get_collection(self):
        try:
            # Test if collection is still valid
            self._collection.count()
            return self._collection
        except Exception:
            logger.warning("SearchNode collection stale, re-initializing")
            self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.storage_path, "search_index"))
            self._collection = self.chroma_client.get_or_create_collection(name="omd_search")
            return self._collection
        
    def _fetch_xml(self, url: str) -> str:
        headers = {}
        if self.token:
            headers["Authorization"] = f"token:{self.token}" if ":" not in self.token else self.token
            
        try:
            logger.info(f"Fetching XML from {url}")
            fetch_url = url if url.endswith('/') or '?' in url else url + '/'
            response = requests.get(fetch_url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

    def _fetch_file_content(self, url: str, convertible: bool, content_type: str) -> str:
        headers = {}
        if self.token:
            headers["Authorization"] = f"token:{self.token}" if ":" not in self.token else self.token
            
        try:
            fetch_url = url
            if convertible:
                fetch_url += "?totext"
            elif content_type in ["text/plain", "text/markdown", "application/json", "text/javascript", "text/css"]:
                fetch_url += "?direct"
            else:
                return ""

            logger.info(f"Fetching file content from {fetch_url}")
            response = requests.get(fetch_url, headers=headers, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching file content {url}: {e}")
            return ""

    def index_url(self, start_url: str):
        queue = [start_url]
        visited = set()
        total_indexed = 0
        
        while queue:
            current_url = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            
            xml_content = self._fetch_xml(current_url)
            if not xml_content or "<omd_index>" not in xml_content:
                continue
                
            items = self._parse_omd_index(xml_content, current_url)
            
            ids = []
            documents = []
            metadatas = []
            
            for item in items:
                item_name = item['url']
                if not item_name or item_name.startswith('?'):
                    continue
                
                item_url = urljoin(current_url.rstrip('/') + '/', item_name)
                
                if item['contentType'] == 'folder':
                    queue.append(item_url)
                    continue
                
                convertible = item.get('convertible') == 'True'
                file_text = self._fetch_file_content(item_url, convertible, item['contentType'])
                
                if not file_text:
                    continue
                
                doc_id = item_url
                text_to_index = f"{item['title']}\n{item['description']}\n{file_text}"[:10000]
                
                ids.append(doc_id)
                documents.append(text_to_index)
                
                parsed_url = urlparse(item_url)
                item_path = parsed_url.path
                
                meta = {
                    "url": item_url,
                    "itemPath": item_path,
                    "title": item['title'] or os.path.basename(item_path),
                    "description": item['description'] or "",
                    "owner": item['owner'] or "",
                    "contentType": item['contentType'] or "",
                    "last_modified": item['last_modified'] or ""
                }
                metadatas.append(meta)
                total_indexed += 1
                
            if ids:
                embeddings = self.model.encode(documents).tolist()
                self.get_collection().upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                logger.info(f"Batch indexed {len(ids)} items from {current_url}")

        return {"status": "ok", "indexed": total_indexed}

    def _parse_omd_index(self, content: str, base_url: str):
        items = []
        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(content.encode('utf-8'), parser=parser)
            
            for doc in root.findall('.//doc'):
                item = {
                    'url': doc.get('url'),
                    'contentType': doc.get('contentType'),
                    'owner': doc.get('owner'),
                    'last_modified': doc.get('last_modified'),
                    'convertible': doc.get('convertible'),
                    'title': '',
                    'description': ''
                }
                
                title_elem = doc.find('title')
                if title_elem is not None:
                    item['title'] = title_elem.text or ''
                    
                desc_elem = doc.find('description')
                if desc_elem is not None:
                    item['description'] = desc_elem.text or ''
                    
                items.append(item)
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        return items

    def delete_path(self, path: str):
        try:
            self.get_collection().delete(where={"itemPath": path})
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error deleting path {path}: {e}")
            return {"status": "error", "message": str(e)}

    def move_path(self, src: str, target: str):
        try:
            results = self.get_collection().get(where={"itemPath": src})
            if results and results['ids']:
                for i, doc_id in enumerate(results['ids']):
                    meta = results['metadatas'][i]
                    old_url = meta['url']
                    new_url = old_url.replace(src, target, 1)
                    meta['url'] = new_url
                    meta['itemPath'] = target
                    
                    doc_content = results['documents'][i]
                    self.get_collection().delete(ids=[doc_id])
                    self.get_collection().add(
                        ids=[new_url],
                        documents=[doc_content],
                        metadatas=[meta],
                        embeddings=[results['embeddings'][i]] if results['embeddings'] else None
                    )
            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Error moving path {src} to {target}: {e}")
            return {"status": "error", "message": str(e)}

    def search(self, query: str, limit: int = 20):
        try:
            query_embedding = self.model.encode([query]).tolist()
            results = self.get_collection().query(
                query_embeddings=query_embedding,
                n_results=limit
            )
            
            out_dict = {}
            if results['ids']:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    meta = results['metadatas'][0][i]
                    doc_text = results['documents'][0][i]
                    snippet = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                    
                    out_dict[doc_id] = {
                        "itemPath": meta.get("itemPath", ""),
                        "snippet": snippet,
                        "title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "owner": meta.get("owner", ""),
                        "contentType": meta.get("contentType", ""),
                        "last_modified": meta.get("last_modified", "")
                    }
            return out_dict
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {}

