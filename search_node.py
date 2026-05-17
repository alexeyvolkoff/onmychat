
import os
import logging
import requests
import chromadb
import json
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
        self._collection = self.chroma_client.get_or_create_collection(name="omd_search", metadata={"hnsw:space": "cosine"})
        self.model = model or SentenceTransformer(model_name)
        self.token = token
        self.blacklist = [
            "node_modules", ".git", ".venv", "venv", "__pycache__", 
            "site-packages", "bin", "obj", "target", "dist", "build",
            ".cache", ".idea", ".vscode", ".metadata", ".ruff_cache",
            ".pytest_cache", "vendor", "bower_components", ".yarn",
            ".npm", ".pnpm", ".gradle", ".terraform", ".next", ".nuxt"
        ]
        
    def get_collection(self):
        try:
            # Test if collection is still valid
            self._collection.count()
            return self._collection
        except Exception:
            logger.warning("SearchNode collection stale, re-initializing")
            self.chroma_client = chromadb.PersistentClient(path=os.path.join(self.storage_path, "search_index"))
            self._collection = self.chroma_client.get_or_create_collection(name="omd_search", metadata={"hnsw:space": "cosine"})
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
        total_skipped = 0
        
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
                    item_name_lower = item_name.lower()
                    if any(b.lower() in item_name_lower for b in self.blacklist):
                        logger.info(f"Skipping blacklisted folder: {item_url}")
                        continue
                    queue.append(item_url)
                    continue
                
                # List of users/groups that can access this item
                allowed_entities = []
                item_owner = item.get('owner', '')
                if item_owner:
                    allowed_entities.append(item_owner)
                
                shared_with = item.get('shared_with', '')
                if shared_with:
                    entities = [e.strip() for e in shared_with.split(',') if e.strip()]
                    allowed_entities.extend(entities)
                
                # If no owner or shared_with, default to alexey for now (compatibility)
                if not allowed_entities:
                    allowed_entities = ['alexey']

                logger.info(f"Indexing {item_url} for entities: {allowed_entities}")
                # Check if item already exists and hasn't changed
                # We check the primary entry (actual owner)
                primary_id = f"{item_url}:{allowed_entities[0]}"
                try:
                    # We also check how many entries we have for this URL
                    # to see if shared_with has changed
                    existing_entries = self.get_collection().get(
                        where={"url": item_url},
                        include=['metadatas']
                    )
                    
                    if existing_entries and existing_entries['metadatas']:
                        primary_meta = None
                        for m in existing_entries['metadatas']:
                            if m.get('owner') == item_owner:
                                primary_meta = m
                                break
                        
                        if primary_meta and primary_meta.get('last_modified') == item['last_modified']:
                            # Check if the number of shared entries matches
                            if len(existing_entries['metadatas']) == len(allowed_entities):
                                logger.info(f"Skipping unchanged file and permissions: {item_url}")
                                total_skipped += 1
                                continue
                            else:
                                logger.info(f"Permissions changed for {item_url}, re-indexing...")
                                # Delete old entries before re-indexing
                                self.get_collection().delete(where={"url": item_url})
                except Exception as e:
                    logger.warning(f"Error checking existing item {item_url}: {e}")

                convertible = item.get('convertible') == 'True'
                file_text = self._fetch_file_content(item_url, convertible, item['contentType'])
                
                if not file_text:
                    continue
                
                text_to_index = f"{item['title']}\n{item['description']}\n{file_text}"[:10000]
                
                parsed_url = urlparse(item_url)
                item_path = parsed_url.path

                # Add an entry for EACH allowed entity
                for entity in allowed_entities:
                    doc_id = f"{item_url}:{entity}"
                    ids.append(doc_id)
                    documents.append(text_to_index)
                    
                    meta = {
                        "url": item_url,
                        "itemPath": item_path,
                        "title": item['title'] or os.path.basename(item_path),
                        "description": item['description'] or "",
                        "owner": entity, # Store the specific entity as owner for filtering
                        "contentType": item['contentType'] or "",
                        "last_modified": item['last_modified'] or ""
                    }
                    metadatas.append(meta)
                
                total_indexed += 1
                
            if ids:
                embeddings = self.model.encode(documents, show_progress_bar=False).tolist()
                self.get_collection().upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                logger.info(f"Batch indexed {len(ids)} items from {current_url}")

        return {"status": "ok", "indexed": total_indexed, "skipped": total_skipped}

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
                    'shared_with': doc.get('shared_with', ''),
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

    def search(self, query: str, limit: int = None, owner: str = "", ctx = None):
        try:
            # Load SEARCH_TOP_K from SETTINGS if limit not provided
            from config import SETTINGS
            if limit is None:
                limit = int(SETTINGS.get("SEARCH_TOP_K", "20"))

            query_embedding = self.model.encode([query], show_progress_bar=False).tolist()
            
            allowed_owners = set()
            if owner:
                allowed_owners.add(owner)
            elif ctx:
                allowed_owners.add(ctx.user_id)
                if hasattr(ctx, 'groups') and ctx.groups:
                    for g in ctx.groups:
                        if g: allowed_owners.add(g)
                allowed_owners.add('public') # Always allow public content
            
            where = None
            if allowed_owners:
                allowed_owners_list = list(allowed_owners)
                where = {"owner": {"$in": allowed_owners_list}}
                
            logger.info(f"Searching for '{query}' with filter: {where}")
            
            # Load SEARCH_THRESHOLD from SETTINGS
            from config import SETTINGS
            threshold = float(SETTINGS.get("SEARCH_THRESHOLD", "0.75"))
            
            # Query a larger number of items so we can filter and compute the actual total matching count
            query_limit = max(limit * 5, 100)
            results = self.get_collection().query(
                query_embeddings=query_embedding,
                n_results=query_limit,
                where=where
            )
            
            out_dict = {}
            valid_results = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_id = results['ids'][0][i]
                    meta = results['metadatas'][0][i]
                    doc_text = results['documents'][0][i]
                    dist = results['distances'][0][i]
                    
                    if dist <= threshold:
                        valid_results.append({
                            "doc_id": doc_id,
                            "meta": meta,
                            "doc_text": doc_text,
                            "distance": dist
                        })
            
            # Explicitly sort by distance (relevance) ascending
            valid_results.sort(key=lambda x: x["distance"])
            total_found = len(valid_results)
            
            # Limit results
            limited_results = valid_results[:limit]
            
            # Include service metadata record
            out_dict["system_info"] = {
                "is_system": True,
                "total_found": total_found,
                "returned_limit": limit,
                "relevance_threshold": threshold
            }
            
            for item in limited_results:
                doc_id = item["doc_id"]
                meta = item["meta"]
                doc_text = item["doc_text"]
                dist = item["distance"]
                snippet = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                
                out_dict[doc_id] = {
                    "itemPath": meta.get("itemPath", ""),
                    "snippet": snippet,
                    "title": meta.get("title", ""),
                    "description": meta.get("description", ""),
                    "owner": meta.get("owner", ""),
                    "contentType": meta.get("contentType", ""),
                    "last_modified": meta.get("last_modified", ""),
                    "relevance": round((1.0 - dist) * 100, 1) # Relevance percentage score
                }
                
            return out_dict
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {}


