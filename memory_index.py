import os
import json
import aiohttp
import numpy as np
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging
import warnings
from config import SETTINGS
from config import BASE_INDEX_DIR
from user_context import UserContext
import requests
from urllib.parse import urlparse
import re
from utils import   upload_vec_to_storage
import time
import chromadb
from chromadb.config import Settings

GATEWAY_URL = SETTINGS["GATEWAY_URL"]


MEMORY_KEYWORDS = [
    "Запомнить:",        # 🇷🇺 Russian
    "Memorize:",         # 🇬🇧 English
    "Remember:",         # 🇬🇧 alt
    "記住：",             # 🇹🇼 Traditional Chinese
    "记住：",             # 🇨🇳 Simplified Chinese
    "Erinnere dich:",    # 🇩🇪 German
    "Recuerda:",         # 🇪🇸 Spanish
    "Souviens-toi :",    # 🇫🇷 French
    "Memoriza:",         # 🇵/🇪 Spanish/Portuguese
    "覚えておいて：",      # 🇯🇵 Japanese
    "Zapomni si:",        # 🇸🇮 Slovenian
    "Zapamti:",           # 🇷🇸 Serbian/Croatian
]

SUPPORTED_CONVERT_EXTS = {"docx", "odt", "pdf", "epub", "fb2", "csv"}
SUPPORTED_PLAIN_EXTS = {"txt", "htm", "html", "xml", "md", "markdown"}


_model = None

def get_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Check if CPU is forced in settings or environment
        if SETTINGS.get("FORCE_CPU_EMBEDDINGS", "false").lower() == "true" or os.environ.get("OMD_FORCE_CPU") == "true":
            device = "cpu"
            
        logging.info(f"Loading SentenceTransformer on {device}...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model

CHROMA_DB_DIR = os.path.join(BASE_INDEX_DIR, "chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

def ensure_collection():
    global _chroma_client
    try:
        # Test the existing client/collection
        col = _chroma_client.get_or_create_collection(name="omd", metadata={"hnsw:space": "cosine"})
        # Verify it actually works by calling a lightweight method
        col.count()
        return col
    except Exception as e:
        logging.warning(f"ChromaDB collection stale or missing, re-initializing: {e}")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        return _chroma_client.get_or_create_collection(name="omd", metadata={"hnsw:space": "cosine"})

_collection = ensure_collection()

def migrate_legacy_data():
    try:
        if ensure_collection().count() > 0:
            return
        
        logging.info("Migrating legacy RAG data to ChromaDB...")
        # Migrate shared collections and document chunks
        if os.path.exists(BASE_INDEX_DIR):
            for root, dirs, files in os.walk(BASE_INDEX_DIR):
                for file in files:
                    if file.endswith((".jsonl", ".vec")):
                        # Determine collection name from path
                        rel_path = os.path.relpath(root, BASE_INDEX_DIR)
                        if rel_path == ".":
                             collection_name = file.replace(".jsonl", "").replace(".vec", "")
                        else:
                             collection_name = rel_path.replace(os.sep, "_")
                        
                        path = os.path.join(root, file)
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                memories = [json.loads(line) for line in f if line.strip()]
                                if memories:
                                    ids = []
                                    embeddings = []
                                    documents = []
                                    metadatas = []
                                    for i, m in enumerate(memories):
                                        mem_id = m.get("memory_id") or m.get("chunk_id")
                                        doc_id = m.get("document_id")
                                        if doc_id and mem_id:
                                             ids.append(f"{doc_id}:{mem_id}")
                                        else:
                                             ids.append(mem_id or str(uuid.uuid4()))
                                             
                                        embeddings.append(m["embedding"])
                                        documents.append(m["text"])
                                        
                                        meta = {k: v for k, v in m.items() if k not in ["embedding", "text", "user_id"]}
                                        meta["collection"] = collection_name
                                        meta.setdefault("relevance", "contextual")
                                        
                                        # Force owner=alexey for internal links and clean document_id
                                        doc_id = meta.get("document_id", "")
                                        if doc_id and doc_id.startswith("http") and "onmydisk.net" in doc_id:
                                            try:
                                                from urllib.parse import urlparse
                                                parsed = urlparse(doc_id)
                                                # Use full path as doc_id, but owner is always alexey
                                                meta["owner"] = "alexey"
                                                # Strip the owner part from path if it was there? 
                                                # User said "сейчас без владельца", so maybe path IS the doc_id.
                                                # Let's just take the whole path and let alexey own it.
                                                meta["document_id"] = parsed.path.lstrip("/")
                                            except:
                                                meta["owner"] = "alexey"
                                        else:
                                            meta.setdefault("owner", "alexey")
                                            
                                        metadatas.append(meta)
                                    
                                    ensure_collection().upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
                                    logging.info(f"Migrated {file} to collection {collection_name} ({len(memories)} items)")
                        except Exception as fe:
                            logging.error(f"Error migrating {path}: {fe}")

    except Exception as e:
        logging.error(f"Migration error: {e}")

# migrate_legacy_data() # Moved to explicit call in api.py


def embed_text(text: str) -> list:
    return get_model().encode(text, show_progress_bar=False).tolist()

def get_index_path(user_id: str | None = None, collection: str = "user") -> str:
    return f"{BASE_INDEX_DIR}/{collection}.jsonl"





def save_memories(ctx: UserContext, memories: list[dict], collection: str = "user"):
    """
    Deprecated for local storage (now uses ChromaDB), but kept for OMD sync compatibility.
    """
    try:
        if (
            ctx.storage
            and ctx.omd_key
            and collection == "user"
        ):
            # dest = f"{ctx.storage}"
            # upload_vec_to_storage(ctx.omd_key, dest, "memory.jsonl", memories, "application/jsonl")
            pass # redundant: handled by OrbitDB in frontend
        else:
            # локальный fallback (JSONL) - keeping for backup/migration if needed
            path = get_index_path(ctx.user_id, collection)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for m in memories:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[memory] Save error: {ctx.user_id} {collection} {e}")


def add_memory_card(
    ctx: UserContext,
    text: str,
    collection: str = "user",
    relevance: str = "contextual",
    document_id: str | None = None,
    mem_id: str | None = None
):
    """
    Добавляет или обновляет запись в ChromaDB.
    """
    if not mem_id:
        mem_id = str(uuid.uuid4())

    vec = embed_text(text)

    metadata = {
        "collection": collection,
        "relevance": relevance,
        "memory_id": mem_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "owner": "alexey"  # Default owner
    }

    if document_id:
        # Extract owner if it's a full OMD URL
        if document_id.startswith("http") and "onmydisk.net" in document_id:
            try:
                from urllib.parse import urlparse
                parsed = urlparse(document_id)
                parts = parsed.path.strip("/").split("/", 1)
                if len(parts) == 2:
                    metadata["owner"] = parts[0]
                    document_id = parts[1]
                else:
                    metadata["owner"] = "alexey"
                    document_id = parts[0]
            except:
                pass
        
        metadata["document_id"] = document_id
        # Если это документ, удаляем старые записи этого документа в этой коллекции с ТЕМ ЖЕ типом важности (обычно сводка)
        ensure_collection().delete(where={"$and": [
            {"document_id": document_id}, 
            {"collection": collection},
            {"relevance": relevance}
        ]})

    ensure_collection().upsert(
        ids=[mem_id],
        embeddings=[vec],
        metadatas=[metadata],
        documents=[text.strip()]
    )

    # Legacy fallback: also save to JSONL if desired (optional, but good for migration period)
    # try:
    #     memories = load_memories(ctx, collection)
    #     memories = [m for m in memories if m.get("memory_id") != mem_id]
    #     memories.append({**metadata, "embedding": vec, "text": text.strip()})
    #     save_memories(ctx, memories, collection)
    # except: pass

    return mem_id

def update_memory_card(
    ctx: UserContext,
    text: str,
    collection: str = "user",
    relevance: str = "contextual",
    document_id: str | None = None,
    mem_id: str | None = None
):
    """
    Обновить существующую карточку памяти в ChromaDB.
    """
    return add_memory_card(
        ctx=ctx,
        text=text,
        collection=collection,
        relevance=relevance,
        document_id=document_id,
        mem_id=mem_id,
    )



def delete_memory_card(
    ctx: UserContext,
    mem_id: str | None = None,
    document_id: str | None = None,
    collection: str = "user"
) -> bool:
    """
    Удаляет карточку памяти из ChromaDB.
    """
    try:
        if document_id:
            where = {"$and": [{"collection": collection}, {"document_id": document_id}]}
            ensure_collection().delete(where=where)
            return True
        elif mem_id:
            where = {"$and": [{"collection": collection}, {"memory_id": mem_id}]}
            ensure_collection().delete(where=where)
            return True
        return False
    except Exception as e:
        print(f"[memory] Delete error: {ctx.user_id} {collection} {e}")
        return False



def extract_memory_from_response(response: str) -> str | None:
    # Use regex to find keywords at the beginning of a line to avoid matching instructions in the system prompt
    # We look for keywords followed by a space or newline.
    for keyword in MEMORY_KEYWORDS:
        # Escape keyword for regex just in case
        esc_kw = re.escape(keyword)
        # Match keyword at the start of the string or after a newline
        pattern = rf"(?:^|\n){esc_kw}\s*(.*)"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        
        if match:
            remaining = match.group(1).strip()
            
            # Explicitly strip out any trailing LLM reasoning blocks 
            # while fully preserving legitimate multiline facts.
            # We also stop if we encounter another "Memorize:" or similar, 
            # or if we see common instruction markers like "DO NOT" or "Chat History"
            # which might indicate the model is echoing the prompt.
            
            # 1. Strip 'reason:' block
            fact = re.sub(r'(?i)\n*reason:[\s\S]*$', '', remaining).strip()
            
            # 2. Strip instructions if the model starts echoing the system prompt
            # (e.g. "DO NOT output the reason", "Chat History", etc.)
            fact = re.split(r'(?i)\n*(?:DO NOT output|Chat History|NEVER make up|\*SCOPE OF FACTS\*)', fact)[0].strip()
            
            if fact and len(fact) < 1000: # Sanity check for fact length
                return fact
    return None


    
def search_document_chunks(
    ctx: UserContext,    
    query: str,
    vec_path: str,
    vec_file: str,
    collection: str,
    top_k: int = 6,
    distance_threshold: float = 0.8,
    document_id: str | None = None
) -> list[dict]:
    """
    Поиск фрагментов документа в ChromaDB.
    """
    query_emb = embed_text(query)
    
    where = {"collection": collection}
    if document_id:
        where = {"$and": [{"collection": collection}, {"document_id": document_id}]}
    elif vec_file:
        # Fallback if only vec_file is provided (strip .vec extension)
        # document_id was likely the source of vec_file
        pass # Better to use document_id if available

    coll = ensure_collection()
    try:
        results = coll.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where
        )
        
        out = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                dist = results['distances'][0][i]
                if dist <= distance_threshold:
                    out.append({
                        "text": results['documents'][0][i],
                        "distance": dist,
                        "source": "document",
                        "document_id": results['metadatas'][0][i].get("document_id", ""),
                        "relevance": "contextual",
                    })
        return out
    except Exception as e:
        print(f"[memory] Search chunks error: {e}")
        return []


def load_memories(ctx: UserContext, collection: str = "user") -> list[dict]:
    """
    Загружает воспоминания. Сначала пытается из ChromaDB, затем из legacy JSONL.
    """
    try:
        where = {"collection": collection}
            
        results = ensure_collection().get(where=where, include=['documents', 'metadatas', 'embeddings'])
        if results['ids']:
            memories = []
            for i in range(len(results['ids'])):
                mem = {
                    **results['metadatas'][i],
                    "text": results['documents'][i],
                }
                if results['embeddings']:
                    mem["embedding"] = results['embeddings'][i]
                memories.append(mem)
            return memories
    except Exception as e:
        print(f"[memory] Chroma load error: {e}")

    # Legacy fallback
    try:
        if (
            ctx.storage
            and ctx.omd_key
            and collection == "user"
        ):
            logging.info(f"loading memories from OMD {ctx.storage}")
            url = f"{GATEWAY_URL}/{ctx.storage}/memory.jsonl?nocache={int(time.time())}"
            headers = {"authorization": f"token:{ctx.omd_key}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.content.strip():
                text = resp.content.decode("utf-8")
                return [json.loads(line) for line in text.splitlines() if line.strip()]
            return []
        else:
            index_path = f"{BASE_INDEX_DIR}/{collection}.jsonl"

            if not os.path.exists(index_path):
                return []

            with open(index_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[memory] Legacy load error: {e}")
        return []

def search_memories(ctx: UserContext, query: str, collection: str = "user", mem_id = "", top_k: int = 5, distance_threshold: float = 0.8) -> list[dict]:
    """
    Поиск воспоминаний в ChromaDB.
    """
    query_emb = embed_text(query)
    
    coll = ensure_collection()
    
    # 1. Permanent memories
    permanent_where = {"$and": [{"collection": collection}, {"relevance": "permanent"}]}
        
    try:
        perm_results = coll.get(where=permanent_where)
        permanent = []
        if perm_results['ids']:
            for i in range(len(perm_results['ids'])):
                doc_id = perm_results['metadatas'][i].get("document_id")
                if doc_id:
                    chunks = search_document_chunks(ctx, query, "", "", collection, document_id=doc_id, distance_threshold=0.8)
                    permanent.extend(chunks)
                else:
                    permanent.append({
                        **perm_results['metadatas'][i],
                        "text": perm_results['documents'][i],
                        "source": "memory"
                    })

        # 2. Contextual memories
        if mem_id:
            # Прямая ссылка по ID
            ctx_results = ensure_collection().get(where={"memory_id": mem_id})
        else:
            # Match contextual OR missing relevance
            contextual_where = {"$and": [
                {"collection": {"$eq": collection}},
                {"relevance": {"$ne": "permanent"}}
            ]}

            coll = ensure_collection()
            ctx_results = coll.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=contextual_where
            )

        contextual = []
        if ctx_results['ids']:
            # results['ids'] - это список списков для query, но плоский для get
            is_query = isinstance(ctx_results['ids'][0], list)
            ids = ctx_results['ids'][0] if is_query else ctx_results['ids']
            metas = ctx_results['metadatas'][0] if is_query else ctx_results['metadatas']
            docs = ctx_results['documents'][0] if is_query else ctx_results['documents']
            distances = ctx_results['distances'][0] if (is_query and 'distances' in ctx_results) else [0]*len(ids)

            for i in range(len(ids)):
                dist = distances[i]
                meta = metas[i]
                
                if dist <= distance_threshold or mem_id:
                    doc_id = meta.get("document_id")
                    # If it's already a chunk (has chunk_id), take it directly
                    if meta.get("chunk_id") is not None:
                        contextual.append({
                            **meta,
                            "text": docs[i],
                            "distance": dist,
                            "source": "memory"
                        })
                    # If it's a document link (summary), search for its chunks
                    elif doc_id:
                        chunks = search_document_chunks(ctx, query, "", "", collection, document_id=doc_id, distance_threshold=0.8)
                        contextual.extend(chunks)
                    else:
                        contextual.append({
                            **meta,
                            "text": docs[i],
                            "distance": dist,
                            "source": "memory"
                        })
        
        logging.debug(f"[memory] Search finished. Found {len(permanent + contextual)} total items.")
        return permanent + contextual
    except Exception as e:
        logging.error(f"[memory] Search error: {e}")
        return []

async def fetch_document_text(url: str, token: str = None) -> str:
    # URL construction
    if "?" not in url:
        ext = os.path.splitext(url.split("?")[0])[-1].lower().lstrip('.')
        if ext in SUPPORTED_CONVERT_EXTS:
           url += "?totext"
    
    # Assertive token handling: Append to URL immediately
    if token and "token=" not in url:
        separator = "&" if "?" in url else "?"
        url += f"{separator}token={token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as resp:
            status = resp.status
            text = await resp.text()
            
            # Detection of login page (indicates authentication failure even if 200)
            is_login = "FileManagerApp" in text or "login" in text.lower()[:500]
            
            if status == 204:
                return f"Failed to fetch document: Document is empty or could not be converted."
            
            if status != 200 or is_login:
                msg = "Access denied (403/401) or invalid token." if (status in [401, 403] or is_login) else f"HTTP Error {status}"
                logging.error(f"[fetch] Failed to retrieve document: {url.split('?')[0]} (Status {status}, Login {is_login})")
                return f"Failed to fetch document: {msg}"

    if not text or not text.strip():
        return f"Failed to fetch document: Unsupported file type or document has no selectable text (e.g. scanned image)."

    # Conversion for HTML
    if url.split("?")[0].lower().endswith((".html", ".htm")) or "<html" in text.lower()[:100]:
        try:
            import subprocess
            cmd = ["pandoc", "-f", "html", "-t", "markdown"]
            result = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
            return result.stdout
        except Exception as e:
            logging.error(f"[fetch] pandoc failed: {e}")
            return text

    return text



def chunk_and_vectorize_to_file(
    ctx: UserContext,
    text: str,
    document_id: str,
    collection: str = "user",
    chunk_size: int = 500,
    overlap: int = 50
):
    """
    Чанкает текст и сохраняет эмбеддинги в ChromaDB.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += chunk_size - overlap

    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for idx, chunk in enumerate(chunks):
        vec = embed_text(chunk)
        ids.append(f"{document_id}:{idx}")
        embeddings.append(vec)
        metadatas.append({
            "collection": collection,
            "document_id": document_id,
            "chunk_id": str(idx),
            "relevance": "document_chunk",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        documents.append(chunk)

    try:
        # Сначала удаляем старые чанки ЭТОГО документа в этой коллекции
        ensure_collection().delete(where={"$and": [
            {"document_id": document_id}, 
            {"collection": collection},
            {"relevance": "document_chunk"}
        ]})
        
        ensure_collection().upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
    except Exception as e:
        print(f"[memory] Chunk upsert error: {e}")

    # Report usage to console
    from utils import report_usage
    report_usage(ctx.omd_key, "kb_chunk", float(len(chunks)))

    return len(chunks)

