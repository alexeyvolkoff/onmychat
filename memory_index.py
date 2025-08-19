import os
import json
import aiohttp
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging
import warnings
from config import USER_DATA_DIR
from user_context import UserContext
import requests
from utils import  upload_data_to_storage, upload_vec_to_storage

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


warnings.filterwarnings("ignore", category=FutureWarning)
_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

BASE_INDEX_DIR = "memory_index"



def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_text(text: str) -> list:
    return _model.encode(text, show_progress_bar=False).tolist()

def get_index_path(user_id: str | None = None, collection: str = "user") -> str:
    index_path = ""
    if collection == "user":
        index_path = f"{USER_DATA_DIR}/{user_id}/memory.jsonl"
    else:
        index_path = f"{BASE_INDEX_DIR}/{collection}.jsonl"
    return index_path


def load_memories(ctx: UserContext, collection: str = "user") -> list[dict]:
    try:
        if (
            ctx.type == "omd"
            and ctx.settings.get("storage")
            and ctx.settings.get("omd_key")
            and collection == "user"
        ):
            url = f"https://onmydisk.net/{ctx.settings['storage']}/{ctx.user_id}/memory.jsonl"
            token = ctx.settings["omd_key"]
            headers = {"Authorization": f"token:{token}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                return [json.loads(line) for line in resp.text.splitlines() if line.strip()]
            return []
        else:
            # локальный fallback
            path = f"{USER_DATA_DIR}/{ctx.user_id}/memory/{collection}.jsonl"
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"[memory] Load error: {ctx.user_id} {collection} {e}")
        return []


def save_memories(ctx: UserContext, memories: list[dict], collection: str = "user"):
    try:
        if (
            ctx.type == "omd"
            and ctx.settings.get("storage")
            and ctx.settings.get("omd_key")
            and collection == "user"
        ):
            dest = f"{ctx.settings['storage']}/{ctx.user_id}"
            upload_vec_to_storage(ctx.settings['omd_key'], dest, "memory.jsonl", memories, "application/jsonl")
        else:
            # локальный fallback
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
    document_id: str | None = None
):
    """Добавляет запись в указанный индекс памяти (локально или в OMD)"""

    vec = embed_text(text)
    mem_id = datetime.now().strftime("%Y%m%dT%H%M%S") + (f"_user_{ctx.user_id}" if ctx.user_id else "")

    entry = {
        "embedding": vec,
        "text": text.strip(),
        "collection": collection,
        "relevance": relevance,
        "user_id": ctx.user_id,
        "memory_id": mem_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if document_id:
        entry["document_id"] = document_id

    try:
        # Загружаем существующие карточки
        memories = load_memories(ctx, collection)
        memories.append(entry)
        save_memories(ctx, memories, collection)
    except Exception as e:
        print(f"[memory] Add error: {ctx.user_id} {collection} {e}")

    return mem_id


def extract_memory_from_response(response: str) -> tuple[str | None, int | None]:
    response_lower = response.lower()
    for keyword in MEMORY_KEYWORDS:
        keyword_lower = keyword.lower()
        pos = response_lower.find(keyword_lower)
        if pos != -1:
            start = pos + len(keyword)
            fact = response[start:].strip()
            return fact, pos
    return None, None


def make_file_name_from_document_id(document_id: str) -> str:
    """
    Преобразует document_id (обычно это URL) в безопасное имя файла.
    """
    import urllib.parse
    import os

    # Декодируем URL и берём имя файла
    decoded = urllib.parse.unquote(document_id)
    file_name = os.path.basename(decoded)
    file_name = file_name.replace("/", "_").replace("\\", "_")
    return file_name


def search_document_chunks(
    ctx: UserContext,    
    query: str,
    vec_path: str,
    vec_file: str,
    top_k: int = 3,
    distance_threshold: float = 0.6
) -> list[dict]:
    """Ищет релевантные чанки в .vec-файле документа"""
    if not os.path.exists(vec_path):
        return []
    
    chunks = []

    try:
        if ctx.type == "omd" and ctx.settings.get("storage") and ctx.settings.get("omd_key"):
            # Подгружаем vec из OMD
            url = f"https://onmydisk.net/{ctx.settings['storage']}/{ctx.user_id}/vecs/{vec_file}"
            token = ctx.settings["omd_key"]
            headers = {"Authorization": f"token:{token}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                chunks = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
        else:
            vec_file_path = f"{vec_path}/{vec_file}"
            with open(vec_file_path, "r", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️ Ошибка чтения {vec_path}: {e}")
        return []

    if not chunks:
        return []

    query_emb = np.array(_model.encode([query], show_progress_bar=False)[0])

    results = []
    for chunk in chunks:
        distance = cosine_distance(query_emb, chunk["embedding"])
        if distance <= distance_threshold:
            logging.info(f"Relevant chunk: {distance}")
            results.append({
                "text": chunk.get("text", "").strip(),
                "distance": distance,
                "source": "document",
                "document_id": chunk.get("document_id", "").strip(),
                "relevance": "contextual",
            })

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]


def load_memories(ctx: UserContext, collection: str = "user") -> list[dict]:
    try:
        if (
            ctx.type == "omd"
            and ctx.settings.get("storage")
            and ctx.settings.get("omd_key")
            and collection == "user"
        ):
            url = f"https://onmydisk.net/{ctx.settings['storage']}/{ctx.user_id}/memory.jsonl"
            headers = {"authorization": f"token:{ctx.settings['omd_key']}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                return [json.loads(line) for line in resp.text.splitlines() if line.strip()]
            return []
        else:
            # локальный fallback
            if collection == "user":
                index_path = f"{USER_DATA_DIR}/{ctx.user_id}/memory.jsonl"
            else:
                index_path = f"{BASE_INDEX_DIR}/{collection}.jsonl"

            if not os.path.exists(index_path):
                return []

            memories = []
            logging.info(f"loading memories {index_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                memories = [json.loads(line) for line in f if line.strip()]
            return   memories


    except Exception as e:
        print(f"[memory] Load error: {ctx.user_id} {collection} {e}")
        return []

def search_memories(ctx: UserContext, query: str, collection: str = "user", top_k: int = 3, distance_threshold: float = 0.6) -> list[dict]:
    # Load memories
    memories = load_memories(ctx, collection)
    vec_path = ""

    if collection == "user":
        vec_path = f"{USER_DATA_DIR}/{ctx.user_id}/vec"
    else:
        vec_path = f"{BASE_INDEX_DIR}/{collection}"
    
    if not memories:
        return []

    query_emb = np.array(_model.encode([query], show_progress_bar=False)[0])

    permanent = []
    contextual = []

    for m in memories:
        relevance = m.get("relevance", "contextual")

        if relevance == "permanent":
            permanent.append(m)
        elif relevance == "contextual":
            distance = cosine_distance(query_emb, m["embedding"])
            memory_id = m.get("memory_id")
            doc_id = m.get("document_id")
            #logging.info(f"Checking memory: {collection} {memory_id} {distance} {doc_id}")
            if distance <= distance_threshold:
                logging.info(f"Relevant memory: {collection} {memory_id} {distance} {doc_id}")
                m["distance"] = distance
                doc_id = m.get("document_id")
                if doc_id:
                    vec_file = make_file_name_from_document_id(doc_id) + ".vec"
                    # Adding relevant document chunks
                    doc_chunks = search_document_chunks(ctx, query, vec_path, vec_file)
                    contextual.extend(doc_chunks)
                else:
                  # Adding memory card
                  contextual.append(m)

    contextual.sort(key=lambda x: x["distance"])
    contextual = contextual[:top_k]

    return permanent + contextual

async def fetch_document_text(url: str, token: str = None) -> str:
    headers = {}
    if token:
        ext = os.path.splitext(url.split("?")[0])[-1].lower().lstrip('.')
        if ext in SUPPORTED_CONVERT_EXTS:
           url += "?totext"
        elif ext in SUPPORTED_PLAIN_EXTS:
           pass
        else:
           raise ValueError(f"Unsupported file type: .{ext}")
        headers["Authorization"] = f"token:{token}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                raise ValueError(f"Failed to fetch document: HTTP {response.status}")
            return await response.text()


def chunk_and_vectorize_to_file(
    ctx: UserContext,
    text: str,
    document_id: str,
    collection: str = "user",
    chunk_size: int = 500,
    overlap: int = 50
):
    """Чанкает текст и сохраняет эмбеддинги в .vec файл (локально или в OMD)"""

    filename = make_file_name_from_document_id(document_id)
    chunks = []
    words = text.split()
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += chunk_size - overlap

    # --- Формируем записи ---
    entries = []
    for idx, chunk in enumerate(chunks):
        vec = embed_text(chunk)
        entry = {
            "embedding": vec,
            "text": chunk,
            "chunk_id": str(idx),
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        entries.append(entry)

    try:
        if (
            ctx.type == "omd"
            and ctx.settings.get("storage")
            and ctx.settings.get("omd_key")
            and collection == "user"
        ):
            # --- Хранение в OMD ---
            dest = f"{ctx.settings['storage']}/{ctx.user_id}/vecs"
            upload_vec_to_storage(ctx.settings['omd_key'], dest, f"{filename}.vec", entries, "application/jsonl")
        else:
            # --- Локальное хранение ---
            if collection == "user":
                vec_dir = f"{USER_DATA_DIR}/{ctx.user_id}/vecs"
            else:
                vec_dir = f"{BASE_INDEX_DIR}/{collection}"

            os.makedirs(vec_dir, exist_ok=True)
            vec_path = os.path.join(vec_dir, f"{filename}.vec")

            with open(vec_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[vec] Save error: {ctx.user_id} {collection} {e}")

    return len(chunks)

