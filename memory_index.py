import os
import json
import aiohttp
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import urllib.parse
import logging

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

_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

BASE_INDEX_DIR = "memory_index"
USER_INDEX_PREFIX = "user_"
SHARED_INDEX_FILE = "shared.jsonl"

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_text(text: str) -> list:
    return _model.encode(text).tolist()

def get_index_path(user_id: int | None = None, collection: str = "user") -> str:
    index_path = ""
    if collection == "user":
        index_path = f"{BASE_INDEX_DIR}/user_{user_id}.jsonl"
    else:
        index_path = f"{BASE_INDEX_DIR}/{collection}.jsonl"
    return index_path

def add_memory_card(
    text: str,
    user_id: int | None = None,
    collection: str = "user",
    relevance: str = "contextual",
    document_id: str | None = None
):
    """Добавляет запись в указанный индекс памяти"""
    index_path = get_index_path(user_id, collection)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    vec = embed_text(text)

    entry = {
        "embedding": vec,
        "text": text.strip(),
        "collection": collection,
        "relevance": relevance,
        "document_id": document_id,
        "user_id": user_id if collection == "user" else None,
        "memory_id": datetime.now().strftime("%Y%m%dT%H%M%S") + (f"_user_{user_id}" if user_id else ""),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


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
    return file_name + ".vec"


def search_document_chunks(
    query: str,
    vec_path: str,
    top_k: int = 3,
    distance_threshold: float = 0.6
) -> list[dict]:
    """Ищет релевантные чанки в .vec-файле документа"""
    if not os.path.exists(vec_path):
        return []

    try:
        with open(vec_path, "r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️ Ошибка чтения {vec_path}: {e}")
        return []

    if not chunks:
        return []

    query_emb = np.array(_model.encode([query])[0])

    results = []
    for chunk in chunks:
        distance = cosine_distance(query_emb, chunk["embedding"])
        logging.info(f"Chunk distance: {distance}")
        if distance <= distance_threshold:
            results.append({
                "text": chunk.get("text", "").strip(),
                "distance": distance,
                "source": "document",
                "document_id": chunk.get("document_id", "").strip(),
                "relevance": "contextual",
            })

    results.sort(key=lambda x: x["distance"])
    return results[:top_k]


def search_memories(query: str, user_id: int, collection: str = "user", top_k: int = 3, distance_threshold: float = 0.6) -> list[dict]:
    if collection == "user":
        index_path = f"memory_index/user_{user_id}.jsonl"
    else:
        index_path = f"memory_index/{collection}.jsonl"

    vec_path = ""
    if collection == "user":
        vec_path = f"user_data/docs/user_{user_id}.jsonl"
    else:
        vec_path = f"user_data/docs/{collection}.jsonl"


    if not os.path.exists(index_path):
        return []

    with open(index_path, "r", encoding="utf-8") as f:
        memories = [json.loads(line) for line in f if line.strip()]

    if not memories:
        return []

    query_emb = np.array(_model.encode([query])[0])

    permanent = []
    contextual = []

    for m in memories:
        relevance = m.get("relevance", "contextual")

        if relevance == "permanent":
            permanent.append(m)
        elif relevance == "contextual":
            distance = cosine_distance(query_emb, m["embedding"])
            memory_id = m.get("memory_id")
            logging.info(f"{index_path} {memory_id} {distance}")
            if distance <= distance_threshold:
                m["distance"] = distance
                doc_id = m.get("document_id")
                if doc_id:
                    logging.info(f"Relevant document: {doc_id}")
                    vec_path = os.path.join(f"{vec_path}", make_file_name_from_document_id(doc_id))
                    if os.path.exists(vec_path):
                        # Adding relevant document chunks
                        doc_chunks = search_document_chunks(query, vec_path)
                        contextual.extend(doc_chunks)
                else:
                  # Adding memory card
                  contextual.append(m)

    contextual.sort(key=lambda x: x["distance"])
    contextual = contextual[:top_k]

    return permanent + contextual

async def fetch_document_text(url: str, token: str) -> str:
    headers = {"Authorization": f"token:{token}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to fetch document: {resp.status}")
            return await resp.text()

def save_vec_file(vectors: list[list[float]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)


def chunk_and_vectorize_to_file(user_id: int, text: str,  document_id: str, collection: str = "user", chunk_size: int = 500, overlap: int = 50):
    """Чанкает текст и сохраняет эмбеддинги в .vec файл"""
    vec_path = ""
    if collection == "user":
        vec_path = f"user_data/docs/user_{user_id}.jsonl"
    else:
        vec_path = f"user_data/docs/{collection}.jsonl"
    vec_path = os.path.join(f"{vec_path}", make_file_name_from_document_id(document_id))
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += chunk_size - overlap

    with open(vec_path, "w", encoding="utf-8") as f:
        for idx, chunk in enumerate(chunks):
            vec = embed_text(chunk)
            entry = {
                "embedding": vec,
                "text": chunk,
                "chunk_id": str(idx),
                "document_id": document_id,
                "timestamp": datetime.now().isoformat(timespec="seconds")
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(chunks)

