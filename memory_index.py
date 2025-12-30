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
from config import USER_DATA_DIR
from user_context import UserContext
import requests
from urllib.parse import urlparse
import re
from utils import   upload_vec_to_storage
import time

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





def save_memories(ctx: UserContext, memories: list[dict], collection: str = "user"):
    try:
        if (
            ctx.storage
            and ctx.omd_key
            and collection == "user"
        ):
            dest = f"{ctx.storage}"
            upload_vec_to_storage(ctx.omd_key, dest, "memory.jsonl", memories, "application/jsonl")
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
    document_id: str | None = None,
    mem_id: str | None = None
):
    """
    Добавляет или обновляет запись в указанном индексе памяти (локально или в OMD).
    - Если передан mem_id → обновляем карточку с этим ID.
    - Если передан document_id (и relevance == "document") → заменяем все записи документа.
    - Если не указано — создаём новую карточку.
    """

    # Загружаем существующие карточки
    memories = []
    try:
        memories = load_memories(ctx, collection)
    except Exception as e:
        print(f"[memory] Empty memory: {ctx.user_id} {collection} {e}")

    # Если есть document_id и это документ — удаляем старые записи документа
    if document_id:
        memories = [m for m in memories if m.get("document_id") != document_id]

    # Если есть mem_id — удаляем запись с этим ID
    if mem_id:
        memories = [m for m in memories if m.get("memory_id") != mem_id]
    else:
        mem_id = str(uuid.uuid4())  # UUID стабильнее, чем timestamp

    vec = embed_text(text)

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

    memories.append(entry)

    try:
        save_memories(ctx, memories, collection)
    except Exception as e:
        print(f"[memory] Save error: {ctx.user_id} {collection} {e}")

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
    Обновить существующую карточку памяти.
    Если указан document_id — перезаписывается карточка документа.
    Если указан mem_id — перезаписывается конкретная карточка.
    Если ни то, ни другое не указано — создаётся новая карточка.
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
    Удаляет карточку памяти по mem_id или document_id.
    Если передан document_id — удаляется все карточки с этим document_id.
    Если mem_id — удаляется только одна карточка.
    Возвращает True, если что-то было удалено.
    """
    try:
        memories = load_memories(ctx, collection)
        before_count = len(memories)

        if document_id:
            memories = [m for m in memories if m.get("document_id") != document_id]
        elif mem_id:
            memories = [m for m in memories if m.get("memory_id") != mem_id]

        if len(memories) < before_count:
            save_memories(ctx, memories, collection)
            return True
        return False
    except Exception as e:
        print(f"[memory] Delete error: {ctx.user_id} {collection} {e}")
        return False



def extract_memory_from_response(response: str) -> str | None:
    response_lower = response.lower()
    for keyword in MEMORY_KEYWORDS:
        keyword_lower = keyword.lower()
        pos = response_lower.find(keyword_lower)
        if pos != -1:
            start = pos + len(keyword)
            fact = response[start:].strip()
            return fact
    return None


def make_file_name_from_document_id(document_id: str) -> str:
    """
    Преобразует URL или путь в "безопасное" имя файла.
    Примеры:
      https://www.geeksforgeeks.org/python/introduction-to-python/
        → www.geeksforgeeks.org_python_introduction-to-python
      https://example.com/
        → example.com
      https://weird.site/q?x=1&y=2
        → weird.site
      file:///home/user/doc.txt
        → home_user_doc.txt
    """
    parsed = urlparse(document_id)

    if parsed.scheme in ("http", "https"):
        # Берём домен
        base = parsed.netloc
        # Убираем query и fragment, заменяем / на _
        path = re.sub(r"[?#].*", "", parsed.path).strip("/")
        if path:
            path = path.replace("/", "_")
            return f"{base}_{path}"
        else:
            return base

    elif parsed.scheme == "file":
        # Для file:/// → берём путь, заменяем / и \
        path = parsed.path.lstrip("/")
        safe_path = path.replace("/", "_").replace("\\", "_")
        return safe_path

    else:
        # Локальные document_id или fallback
        return document_id.replace("/", "_").replace("\\", "_")
    
def escape_text_field(line: str) -> dict:
    """
    Исправляет все кавычки, экранирует специальные символы и обрабатывает переносы строк в поле text.
    """
    # Проверяем, есть ли незакрытое поле text
    partial_text_match = re.search(r'("text"\s*:\s*)"((?:\\.|[^"])*)$', line)
    if partial_text_match:
        print("Найден разорванный блок 'text':")
        print(f"Содержимое: {partial_text_match.group(2)[:100]}...")
    
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
        print(f"Problematic line (first 100 chars): {line[:100]}...")
        raise


def search_document_chunks(
    ctx: UserContext,    
    query: str,
    vec_path: str,
    vec_file: str,
    collection: str,
    top_k: int = 6,
    distance_threshold: float = 0.6
) -> list[dict]:
    
    chunks = []
    logging.info(f"Loading document: {vec_file} with threshold {distance_threshold}")

    try:
        if collection == "user" and ctx.storage and ctx.omd_key:
            # Подгружаем vec из OMD
            url = f"{GATEWAY_URL}/{ctx.storage}/vecs/{vec_file}?nocache={int(time.time())}"
            token = ctx.omd_key
            headers = {"Authorization": f"token:{token}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                lines = resp.text.splitlines()
                #logging.info(f"Loaded document: {len(lines)}")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk_obj = json.loads(line)
                        chunks.append(chunk_obj)
                    except json.JSONDecodeError as e:
                        print("JSON error:", e)
        else:
            if not os.path.exists(vec_path):
                return []
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
            ctx.storage
            and ctx.omd_key
            and collection == "user"
        ):
            url = f"{GATEWAY_URL}/{ctx.storage}/memory.jsonl?nocache={int(time.time())}"
            headers = {"authorization": f"token:{ctx.omd_key}"}
            logging.info(f"loading memories from {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.content.strip():
                text = resp.content.decode("utf-8")
                return [json.loads(line) for line in text.splitlines() if line.strip()]
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
        print(f"[memory] No personal memories yet: {ctx.user_id} {collection} {e}")
        return []

def search_memories(ctx: UserContext, query: str, collection: str = "user", mem_id = "", top_k: int = 3, distance_threshold: float = 0.6) -> list[dict]:
    # Load memories
    memories = load_memories(ctx, collection)
    vec_path = ""

    if not collection == "user":
        vec_path = f"{BASE_INDEX_DIR}/{collection}"
    
    if not memories:
        return []

    query_emb = np.array(_model.encode([query], show_progress_bar=False)[0])

    permanent = []
    contextual = []

    for m in memories:
        relevance = m.get("relevance", "contextual")

        if relevance == "permanent":
            doc_id = m.get("document_id")
            if doc_id:
                vec_file = make_file_name_from_document_id(doc_id) + ".vec"
                # Adding relevant document chunks with lenient threshold
                doc_chunks = search_document_chunks(ctx, query, vec_path, vec_file, collection, distance_threshold=0.8)
                permanent.extend(doc_chunks)
            else:
                # Adding memory card
                permanent.append(m)
        elif relevance == "contextual":
            memory_id = m.get("memory_id")
            if mem_id and mem_id == memory_id:
                #direct reference
                distance = 0
            else:
                distance = cosine_distance(query_emb, m["embedding"])    
            doc_id = m.get("document_id")
            #logging.info(f"Checking memory: {collection} {memory_id} {distance} {doc_id}")
            if distance <= distance_threshold:
                logging.info(f"Relevant memory: {collection} {memory_id} {distance} {doc_id}")
                m["distance"] = distance
                doc_id = m.get("document_id")
                if doc_id:
                    vec_file = make_file_name_from_document_id(doc_id) + ".vec"
                    # Adding relevant document chunks with lenient threshold
                    doc_chunks = search_document_chunks(ctx, query, vec_path, vec_file, collection, distance_threshold=0.8)
                    contextual.extend(doc_chunks)
                else:
                  # Adding memory card
                  contextual.append(m)

    contextual.sort(key=lambda x: x["distance"])
    contextual = contextual[:top_k]

    return permanent + contextual

async def fetch_document_text(url: str, token: str = None) -> str:
    # URL construction
    if "?" not in url:
        ext = os.path.splitext(url.split("?")[0])[-1].lower().lstrip('.')
        if ext in SUPPORTED_CONVERT_EXTS:
           url += "?totext"
        elif ext in SUPPORTED_PLAIN_EXTS or not ext:
           pass
        else:
           # Do not return token in error
           return f"Unsupported file type: .{ext}"
    
    headers = {
        "Authorization": f"token:{token}" if token else ""
    }

    async with aiohttp.ClientSession() as session:
        async def do_fetch(current_url, current_headers):
            async with session.get(current_url, headers=current_headers, timeout=15) as resp:
                status = resp.status
                text = await resp.text()
                # Detection of login page
                is_login = "FileManagerApp" in text or "login" in text.lower()[:500]
                return status, text, is_login

        status, text, is_login = await do_fetch(url, headers)

        # Fallback to token in URL if header failed or returned login page
        if (status != 200 or is_login) and token and GATEWAY_URL in url:
            separator = "&" if "?" in url else "?"
            fallback_url = f"{url}{separator}token={token}"
            logging.info(f"[fetch] Header failed (status {status}, login: {is_login}), trying token in URL...")
            status, text, is_login = await do_fetch(fallback_url, {})

        if status != 200:
            return f"Failed to fetch document: {url.split('?')[0]} HTTP {status}"
        
        if is_login:
            return "Failed to fetch document: Received login page instead of content. check your storage link."

        # Conversion for HTML
        if url.split("?")[0].lower().endswith((".html", ".htm")) or "<html" in text.lower()[:100]:
            try:
                import subprocess
                cmd = ["pandoc", "-f", "html", "-t", "markdown"]
                result = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True)
                return result.stdout
            except Exception as e:
                logging.error(f"[fetch] pandoc failed: {e}")
                # Fallback to raw text if pandoc fails
                return text

        return text


def escape_chunk_text(text: str) -> str:
    """
    Экранирует кавычки, переносы строк и обратные слэши внутри текста,
    чтобы JSONL можно было безопасно читать.
    """
    # заменяем нестандартные кавычки на стандартные
    text = text.replace('“', '"').replace('”', '"').replace('„', '"')
    # экранируем обратные слэши
    text = text.replace('\\', '\\\\')
    # экранируем двойные кавычки
    text = text.replace('"', '\\"')
    # экранируем переносы строк и табуляции
    text = text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return text

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
            "text": escape_chunk_text(chunk),
            "chunk_id": str(idx),
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        entries.append(entry)

    try:
        if (
            ctx.storage
            and ctx.omd_key
            and collection == "user"
        ):
            # --- Хранение в OMD ---
            dest = f"{ctx.storage}/vecs"
            upload_vec_to_storage(ctx.omd_key, dest, f"{filename}.vec", entries, "application/jsonl")
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

