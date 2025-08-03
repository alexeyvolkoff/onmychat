import os
import json
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

MEMORY_KEYWORDS = [
    "Запомнить:",        # 🇷🇺 Russian
    "Memorize:",         # 🇬🇧 English
    "Remember:",         # 🇬🇧 alt
    "記住：",             # 🇨🇳 Traditional Chinese
    "记住：",             # 🇨🇳 Simplified Chinese
    "Erinnere dich:",    # 🇩🇪 German
    "Recuerda:",         # 🇪🇸 Spanish
    "Souviens-toi :",    # 🇫🇷 French
    "Memoriza:",         # 🇵🇹/🇪🇸 Portuguese
    "覚えておいて：",      # 🇯🇵 Japanese
    "Zapomni si:",        # 🇸🇮 Slovenian
    "Zapamti:",           # 🇷🇸/🇭🇷/🇧🇦/🇲🇪
]


# Загрузка модели эмбеддинга
_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Пути к индексам (можно вынести в config.ini)
BASE_INDEX_DIR = "memory_index"
USER_INDEX_PREFIX = "user_"
SHARED_INDEX_FILE = "shared.jsonl"

def embed_text(text: str) -> list:
    """Получает эмбеддинг из строки"""
    return _model.encode(text).tolist()

def get_index_path(user_id: int | None = None, collection: str = "user") -> str:
    """Определяет путь к нужному .jsonl-файлу индекса"""
    if collection == "user":
        return os.path.join(BASE_INDEX_DIR, f"{USER_INDEX_PREFIX}{user_id}.jsonl")
    elif collection == "shared":
        return os.path.join(BASE_INDEX_DIR, SHARED_INDEX_FILE)
    else:
        raise ValueError(f"Неизвестная коллекция: {collection}")

def add_memory_card(text: str, user_id: int | None = None, collection: str = "user", relevance: str = "contextual"):
    """Добавляет запись в указанный индекс памяти"""
    index_path = get_index_path(user_id, collection)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    vec = embed_text(text)
    entry = {
        "embedding": vec,
        "text": text.strip(),
        "source": "memory" if collection == "user" else "file",
        "collection": collection,
        "relevance": relevance,
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
            # извлекаем фактический текст памяти после ключевого слова
            start = pos + len(keyword)
            fact = response[start:].strip()
            return fact, pos
    return None, None

def search_memories(
    query: str,
    user_id: int,
    collection: str = "user",
    top_k: int = 3,
    distance_threshold: float = 0.4
) -> list[dict]:
    global _model  # Модель SentenceTransformer уже инициализирована

    if collection == "user":
        index_path = f"memory_index/user_{user_id}.jsonl"
    elif collection == "shared":
        index_path = "memory_index/shared.jsonl"
    else:
        return []

    if not os.path.exists(index_path):
        return []

    # Загружаем воспоминания
    with open(index_path, "r", encoding="utf-8") as f:
        memories = [json.loads(line) for line in f if line.strip()]

    if not memories:
        return []

    # Векторизуем запрос
    query_emb = np.array(_model.encode([query])[0])

    # Фильтруем и сортируем
    permanent = []
    contextual = []

    def cosine_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for m in memories:
        relevance = m.get("relevance", "contextual")

        if relevance == "permanent":
            permanent.append(m)
        elif relevance == "contextual":
            distance = cosine_distance(query_emb, m["embedding"])
            if distance <= distance_threshold:
                m["distance"] = distance
                contextual.append(m)

    contextual.sort(key=lambda x: x["distance"])
    contextual = contextual[:top_k]

    return permanent + contextual



# Быстрый тест
if __name__ == "__main__":
    fact = "Запомнить: Однажды шлюз сам начал рекомендовать обновления."
    memory = extract_memory_from_response(fact)
    if memory:
        result = add_memory_card(memory, user_id=1551662876, collection="user")
        print("✔ Добавлено в личную память:", result["text"])
        result2 = add_memory_card("Папка /docs/shared содержит инструкции", collection="shared")
        print("✔ Добавлено в общую память:", result2["text"])

