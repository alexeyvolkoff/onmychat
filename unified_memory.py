"""
unified_memory.py  —  OMD 3.0 RAG
Единая ChromaDB коллекция omd_unified вместо трёх legacy баз.

Схема metadata:
    type        : "memory_card" | "file_chunk" | "qa"
    tags        : строка через запятую, напр. "omd,docs,personal"
    owner       : user_id владельца записи
    document_id : исходный путь/URL документа
    chunk_id    : "0", "1", ... (для чанков)
    title       : заголовок
    relevance   : "contextual" | "permanent" | "document_chunk"
    timestamp   : ISO-строка
"""

import os
import math
import time
import uuid
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import SETTINGS, BASE_INDEX_DIR
import user_context

logger = logging.getLogger(__name__)

# ─── Пути ────────────────────────────────────────────────────────────────────

UNIFIED_DB_DIR = os.path.join(BASE_INDEX_DIR, "unified_index")
os.makedirs(UNIFIED_DB_DIR, exist_ok=True)

# Публичные теги — для гостей (читается из config.ini)
PUBLIC_TAGS: list[str] = [
    t.strip()
    for t in SETTINGS.get("PUBLIC_KNOWLEDGE_TAGS", "omd").split(",")
    if t.strip()
]

RAG_THRESHOLD   = float(SETTINGS.get("RAG_THRESHOLD",   "0.75"))
SEARCH_THRESHOLD = float(SETTINGS.get("SEARCH_THRESHOLD", "0.75"))
RAG_TOP_K       = int(SETTINGS.get("RAG_TOP_K",   "8"))
SEARCH_TOP_K    = int(SETTINGS.get("SEARCH_TOP_K", "20"))

# Буст «свежести» файла при RAG-поиске (факт-инъекциях):
# более новый файл получает усиление к оценке релевантности.
# RECENCY_WEIGHT        - максимальный множитель (0 = отключено)
# RECENCY_HALF_LIFE_DAYS- возраст файла в днях, на котором буст падает вдвое
RECENCY_WEIGHT         = float(SETTINGS.get("RECENCY_WEIGHT", "0.35"))
RECENCY_HALF_LIFE_DAYS = float(SETTINGS.get("RECENCY_HALF_LIFE_DAYS", "30"))

# ─── Embedding model ──────────────────────────────────────────────────────────
# Мультиязычная модель: связывает запросы/теги на разных языках (напр. «Ужгород»
# ↔ "Uzhhorod"). Смените через `EMBEDDING_MODEL` в config.ini.
EMBEDDING_MODEL = SETTINGS.get("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if (
            SETTINGS.get("FORCE_CPU_EMBEDDINGS", "false").lower() == "true"
            or os.environ.get("OMD_FORCE_CPU") == "true"
        ):
            device = "cpu"
        logger.info(f"[unified] Loading SentenceTransformer {EMBEDDING_MODEL} on {device}...")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def embed(text: str) -> list:
    return get_model().encode(text, show_progress_bar=False).tolist()


# ─── ChromaDB клиент ──────────────────────────────────────────────────────────

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=UNIFIED_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    """Возвращает коллекцию omd_unified, пересоздаёт при ошибке."""
    global _collection, _client
    try:
        if _collection is not None:
            _collection.count()
            return _collection
    except Exception:
        logger.warning("[unified] Collection stale, re-initializing")
        _client = None
        _collection = None

    _collection = _get_client().get_or_create_collection(
        name="omd_unified",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ─── Вспомогательные ─────────────────────────────────────────────────────────

def _normalize_tag(t: str) -> str:
    """' #OMD  ' -> 'omd'"""
    return t.strip().lstrip("#").lower()


def _tags_meta(tags: list) -> dict:
    """
    ['omd', '#docs'] -> {'t_omd': 1, 't_docs': 1, 'tags': 'omd,docs'}
    Каждый тег — отдельное поле t_<tag>: 1 (для $eq-фильтрации).
    Поле 'tags' сохраняется как человекочитаемая строка.
    """
    clean = [_normalize_tag(t) for t in tags if t.strip()]
    meta = {"tags": ",".join(clean)}
    for t in clean:
        meta[f"t_{t}"] = 1
    return meta


def _tags_match_filter(tags_filter: list) -> dict:
    """
    ChromaDB where-фильтр: хотя бы один из тегов присутствует.
    Использует отдельные t_<tag> поля с $eq (обход бага $contains в ChromaDB 1.5.x).
    """
    if not tags_filter:
        return {}
    if len(tags_filter) == 1:
        return {f"t_{_normalize_tag(tags_filter[0])}": {"$eq": 1}}
    return {"$or": [{f"t_{_normalize_tag(t)}": {"$eq": 1}} for t in tags_filter]}


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / _norm(a) / _norm(b) if _norm(a) and _norm(b) else 0.0

def _norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def _parse_meta_time(value) -> Optional[datetime]:
    """
    ISO-строка метады (source_stamp — mtime файла, либо timestamp — время
    индексации) → aware-datetime в локальной таймзоне. None при ошибке.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def _recency_factor(ts: datetime, now: datetime, half_life_days: float) -> float:
    """Гармонический спад: 1.0 при ts == now, ~0.5 на half_life, → 0 со временем."""
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 1.0 / (1.0 + age_days / max(half_life_days, 1.0))


def _apply_recency_rerank(rows: list, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> None:
    """
    Переранжирует найденные записи по relevance × recency на месте (in-place):
    свежие файлы/заметки поднимаются выше при сопоставимой релевантности.
    Для каждой записи заполняет recency_factor, recency_boost и combined final_score.
    """
    if RECENCY_WEIGHT <= 0:
        return
    now = datetime.now().astimezone()
    for r in rows:
        ts = _parse_meta_time(r.get("source_stamp") or r.get("timestamp"))
        if ts is not None:
            rf = _recency_factor(ts, now, half_life_days)
            r["recency_factor"] = round(rf, 4)
            r["recency_boost"]  = round(RECENCY_WEIGHT * rf, 4)
        else:
            r["recency_factor"] = 0.0
            r["recency_boost"]  = 0.0
        r["final_score"] = round(float(r.get("relevance", 0.0)) * (1.0 + r.get("recency_boost", 0.0)), 3)
    rows.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)


# ─── Поиск ───────────────────────────────────────────────────────────────────

def search(
    query: str,
    tags_filter=None,
    owner=None,
    top_k=None,
    threshold=None,
    record_types=None,
    document_id=None,
    recent_first=False,
) -> list:
    """
    Единая точка поиска.

    Args:
        query         - поисковый запрос
        tags_filter   - список тегов: только записи с хотя бы одним из тегов
                        (для гостевого режима: ["omd"])
        owner         - фильтр по владельцу
        top_k         - максимум результатов
        threshold     - косинусное расстояние (0=идентично, ниже=лучше)
        record_types  - фильтр по type: ["file_chunk", "memory_card", ...]
        document_id   - фильтр по document_id (напр. "/Documents/a.pdf" для
                        /learn — граничит факты конкретно этим файлом/карточкой)
        recent_first  - переранжировать по relevance × recency: свежие файлы
                        (source_stamp/timestamp) поднимаются выше
    """
    if top_k is None:
        top_k = RAG_TOP_K
    if threshold is None:
        threshold = RAG_THRESHOLD

    coll = get_collection()
    if coll.count() == 0:
        return []

    query_emb = embed(query)

    conditions = []
    if document_id:
        conditions.append({"document_id": {"$eq": document_id}})
    if tags_filter:
        conditions.append(_tags_match_filter(tags_filter))
    if owner:
        conditions.append({"owner": {"$eq": owner}})
    if record_types:
        if len(record_types) == 1:
            conditions.append({"type": {"$eq": record_types[0]}})
        else:
            conditions.append({"$or": [{"type": {"$eq": t}} for t in record_types]})

    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    try:
        fetch_k = max(top_k * 5, 50)
        n_results = min(fetch_k, max(coll.count(), 1))
        kwargs = dict(query_embeddings=[query_emb], n_results=n_results)
        if where:
            kwargs["where"] = where
        results = coll.query(**kwargs)
    except Exception as e:
        logger.error(f"[unified] search error: {e}")
        return []

    out = []
    ids   = results.get("ids",   [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    docs  = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for i in range(len(ids)):
        dist = dists[i] if i < len(dists) else 1.0
        if dist <= threshold:
            score = round((1.0 - dist) * 100, 1)
            row = {"id": ids[i], "text": docs[i], "distance": dist}
            row.update(metas[i])
            row["relevance"] = score
            row["score"] = score
            out.append(row)

    # Метадата-boost: токены запроса, буквально встречающиеся в document_id/title
    # (напр. «vieste» в «Bineon_račun_Vieste_29.pdf»),bridging multilingual-модели,
    # которая не тянет перекрёстные языки (račun ↮ invoice).
    try:
        import re as _re
        tokens = set(_re.findall(r"[a-z0-9]{4,}", query.lower()))
        if tokens:
            all_recs = coll.get(include=["metadatas"])
            kw_ids = [
                rid for rid, meta in zip(all_recs.get("ids", []), all_recs.get("metadatas", []))
                if (
                    (not document_id or meta.get("document_id") == document_id)
                    and any(
                        tok in f"{meta.get('document_id', '')} {meta.get('title', '')}".lower()
                        for tok in tokens
                    )
                )
            ]
            if kw_ids:
                # не дублируем уже найденные векторной выдачей
                got = set(ids)
                kw_ids = [rid for rid in kw_ids if rid not in got]
            if kw_ids:
                kw_docs = coll.get(ids=kw_ids, include=["embeddings", "metadatas", "documents"])
                kw_boost = float(SETTINGS.get("META_KEYWORD_BOOST", "0.8"))
                for rid, emb, meta, doc in zip(
                    kw_docs.get("ids", []),
                    kw_docs.get("embeddings", []),
                    kw_docs.get("metadatas", []),
                    kw_docs.get("documents", []),
                ):
                    dist = (1.0 - _cosine(query_emb, list(emb))) * kw_boost
                    if dist <= threshold:
                        score = round((1.0 - dist) * 100, 1)
                        row = {"id": rid, "text": doc or "", "distance": dist}
                        row.update(meta)
                        row["relevance"] = score
                        row["score"] = score
                        out.append(row)
    except Exception as e:
        logger.warning(f"[unified] metadata boost error: {e}")

    out.sort(key=lambda x: x["distance"])
    seen = set()
    deduped = []
    for r in out:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        deduped.append(r)
    if recent_first:
        _apply_recency_rerank(deduped)
    return deduped[:top_k]


# ─── Карточки памяти и сопроводительные Readme.md ──────────────────────────

def resolve_doc_path(document_id: str) -> str | None:
    """Виртуальный путь документа (document_id) → реальный путь на диске ноды,
    если файл/папка существует, иначе None."""
    if not document_id or document_id.startswith(("http", "user:")):
        return None
    import getpass
    user = getpass.getuser()

    # Варианты путей на диске
    candidates = []
    if os.path.exists(document_id):
        candidates.append(document_id)
    if document_id.startswith("/"):
        candidates.append(f"/home/{user}{document_id}")
    norm = "/" + document_id.strip("/")
    if norm.startswith("/home/"):
        parts = norm.split("/", 3)
        if len(parts) == 4:
            candidates.append(f"/home/{user}/{parts[3]}")

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def resolve_companion_readme(document_id: str) -> tuple[str | None, str | None]:
    """
    Разрешает путь к сопроводительному Readme.md на локальном диске для document_id.
    Возвращает (readme_path, content_if_exists).
    """
    if not document_id or document_id.startswith("http") or document_id.startswith("user:note:"):
        return None, None
    import getpass
    user = getpass.getuser()

    real_path = resolve_doc_path(document_id)
    if not real_path:
        real_path = f"/home/{user}/{document_id.lstrip('/')}"

    # Для папки: Readme.md внутри папки
    if os.path.isdir(real_path):
        for name in ("Readme.md", "readme.md", "README.md"):
            p = os.path.join(real_path, name)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        return p, f.read().strip()
                except Exception:
                    pass
        return os.path.join(real_path, "Readme.md"), None

    # Для файла: <file>.Readme.md (без исходного расширения, напр. photo.Readme.md)
    base_path = os.path.splitext(real_path)[0]
    for suffix in (".Readme.md", ".readme.md", ".README.md"):
        p = base_path + suffix
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return p, f.read().strip()
            except Exception:
                pass
        p_full = real_path + suffix
        if os.path.isfile(p_full):
            try:
                with open(p_full, "r", encoding="utf-8", errors="replace") as f:
                    return p_full, f.read().strip()
            except Exception:
                pass
    return base_path + ".Readme.md", None


def save_companion_readme(document_id: str, text: str) -> str | None:
    """Сохраняет текст в сопроводительный Readme.md на диске."""
    readme_path, _ = resolve_companion_readme(document_id)
    if not readme_path:
        return None
    try:
        os.makedirs(os.path.dirname(readme_path), exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(text.strip() + "\n")
        _invalidate_indexed_docs_cache()
        return readme_path
    except Exception as e:
        logger.error(f"[unified] save_companion_readme error {readme_path}: {e}")
        return None


def find_memory_card_id(document_id: str, title: str = None) -> str | None:
    """Возвращает memory_id существующей карточки по document_id (или title), либо None."""
    if not (document_id or title):
        return None
    conditions = [{"type": {"$eq": "memory_card"}}]
    if document_id:
        conditions.append({"document_id": {"$eq": document_id}})
    elif title:
        conditions.append({"title": {"$eq": title}})
    try:
        res = get_collection().get(where={"$and": conditions}, include=["metadatas"], limit=2)
        ids = res.get("ids", [])
        if not ids:
            return None
        if document_id:
            return ids[0]
        # По имени карточку документа ищем только среди карточек документов
        for i, rid in enumerate(ids):
            if res["metadatas"][i].get("document_id"):
                return rid
        return ids[0] if len(ids) == 1 else None
    except Exception as e:
        logger.error(f"[unified] find_memory_card_id error: {e}")
        return None


def upsert_memory_card(
    text: str,
    mem_id=None,
    owner=None,
    tags=None,
    title="",
    document_id=None,
    relevance="contextual",
    image_preview="",
) -> str:
    """Добавляет/обновляет карточку памяти. Возвращает mem_id.

    Дедуп: при document_id (или одинаковом title) обновляем существующую
    карточку этого документа вместо создания новой — старые дубли удаляются.
    """
    if not owner:
        owner = user_context.node_owner()

    # Сначала схлопываем старые карточки этого документа (если есть),
    # затем вставляем/обновляем свежую — на дисплее остаётся одна.
    if document_id:
        try:
            get_collection().delete(where={"$and": [
                {"type":        {"$eq": "memory_card"}},
                {"document_id": {"$eq": document_id}},
            ]})
        except Exception as e:
            logger.warning(f"[unified] dedupe stale memory_cards error: {e}")

    if not mem_id and (document_id or title):
        mem_id = None
    if not mem_id:
        mem_id = str(uuid.uuid4())

    metadata = {
        "type":      "memory_card",
        "owner":     owner,
        "title":     title,
        "relevance": relevance,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if image_preview:
        metadata["image_preview"] = image_preview
    metadata.update(_tags_meta(tags or []))
    if document_id:
        metadata["document_id"] = document_id

    get_collection().upsert(
        ids=[mem_id],
        embeddings=[embed(text)],
        documents=[text.strip()],
        metadatas=[metadata],
    )
    _invalidate_indexed_docs_cache()
    return mem_id


def update_memory_card(mem_id: str, text: str = None, title: str = None, tags=None) -> str | None:
    """Обновляет существующую memory_card (текст/заголовок/теги), сохраняя document_id/owner/relevance."""
    coll = get_collection()
    try:
        # Поддержка виртуального id документа document::<document_id>
        if mem_id and mem_id.startswith("document::"):
            doc_id = mem_id[len("document::"):]
            existing_id = find_memory_card_id(doc_id)
            if existing_id:
                mem_id = existing_id
            else:
                new_text = (text or "").strip()
                if not new_text:
                    _, readme_content = resolve_companion_readme(doc_id)
                    new_text = readme_content or ""
                return upsert_memory_card(
                    new_text,
                    document_id=doc_id,
                    title=title or doc_id.split("/")[-1],
                    tags=tags or [],
                    relevance="permanent",
                )

        res = coll.get(ids=[mem_id], include=["documents", "metadatas"])
        if not res.get("ids"):
            return None
        meta = res["metadatas"][0]
        old_text = res["documents"][0] if res.get("documents") else ""
        new_text = (text if text is not None else old_text).strip()
        old_tags = [t for t in (meta.get("tags") or "").split(",") if t]
        new_tags = [t for t in (tags or [])] if tags is not None else old_tags
        return upsert_memory_card(
            new_text,
            mem_id=mem_id,
            owner=meta.get("owner"),
            tags=new_tags,
            title=title if title is not None else meta.get("title", ""),
            document_id=meta.get("document_id"),
            relevance=meta.get("relevance", "contextual"),
            image_preview=meta.get("image_preview", ""),
        )
    except Exception as e:
        logger.error(f"[unified] update_memory_card error: {e}")
        return None


def delete_memory_card(mem_id=None, document_id=None):
    """Удаляет карточку памяти по id или document_id."""
    coll = get_collection()
    try:
        if mem_id:
            coll.delete(ids=[mem_id])
        elif document_id:
            coll.delete(where={"$and": [
                {"type": {"$eq": "memory_card"}},
                {"document_id": {"$eq": document_id}},
            ]})
        _invalidate_indexed_docs_cache()
    except Exception as e:
        logger.error(f"[unified] delete_memory_card error: {e}")


# ─── Кэш списка проиндексированных документов ─────────────────────────────────
# get_indexed_documents читает ВСЕ file_chunk + ВСЕ memory_card на каждый вызов
# (главный источник медленного /api/device_memory при тысячах записей).
# Кэшируем с TTL и инвалидируем при любой записи в unified коллекцию.

_INDEXED_DOCS_CACHE_TTL = 30.0
_indexed_docs_cache: dict = {"valid": False, "ts": 0.0, "data": None}
_cards_cache: dict = {"valid": False, "ts": 0.0, "data": None}


def _invalidate_indexed_docs_cache():
    _indexed_docs_cache["valid"] = False
    _cards_cache["valid"] = False


def _indexed_docs_cache_fresh() -> bool:
    return bool(
        _indexed_docs_cache["valid"]
        and (_indexed_docs_cache["ts"] + _INDEXED_DOCS_CACHE_TTL) > time.time()
    )


def get_all_memory_cards(owner=None) -> list:
    """Возвращает все карточки памяти (для листинга в UI).

    owner=None (UI-листинг) — результат кэшируется с TTL 30s + инвалидация
    при записи, обходить можно опциональным owner (см. get_indexed_documents).
    """
    coll = get_collection()
    try:
        if owner is None and _cards_cache["valid"] and (_cards_cache["ts"] + _INDEXED_DOCS_CACHE_TTL) > time.time():
            return [dict(c) for c in _cards_cache["data"]]
        where = {"type": {"$eq": "memory_card"}}
        if owner:
            where = {"$and": [where, {"owner": {"$eq": owner}}]}
        results = coll.get(where=where, include=["documents", "metadatas"])
        out = []
        for i, doc_id in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i]
            # Карточки-документы дублируются группой file_chunk из get_indexed_documents,
            # для них не плодим отдельные записи в списке.
            if meta.get("type") == "memory_card" and meta.get("document_id"):
                continue
            out.append({
                "memory_id": doc_id,
                "_id": doc_id,
                "text": results["documents"][i],
                **meta,
                "tags": [k[2:] for k in meta if k.startswith("t_")],
            })
        if owner is None:
            _cards_cache["valid"] = True
            _cards_cache["ts"] = time.time()
            _cards_cache["data"] = out
        return out
    except Exception as e:
        logger.error(f"[unified] get_all_memory_cards error: {e}")
        return []


def get_indexed_documents(owner=None) -> list:
    """Возвращает проиндексированные документы (file_chunk), сгруппированные по document_id.

    Результат кэшируется (TTL 30s + инвалидация при записи), т.к. перечитывание
    ВСЕХ чанков и карточек на каждый вызов стоит секунды при тысячах записей.
    Опциональный owner минует кэш (на практике вызовы идут с owner=None).
    """
    if owner is None and _indexed_docs_cache_fresh():
        return [dict(d) for d in _indexed_docs_cache["data"]]

    coll = get_collection()
    try:
        where = {"type": {"$eq": "file_chunk"}}
        if owner:
            where = {"$and": [where, {"owner": {"$eq": owner}}]}
        results = coll.get(where=where, include=["metadatas"])
        docs: dict = {}
        for i, rid in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i]
            doc_id = meta.get("document_id") or "(no path)"
            title = meta.get("title") or doc_id.split("/")[-1]
            d = docs.setdefault(doc_id, {
                "memory_id": f"document::{doc_id}",
                "type": "document",
                "document_id": doc_id,
                "title": title,
                "owner": meta.get("owner", ""),
                "timestamp": meta.get("timestamp", ""),
                "tags_list": set(),
                "chunks": 0,
                "has_preview": False,
                "is_folder": False,
            })
            d["chunks"] += 1
            if meta.get("is_folder"):
                d["is_folder"] = True
            # Превью не тащим в листинг (base64 по картинке = сотни МБ при тысячах
            # фото). Оставляем только флаг для ленивой подгрузки thumb-эндпоинтом.
            if meta.get("image_preview"):
                d["has_preview"] = True
            d["tags_list"].update(k[2:] for k in meta if k.startswith("t_"))
            ts = meta.get("timestamp", "")
            if ts and ts > d["timestamp"]:
                d["timestamp"] = ts
        for d in docs.values():
            d["tags"] = sorted(d.pop("tags_list"))
        # Обогащаем группы чанков аннотацией memory_card этого документа (если есть).
        # Фильтр document_id делаем в Python — $ne не гарантирован на старых версиях ChromaDB.
        ann_where = {"type": {"$eq": "memory_card"}}
        if owner:
            ann_where = {"$and": [{"type": {"$eq": "memory_card"}}, {"owner": {"$eq": owner}}]}
        try:
            ann_res = coll.get(where=ann_where, include=["documents", "metadatas"], limit=10000)
            enriched = 0
            for i, rid in enumerate(ann_res.get("ids", [])):
                am = ann_res["metadatas"][i]
                adoc = am.get("document_id")
                if adoc and adoc in docs:
                    docs[adoc]["annotation"] = ann_res["documents"][i]
                    docs[adoc]["text"] = ann_res["documents"][i]
                    docs[adoc]["relevance"] = am.get("relevance", "permanent")
                    enriched += 1
            if enriched:
                logger.info(f"[unified] enriched {enriched} documents with annotations")
        except Exception as e:
            logger.warning(f"[unified] get_indexed_documents annotation enrichment error: {e}")

        # Автоматически обогащаем из companion Readme.md на диске, если в ChromaDB ещё нет memory_card
        for doc_id, d in docs.items():
            if not d.get("text"):
                _, readme_content = resolve_companion_readme(doc_id)
                if readme_content:
                    d["text"] = readme_content
                    d["annotation"] = readme_content
                    try:
                        upsert_memory_card(
                            readme_content,
                            owner=d.get("owner"),
                            tags=d.get("tags"),
                            title=d.get("title"),
                            document_id=doc_id,
                            relevance="permanent",
                        )
                    except Exception as err:
                        logger.debug(f"[unified] auto-sync memory_card for {doc_id}: {err}")

        out = list(docs.values())
        if owner is None:
            _indexed_docs_cache["valid"] = True
            _indexed_docs_cache["ts"] = time.time()
            _indexed_docs_cache["data"] = out
            return [dict(d) for d in out]
        return out
    except Exception as e:
        logger.error(f"[unified] get_indexed_documents error: {e}")
        return []


def delete_document(document_id: str):
    """Удаляет все чанки документа (по document_id)."""
    if not document_id:
        return
    coll = get_collection()
    try:
        coll.delete(where={"document_id": {"$eq": document_id}})
        _invalidate_indexed_docs_cache()
    except Exception as e:
        logger.error(f"[unified] delete_document error: {e}")


# ─── Индексация файловых чанков ───────────────────────────────────────────────

def has_document(document_id: str, source_stamp: str = None) -> bool:
    """Есть ли в индексе чанки документа; source_stamp — для проверки неизменённости."""
    try:
        where = {"document_id": {"$eq": document_id}}
        if source_stamp:
            where = {"$and": [where, {"source_stamp": {"$eq": source_stamp}}]}
        res = get_collection().get(where=where, include=["metadatas"], limit=1)
        return bool(res.get("ids"))
    except Exception as e:
        logger.error(f"[unified] has_document error: {e}")
        return False


def chunk_and_index_document(
    text: str,
    document_id: str,
    owner=None,
    tags=None,
    title="",
    chunk_size=500,
    overlap=50,
    source_stamp=None,
    image_preview="",
    is_folder=False,
) -> int:
    """
    Разбивает текст на чанки и индексирует в unified коллекцию.
    Предварительно удаляет старые чанки этого документа.
    Возвращает количество созданных чанков.
    """
    if not owner:
        owner = user_context.node_owner()

    try:
        get_collection().delete(where={"$and": [
            {"type":        {"$eq": "file_chunk"}},
            {"document_id": {"$eq": document_id}},
        ]})
    except Exception as e:
        logger.warning(f"[unified] Could not delete old chunks for {document_id}: {e}")

    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap

    if not chunks:
        return 0

    # Поисковый буст: имя файла нередко содержит ключевые слова (клиент, тип),
    # которых нет в тексте документа (напр. «račun_Vieste» при англ. запросе).
    if title:
        chunks[0] = f"[File: {title}]\n{chunks[0]}"

    tags_dict = _tags_meta(tags or [])
    ts = datetime.now().isoformat(timespec="seconds")
    title = title or document_id.split("/")[-1]

    ids        = [f"{document_id}:chunk:{idx}" for idx in range(len(chunks))]
    embeddings = get_model().encode(chunks, show_progress_bar=False).tolist()
    metadatas  = [
        {
            "type":        "file_chunk",
            "owner":       owner,
            "document_id": document_id,
            "chunk_id":    str(idx),
            "title":       title,
            "relevance":   "document_chunk",
            "timestamp":   ts,
            **({"source_stamp": source_stamp} if source_stamp else {}),
            **({"image_preview": image_preview} if image_preview and idx == 0 else {}),
            **({"is_folder": True, "contentType": "folder"} if is_folder else {}),
            **tags_dict,
        }
        for idx in range(len(chunks))
    ]

    try:
        get_collection().upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
    except Exception as e:
        if "DuplicateIDError" not in type(e).__name__ and "Expected IDs to be unique" not in str(e):
            logger.error(f"[unified] chunk upsert error: {e}")

    _invalidate_indexed_docs_cache()
    return len(chunks)


def delete_document(document_id: str):
    """Удаляет все записи документа из индекса."""
    try:
        get_collection().delete(where={"document_id": {"$eq": document_id}})
        _invalidate_indexed_docs_cache()
    except Exception as e:
        logger.error(f"[unified] delete_document error {document_id}: {e}")


def delete_path_prefix(path_prefix: str):
    """
    Удаляет все записи, у которых document_id начинается с path_prefix.
    Используется при удалении шары или каталога.
    """
    coll = get_collection()
    try:
        results = coll.get(include=["metadatas"])
        ids_to_delete = [
            results["ids"][i]
            for i, meta in enumerate(results.get("metadatas", []))
            if meta.get("document_id", "").startswith(path_prefix)
        ]
        if ids_to_delete:
            coll.delete(ids=ids_to_delete)
            logger.info(f"[unified] Deleted {len(ids_to_delete)} records for prefix {path_prefix}")
    except Exception as e:
        logger.error(f"[unified] delete_path_prefix error: {e}")


def move_file(old_path: str, new_path: str):
    """Обновляет document_id при переименовании/перемещении."""
    coll = get_collection()
    try:
        results = coll.get(
            where={"document_id": {"$eq": old_path}},
            include=["documents", "metadatas", "embeddings"],
        )
        if not results["ids"]:
            return
        coll.delete(ids=results["ids"])
        new_ids = [doc_id.replace(old_path, new_path, 1) for doc_id in results["ids"]]
        new_metas = [{**meta, "document_id": new_path} for meta in results["metadatas"]]
        coll.upsert(
            ids=new_ids,
            embeddings=results["embeddings"],
            documents=results["documents"],
            metadatas=new_metas,
        )
    except Exception as e:
        logger.error(f"[unified] move_file error {old_path} -> {new_path}: {e}")


# ─── QA-индекс ───────────────────────────────────────────────────────────────

def load_qa_entries(entries: list) -> int:
    """Загружает Q&A пары (заменяет старый qa_index)."""
    if not entries:
        return 0
    coll = get_collection()
    try:
        existing = coll.get(where={"type": {"$eq": "qa"}}, include=[])
        if existing["ids"]:
            coll.delete(ids=existing["ids"])
    except Exception:
        pass

    questions  = [e["question"] for e in entries]
    answers    = [e["answer"]   for e in entries]
    ids        = [f"qa:{i}" for i in range(len(questions))]
    embeddings = get_model().encode(questions, show_progress_bar=False).tolist()
    metadatas  = [
        {"type": "qa", "answer": a, "owner": "system", "t_qa": 1, "t_omd": 1, "tags": "qa,omd"}
        for a in answers
    ]
    coll.upsert(ids=ids, embeddings=embeddings, documents=questions, metadatas=metadatas)
    return len(questions)


def match_qa(question: str, min_score: float = 0.8, top_k: int = 1) -> dict:
    """Поиск ответа в QA-индексе."""
    results = search(
        question,
        tags_filter=["qa"],
        top_k=top_k,
        threshold=1.0 - min_score,
        record_types=["qa"],
    )
    if not results:
        return {"answer": None, "score": 0.0}
    best = results[0]
    return {"answer": best.get("answer", ""), "score": round(best["relevance"] / 100.0, 4)}


# ─── Высокоуровневые RAG-запросы ──────────────────────────────────────────────

def search_for_rag(
    query: str,
    private_mode: bool = False,
    top_k=None,
    tags_filter=None,
    document_id=None,
) -> list:
    """
    Поиск для RAG-инъекции:
    - tags_filter (хештеги из промпта) -> только эти теги
    - private_mode=True  + нет тегов   -> вся база (владелец ноды, все владельцы)
    - private_mode=False + нет тегов   -> только PUBLIC_TAGS (гость)
    - document_id (focus /learn)       -> только чанки/карточка этого файла (теги не фильтруем)
    - recency: результат ранжируется relevance × recency (свежие файлы выше)
    owner-фильтра больше нет: в private-режиме видны записи всех владельцев ноды.
    """
    if top_k is None:
        top_k = RAG_TOP_K
    if tags_filter:
        tag_filter = tags_filter
    elif document_id:
        # При document_id фильтр по тегам не нужен — document_id и так
        # ограничивает выдачу одним документом; PUBLIC_TAGS убьёт чанки
        #OWNER-документов (они без тега "public").
        tag_filter = None
    else:
        tag_filter = None if private_mode else PUBLIC_TAGS
    return search(
        query,
        tags_filter=tag_filter,
        top_k=top_k,
        threshold=RAG_THRESHOLD,
        record_types=["file_chunk", "memory_card"],
        document_id=document_id,
        recent_first=True,
    )


def search_files_for_ui(
    query: str,
    private_mode: bool = True,
    top_k=None,
) -> list:
    """
    Поиск файлов для файлового менеджера.
    """
    if top_k is None:
        top_k = SEARCH_TOP_K
    tags_filter = None if private_mode else PUBLIC_TAGS
    return search(
        query,
        tags_filter=tags_filter,
        top_k=top_k,
        threshold=SEARCH_THRESHOLD,
        record_types=["file_chunk"],
    )


def chunk_document(text: str, chunk_size=500, overlap=50) -> list[str]:
    """Разбивает текст на чанки (без индексации). Используется stateless /learn."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ─── Локальная конвертация raw-документов (гостевой/персональный слой) ────────

PANDOC_FORMATS = {
    "docx", "odt", "epub", "fb2", "html", "htm", "csv", "md", "markdown",
    "rst", "rtf", "org", "mediawiki", "tex", "typst",
}

def _split_filename(name: str) -> tuple[str, str]:
    base = (name or "").split("?")[0]
    ext = os.path.splitext(base)[1].lower().lstrip(".")
    return base, ext


def convert_bytes_to_text(data: bytes, filename: str = "") -> str:
    """
    Конвертирует сырые байты документа в текст ЛОКАЛЬНО (без шлюза).
    PDF → pdftotext; docx/odt/epub/fb2/html/md/... → pandoc; текст как есть.
    Временный файл удаляется после конвертации.
    """
    base, ext = _split_filename(filename)
    if not data:
        return ""

    # Текстовые файлы — просто декодируем
    if ext in ("txt", "text") or base.lower().endswith((".txt", ".text")):
        try:
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    tmp_path = None
    try:
        suffix = "." + ext if ext else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        # PDF → pdftotext (poppler-utils)
        if ext == "pdf":
            if not shutil.which("pdftotext"):
                logger.error("[unified] pdftotext not available")
                return ""
            try:
                res = subprocess.run(
                    ["pdftotext", "-layout", "-enc", "UTF-8", tmp_path, "-"],
                    capture_output=True, timeout=120,
                )
                text = (res.stdout or b"").decode("utf-8", errors="replace")
                return text.strip()
            except Exception as e:
                logger.error(f"[unified] pdftotext failed: {e}")
                return ""

        # Остальное → pandoc
        if ext in PANDOC_FORMATS:
            if not shutil.which("pandoc"):
                logger.error("[unified] pandoc not available")
                return ""
            try:
                res = subprocess.run(
                    ["pandoc", tmp_path, "-t", "markdown"],
                    capture_output=True, timeout=120,
                )
                if res.returncode != 0:
                    logger.error(f"[unified] pandoc failed ({res.returncode}): {res.stderr[:300]}")
                    return ""
                text = (res.stdout or b"").decode("utf-8", errors="replace")
                return text.strip()
            except Exception as e:
                logger.error(f"[unified] pandoc error: {e}")
                return ""

        # Неизвестный тип — пробуем прочитать как текст
        try:
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def cleanup_guest_data(owners_to_keep: list[str]) -> int:
    """
    Удаляет из ChromaDB все записи, чей owner НЕ в owners_to_keep,
    за исключением записей с t_omd=1 (публичные знания владельца).
    Возвращает количество удалённых записей.
    """
    coll = get_collection()
    if coll.count() == 0:
        return 0
    try:
        results = coll.get(include=["metadatas"])
    except Exception as e:
        logger.error(f"[unified] cleanup_guest_data get error: {e}")
        return 0

    ids_to_delete = []
    for i, meta in enumerate(results.get("metadatas", [])):
        owner = meta.get("owner", "")
        has_omd = meta.get("t_omd", 0) == 1
        if owner not in owners_to_keep and not has_omd:
            ids_to_delete.append(results["ids"][i])

    if ids_to_delete:
        coll.delete(ids=ids_to_delete)
        logger.info(f"[unified] Cleanup: deleted {len(ids_to_delete)} guest records (kept owners={owners_to_keep})")
    return len(ids_to_delete)


# ─── Инициализация ────────────────────────────────────────────────────────────

def init():
    """
    Инициализация. Вызывается при старте api.py.
    Создаёт базу, если её нет.
    """
    try:
        coll = get_collection()
        count = coll.count()
        logger.info(f"[unified] omd_unified initialized, {count} records in {UNIFIED_DB_DIR}")
    except Exception as e:
        logger.error(f"[unified] init error: {e}")
        raise
