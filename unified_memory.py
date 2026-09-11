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

# ─── Embedding model ──────────────────────────────────────────────────────────

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
        logger.info(f"[unified] Loading SentenceTransformer on {device}...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
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


# ─── Поиск ───────────────────────────────────────────────────────────────────

def search(
    query: str,
    tags_filter=None,
    owner=None,
    top_k=None,
    threshold=None,
    record_types=None,
    document_id=None,
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
            row = {"id": ids[i], "text": docs[i], "distance": dist,
                   "relevance": round((1.0 - dist) * 100, 1)}
            row.update(metas[i])
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
                        row = {"id": rid, "text": doc or "", "distance": dist,
                               "relevance": round((1.0 - dist) * 100, 1)}
                        row.update(meta)
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
    return deduped[:top_k]


# ─── Карточки памяти ─────────────────────────────────────────────────────────

def upsert_memory_card(
    text: str,
    mem_id=None,
    owner=None,
    tags=None,
    title="",
    document_id=None,
    relevance="contextual",
) -> str:
    """Добавляет/обновляет карточку памяти. Возвращает mem_id."""
    if not mem_id:
        mem_id = str(uuid.uuid4())
    if not owner:
        owner = user_context.node_owner()

    metadata = {
        "type":      "memory_card",
        "owner":     owner,
        "title":     title,
        "relevance": relevance,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metadata.update(_tags_meta(tags or []))
    if document_id:
        metadata["document_id"] = document_id

    get_collection().upsert(
        ids=[mem_id],
        embeddings=[embed(text)],
        documents=[text.strip()],
        metadatas=[metadata],
    )
    return mem_id


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
    except Exception as e:
        logger.error(f"[unified] delete_memory_card error: {e}")


def get_all_memory_cards(owner=None) -> list:
    """Возвращает все карточки памяти (для листинга в UI)."""
    coll = get_collection()
    try:
        where = {"type": {"$eq": "memory_card"}}
        if owner:
            where = {"$and": [where, {"owner": {"$eq": owner}}]}
        results = coll.get(where=where, include=["documents", "metadatas"])
        out = []
        for i, doc_id in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i]
            out.append({
                "memory_id": doc_id,
                "_id": doc_id,
                "text": results["documents"][i],
                **meta,
                "tags": [k[2:] for k in meta if k.startswith("t_")],
            })
        return out
    except Exception as e:
        logger.error(f"[unified] get_all_memory_cards error: {e}")
        return []


def get_indexed_documents(owner=None) -> list:
    """Возвращает проиндексированные документы (file_chunk), сгруппированные по document_id."""
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
            })
            d["chunks"] += 1
            d["tags_list"].update(k[2:] for k in meta if k.startswith("t_"))
            ts = meta.get("timestamp", "")
            if ts and ts > d["timestamp"]:
                d["timestamp"] = ts
        for d in docs.values():
            d["tags"] = sorted(d.pop("tags_list"))
        return list(docs.values())
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

    return len(chunks)


def delete_document(document_id: str):
    """Удаляет все записи документа из индекса."""
    try:
        get_collection().delete(where={"document_id": {"$eq": document_id}})
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
    - document_id (focus /learn)       -> только чанки/карточка этого файла
    owner-фильтра больше нет: в private-режиме видны записи всех владельцев ноды.
    """
    if top_k is None:
        top_k = RAG_TOP_K
    if tags_filter:
        tag_filter = tags_filter
    else:
        tag_filter = None if private_mode else PUBLIC_TAGS
    return search(
        query,
        tags_filter=tag_filter,
        top_k=top_k,
        threshold=RAG_THRESHOLD,
        record_types=["file_chunk", "memory_card"],
        document_id=document_id,
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
