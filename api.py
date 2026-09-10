from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response, RedirectResponse, JSONResponse
from fastapi import Header, Depends
from PIL import Image
import io
import re
import mimetypes
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import logging

# Suppress logging of DuplicateIDError and telemetry warnings from chromadb
class DuplicateIDFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        if "DuplicateIDError" in message or "Expected IDs to be unique" in message:
            return False
        if "chromadb.telemetry" in record.name or "Failed to send telemetry event" in message:
            return False
        if record.exc_info:
            exc_type, exc_value, _ = record.exc_info
            if exc_type and ("DuplicateIDError" in exc_type.__name__ or "DuplicateIDError" in str(exc_type)):
                return False
            if exc_value and "Expected IDs to be unique" in str(exc_value):
                return False
        return True

logging.basicConfig(level=logging.INFO)
# Add filter to root logger and standard loggers
for logger_name in [None, "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "chromadb", "chromadb.telemetry"]:
    l = logging.getLogger(logger_name)
    l.addFilter(DuplicateIDFilter())
    for handler in l.handlers:
        handler.addFilter(DuplicateIDFilter())

import hashlib
import email.utils
import datetime
import time
import base64
import requests
import aiohttp
import asyncio
import json
import subprocess
import numpy as np
from urllib.parse import urlparse
from typing import List, Optional

import core_service
import user_context

# Create a global session for proxying to avoid socket exhaustion
_proxy_session = None

# The question tool must only ever ask ONE question at a time: when an agent
# streams a "question" tool call its arguments are rewritten so that only the
# first question survives (SSH tunnel agents still rely on this contract).
SINGLE_QUESTION_TOOL_NAMES = ("question", "question.v2")



async def get_proxy_session():
    global _proxy_session
    if _proxy_session is None or _proxy_session.closed:
        _proxy_session = aiohttp.ClientSession()
    return _proxy_session

import memory_index
import unified_memory
from config import USER_DATA_DIR
from config import BASE_INDEX_DIR
from config import SETTINGS

GATEWAY_URL = SETTINGS["GATEWAY_URL"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:8080", "http://localhost", "https://onmydisk.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


AI_TOKEN = SETTINGS.get("AI_TOKEN", "")

# Search Node (legacy, kept for /indexer/from_crawl backward compat)
try:
    from search_node import SearchNode
    search_node = SearchNode(storage_path=BASE_INDEX_DIR, model=memory_index.get_model(), token=AI_TOKEN)
    logging.info("[api] SearchNode initialized (legacy compat)")
except Exception as e:
    logging.error(f"[api] SearchNode init failed: {e}")
    search_node = None

@app.on_event("startup")
async def on_startup():
    """Initialize unified_memory on startup."""
    try:
        unified_memory.init()
    except Exception as e:
        logging.error(f"[api] unified_memory init error: {e}")

# PeARS-compatible endpoints

@app.get("/indexer/from_crawl")
async def indexer_from_crawl(request: Request, background_tasks: BackgroundTasks):
    path = request.query_params.get("path") or request.headers.get("path")
    url = request.query_params.get("url") or request.headers.get("url")
    collection = request.query_params.get("collection") or request.headers.get("collection")
    is_async = (request.query_params.get("async") or request.headers.get("async") or "false").lower() == "true"
    
    if not url and path:
        gateway = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net").rstrip('/')
        url = f"{gateway}/{path.lstrip('/')}"
        
    if not url:
         raise HTTPException(status_code=422, detail="Either 'url' or 'path' is required in query or headers")

    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    if is_async:
        background_tasks.add_task(search_node.index_url, url, collection)
        return {"status": "indexing_started", "url": url}
    else:
        result = search_node.index_url(url, collection=collection)
        return result

@app.get("/api/urls/delete")
async def delete_url(request: Request):
    path = request.query_params.get("path") or request.headers.get("path")
    if not path:
        raise HTTPException(status_code=422, detail="path is required")
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    unified_memory.delete_document(path)
    return {"status": "ok", "path": path}

@app.get("/api/urls/move")
async def move_url(request: Request):
    src    = request.query_params.get("src")    or request.headers.get("src")
    target = request.query_params.get("target") or request.headers.get("target")
    if not src or not target:
        raise HTTPException(status_code=422, detail="src and target are required")
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    unified_memory.move_file(src, target)
    return {"status": "ok", "src": src, "target": target}


# ─── RAG 3.0: новые эндпоинты ────────────────────────────────────────────────

@app.get("/api/convert")
async def convert_document(request: Request):
    """
    Конвертирует файл/URL в markdown-текст.
    Вызывается через P2P data channel (WebRTCDataChannelDrive.sendRequest).
    Заменяет шлюзовый ?totext для клиентского /learn.
    """
    path = request.query_params.get("path") or request.headers.get("path")
    if not path:
        raise HTTPException(status_code=422, detail="path is required")
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    omd_key = _extract_omd_key(request)
    try:
        text = await core_service.fetch_document_text(
            path if path.startswith("http") else f"{GATEWAY_URL}{path}",
            token=omd_key
        )
        return {"text": text, "path": path}
    except Exception as e:
        logging.error(f"[api/convert] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learn")
async def learn_document(request: Request):
    """
    Stateless /learn: аннотация + теги + эмбеддинги.
    Ничего не пишет в ChromaDB — фронт сохраняет в GunDB.
    Возвращает.annotation, tags, annotation_embedding, chunks с эмбеддингами.
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body     = await request.json()
    path     = body.get("path", "")
    text     = body.get("text", "")
    tags     = body.get("tags", [])     # уже распарсенные теги (без #)
    question = body.get("question", "") # необязательный вопрос для ответа

    if not text or not path:
        raise HTTPException(status_code=422, detail="path and text are required")

    ctx = await _build_ctx_from_request(request)

    doc_id = _norm_doc_path(path) if path else path
    title  = path.split("/")[-1].split("?")[0]

    # Аннотация + теги (stateless, LLM)
    try:
        annotation = await core_service.summarize_for_memory(ctx, text)
        ai_tags    = await core_service.extract_tags_from_text(ctx, text)
    except Exception as e:
        logging.error(f"[api/learn] summarize error: {e}")
        annotation = text[:500]
        ai_tags    = tags

    all_tags = sorted(set(tags) | set(ai_tags or []))

    # Аннотационный эмбеддинг (теги активно участвуют в embed)
    annotation_text = annotation + " " + " ".join(all_tags)
    annotation_embedding = unified_memory.embed(annotation_text)

    # Чанки + эмбеддинги чанков
    chunks_texts = unified_memory.chunk_document(text)
    chunks = []
    if chunks_texts:
        chunk_embeddings = unified_memory.get_model().encode(chunks_texts, show_progress_bar=False).tolist()
        chunks = [
            {"text": ct, "embedding": ce}
            for ct, ce in zip(chunks_texts, chunk_embeddings)
        ]

    logging.info(f"[api/learn] Stateless: doc={doc_id} chunks={len(chunks)} tags={all_tags}")

    return {
        "annotation":          annotation,
        "tags":                all_tags,
        "annotation_embedding": annotation_embedding,
        "chunks":              chunks,
        "documentId":          doc_id,
        "title":               title,
        "chunkCount":          len(chunks),
        "question":            question,
    }

# ─── RAG 3.0 (docs/RAG_3_0.md §2): пути нормализуем срезанием префикса юзера ─

def _norm_doc_path(path: str) -> str:
    """Локальный путь-/home/<user>/<share>/... или itemPath /<user>/<share>/... → /<share>/...; http — как есть."""
    if not path:
        return path
    if path.startswith("http"):
        return path
    s = "/" + path.strip("/")
    if s.startswith("/home/"):
        # realPath ноды: /home/<user>/<share>/...
        parts = s.split("/", 3)
        return f"/{parts[3]}" if len(parts) == 4 else s.rstrip("/")
    # виртуальный itemPath: /<user>/<share>/...
    parts = s.split("/", 2)
    return f"/{parts[2]}" if len(parts) == 3 else s.rstrip("/")


RAG_INDEX_STATUS = {}   # path → {"indexed": n, "pending": n, "failed": n, "lastError": ..., "done": bool}

def _rag_scope_tags(body_tags, scope: str) -> list:
    tags = [t for t in (body_tags or []) if t]
    if scope == "public" and "public" not in tags:
        tags.append("public")
    return tags


HUB_URL = SETTINGS.get("HUB_URL", "https://direct.onmydisk.net:8765")

async def fetch_chunks_from_hub(token_hash: str, memory_id: str) -> list:
    """Fetch document chunks from GunDB hub for chunk-level RAG."""
    try:
        import aiohttp
        url = f"{HUB_URL}/api/chunks?tokenHash={token_hash}&memoryId={memory_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("chunks", []) if data.get("ok") else []
    except Exception as e:
        logging.warning(f"[hub] fetch_chunks failed for {memory_id}: {e}")
        return []


def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0


async def _rag_stateless_payload(ctx, raw_text: str, doc_id: str, title: str, tags: list, owner: str):
    """Общая stateless-логика: аннотация + теги + эмбеддинги + чанки. Без ChromaDB."""
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=422, detail="document text is empty after conversion")

    try:
        annotation = await core_service.summarize_for_memory(ctx, raw_text)
        ai_tags = await core_service.extract_tags_from_text(ctx, raw_text)
    except Exception as e:
        logging.error(f"[rag/import] summarize error: {e}")
        annotation = raw_text[:500]
        ai_tags = []

    all_tags = sorted(set(tags) | set(ai_tags or []))

    # Аннотационный эмбеддинг (теги активно участвуют)
    annotation_text = annotation + " " + " ".join(all_tags)
    annotation_embedding = unified_memory.embed(annotation_text)

    # Чанки + эмбеддинги чанков
    chunks_texts = unified_memory.chunk_document(raw_text)
    chunks = []
    if chunks_texts:
        chunk_embeddings = unified_memory.get_model().encode(chunks_texts, show_progress_bar=False).tolist()
        chunks = [
            {"text": ct, "embedding": ce}
            for ct, ce in zip(chunks_texts, chunk_embeddings)
        ]

    logging.info(f"[rag/import] Stateless: doc={doc_id} chunks={len(chunks)} tags={all_tags} owner={owner}")

    return {
        "docId":               doc_id,
        "documentId":          doc_id,
        "title":               title,
        "annotation":          annotation,
        "tags":                all_tags,
        "annotation_embedding": annotation_embedding,
        "chunks":              chunks,
        "chunkCount":          len(chunks),
    }


def _decode_hdr(request: Request, name: str) -> str | None:
    """URL-encoded имена файлов в заголовках (X-OMD-Filename / X-OMD-Source)."""
    raw = request.headers.get(name)
    if not raw:
        return None
    if "%" in raw:
        try:
            from urllib.parse import unquote
            return unquote(raw, errors="replace")
        except Exception:
            return raw
    return raw


@app.post("/rag/import/raw")
async def rag_import_raw_endpoint(request: Request):
    """
    Stateless импорт сырого документа (гостевой/персональный слой):
    клиент аплоадит байты файла → нода конвертирует ЛОКАЛЬНО (pdftotext/pandoc,
    без шлюза) → аннотация/теги/чанки/эмбеддинги → возвращает payload, а временный
    файл удаляется. В ChromaDB ничего не пишется — фронт хранит карточку в GunDB.
    Заголовки: X-OMD-Filename (URL-encoded), X-OMD-Source, X-OMD-RAG-Scope,
    X-OMD-Tags (JSON), X-OMD-Owner.
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        data = await request.body()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read request body: {e}")

    filename = _decode_hdr(request, "X-OMD-Filename") or _decode_hdr(request, "filename") or "document.bin"
    source = _decode_hdr(request, "X-OMD-Source") or filename
    scope = request.headers.get("X-OMD-RAG-Scope", "private")
    owner = request.headers.get("X-OMD-Owner") or ""
    try:
        tags_in = json.loads(request.headers.get("X-OMD-Tags", "[]")) or []
    except Exception:
        tags_in = []

    if not data:
        raise HTTPException(status_code=422, detail="empty file body")

    raw_text = unified_memory.convert_bytes_to_text(data, filename)
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=422, detail="could not extract text from document (unsupported or scanned file)")

    ctx = await _build_ctx_from_request(request)
    owner = owner or ctx.user_id or "alexey"
    doc_id = _norm_doc_path(source) if source else f"upload:{filename}"
    title = filename.split("/")[-1].split("?")[0].lstrip("/") or doc_id
    tags = _rag_scope_tags(tags_in, scope)

    return await _rag_stateless_payload(ctx, raw_text, doc_id, title, tags, owner)


@app.post("/rag/import")
async def rag_import_endpoint(request: Request):
    """
    Stateless RAG 3.0 импорт: аннотация + теги + эмбеддинги + чанки.
    Ничего не пишет в ChromaDB — фронт сохраняет в GunDB.
    Текст документа присылает клиент (source + text) — нода не лезет в шлюз.
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body  = await request.json()
    source = (body.get("source") or "").strip()
    raw_text = (body.get("text") or "").strip()
    scope = body.get("scope") or request.headers.get("X-OMD-RAG-Scope", "private")
    owner = body.get("owner") or ""
    tags_in = body.get("tags") or []

    if not raw_text:
        raise HTTPException(status_code=422, detail="text is required (client fetches the document and sends its text)")

    doc_id = _norm_doc_path(source) if source else f"user:note:{body.get('title') or 'note'}"

    ctx = await _build_ctx_from_request(request)
    owner = owner or ctx.user_id or "alexey"
    title = body.get("title") or doc_id.split("/")[-1].split("?")[0].lstrip("/")
    tags = _rag_scope_tags(tags_in, scope)

    return await _rag_stateless_payload(ctx, raw_text, doc_id, title, tags, owner)


@app.post("/embed")
async def embed_endpoint(request: Request):
    """
    Stateless embedding: {text: ""} → {embedding: []}.
    Для фронта: эмбеддит вопрос при отправке (cosine vs annotation_embedding карточек).
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is required")
    try:
        embedding = unified_memory.embed(text)
    except Exception as e:
        logging.error(f"[embed] error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    return {"embedding": embedding}


@app.post("/rag/index")
async def rag_index_endpoint(request: Request, background_tasks: BackgroundTasks):
    """
    RAG 3.0 контракт: crawl шары/пути (вызывает C++ нода: локальный realPath).
    path-/home/<user>/<share>/... нормализуется до /<share>/... и обходится через gateway XML-index.
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    source_path = (body.get("path") or "").strip()
    if not source_path:
        raise HTTPException(status_code=422, detail="path is required")
    force = bool(body.get("force", False))
    scope = request.headers.get("X-OMD-RAG-Scope", body.get("scope", "private"))

    doc_root = _norm_doc_path(source_path)          # /<share>/...
    crawl_url = source_path if source_path.startswith("http") else f"{GATEWAY_URL}{doc_root}"
    ctx = await _build_ctx_from_request(request)
    owner = ctx.user_id or "alexey"
    tags = _rag_scope_tags(body.get("tags") or [], scope)

    RAG_INDEX_STATUS[doc_root] = {"indexed": 0, "pending": 0, "failed": 0, "lastError": None, "done": False}

    async def _crawl():
        status = RAG_INDEX_STATUS[doc_root]
        from lxml import etree
        queue = [crawl_url if crawl_url.endswith("/") else crawl_url + "/"]
        visited = set()
        blacklist = ["node_modules", ".git", ".venv", "venv", "__pycache__", "site-packages",
                     "bin", "obj", "target", "dist", "build", ".cache", ".idea", ".vscode",
                     ".pytest_cache", ".npm", ".yarn", "node_modules/"]
        exact_blacklist = {"proc", "sys", "system", "data", "dev", "run", "etc", "boot",
                           "lib", "lib64", "opt", "srv", ".local", ".config", ".cache"}
        token = _extract_omd_key(request) or AI_TOKEN

        async def fetch(url, params=""):
            try:
                full = url + params
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as s:
                    async with s.get(full, headers={"Authorization": f"token:{token}"}) as r:
                        if r.status == 200:
                            return await r.text()
                        return None
            except Exception as e:
                status["failed"] += 1
                status["lastError"] = str(e)
                return None

        while queue and not RAG_INDEX_STATUS.get(doc_root, {}).get("abort", False):
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            xml = await fetch(current, "?index")
            if not xml or "<omd_index>" not in xml:
                status["lastError"] = f"no index xml: {current}"
                continue

            try:
                root_doc = etree.fromstring(xml.encode("utf-8"), etree.XMLParser(recover=True))
            except Exception as e:
                status["failed"] += 1
                status["lastError"] = f"xml parse: {e}"
                continue

            for doc in root_doc.findall(".//doc"):
                name = doc.get("url") or ""
                if not name or name.startswith("?"):
                    continue
                item_url = current.rstrip("/") + "/" + name
                item_path = urlparse(item_url).path
                doc_id = _norm_doc_path(item_path)
                if doc.get("contentType") == "folder":
                    low = name.lower().strip("/")
                    if low in exact_blacklist or any(b in low for b in blacklist):
                        continue
                    queue.append(item_url)
                    continue

                last_modified = doc.get("last_modified") or ""
                if not force and unified_memory.has_document(doc_id, source_stamp=last_modified):
                    logging.info(f"[rag/index] skip unchanged {doc_id}")
                    continue

                raw = await core_service.fetch_document_text(item_url, token=token)
                if not raw or raw.startswith("Failed to fetch"):
                    status["failed"] += 1
                    status["lastError"] = raw or "empty"
                    continue
                try:
                    n = unified_memory.chunk_and_index_document(
                        raw,
                        document_id=doc_id,
                        owner=owner,
                        tags=tags,
                        title=doc_id.split("/")[-1],
                        source_stamp=last_modified,
                    )
                    status["indexed"] += 1
                    logging.info(f"[rag/index] {doc_id}: {n} chunks")
                except Exception as e:
                    status["failed"] += 1
                    status["lastError"] = str(e)

        status["done"] = True

    background_tasks.add_task(_crawl)
    return {"status": "indexing_started", "path": doc_root, "crawl_url": crawl_url}


@app.get("/rag/status")
async def rag_status_endpoint(request: Request, path: str = "", docId: str = ""):
    """Прогресс индексации пути (RAG 3.0 §2)."""
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    key = _norm_doc_path(path or docId)
    entry = RAG_INDEX_STATUS.get(key, {})
    return {"path": key, **entry}


@app.post("/rag/delete")
async def rag_delete_endpoint(request: Request):
    """Удаление документа/потоков из индекса (source — itemPath/URL/docId)."""
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    source = (body.get("source") or body.get("docId") or "").strip()
    if not source:
        raise HTTPException(status_code=422, detail="source or docId is required")
    doc_id = _norm_doc_path(source)
    unified_memory.delete_document(doc_id)
    logging.info(f"[rag/delete] {doc_id}")
    return {"deleted": 1, "path": doc_id}


@app.get("/api/learn_preview")
async def learn_preview(request: Request):
    """
    Preview: конвертирует документ и генерирует аннотацию + облако тегов.
    НЕ сохраняет в ChromaDB — только для MemoryCardModal (preview перед сохранением).
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    path    = request.query_params.get("path", "")
    omd_key = _extract_omd_key(request)

    if not path:
        raise HTTPException(status_code=422, detail="path is required")

    ctx = await _build_ctx_from_request(request)
    try:
        text = await core_service.fetch_document_text(
            path if path.startswith("http") else f"{GATEWAY_URL}{path}",
            token=omd_key
        )
        annotation = await core_service.summarize_for_memory(ctx, text)
        ai_tags    = await core_service.extract_tags_from_text(ctx, text)
        return {"annotation": annotation, "tags": ai_tags, "path": path}
    except Exception as e:
        logging.error(f"[api/learn_preview] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indexer/index_share")
async def index_share(request: Request, background_tasks: BackgroundTasks):
    """
    Индексирует конкретную шару (вызывается C++ нодой напрямую через localhost).
    Заменяет /indexer/from_crawl для шар OMD 3.0.
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    share   = request.query_params.get("share", "")
    path    = request.query_params.get("path",  "")
    tag     = request.query_params.get("tag",   share)
    is_async = request.query_params.get("async", "true").lower() == "true"

    if not path:
        raise HTTPException(status_code=422, detail="path is required")

    # Строим URL для индексации
    gateway = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net").rstrip("/")
    url = path if path.startswith("http") else f"{gateway}/{path.lstrip('/')}"
    tags = [t for t in [tag, share] if t]

    def _do_index_share():
        try:
            if search_node:
                search_node.index_url(url, collection=tag or share)
            logging.info(f"[indexer/share] Indexed share={share} path={path} tags={tags}")
        except Exception as e:
            logging.error(f"[indexer/share] error: {e}")

    if is_async:
        background_tasks.add_task(_do_index_share)
        return {"status": "indexing_started", "share": share, "path": path}
    else:
        _do_index_share()
        return {"status": "ok", "share": share, "path": path}


@app.delete("/indexer/remove_path")
async def remove_path(request: Request):
    """Удаляет путь из unified индекса (вызывается C++ при удалении файла/шары)."""
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    path = request.query_params.get("path", "")
    if not path:
        raise HTTPException(status_code=422, detail="path is required")
    unified_memory.delete_path_prefix(path)
    return {"status": "ok", "path": path}


async def _build_ctx_from_request(request: Request):
    """Строит UserContext из заголовков запроса."""
    omd_key = _extract_omd_key(request)
    ctx = user_context.UserContext(
        type="omd",
        user_id  = request.headers.get("X-OMD-User", "") or SETTINGS.get("NODE_OWNER", ""),
        settings = {},
        history  = [],
        omd_key  = omd_key,
    )
    ctx.private_mode = is_private_mode(request, ctx)
    return ctx


def _extract_omd_key(request: Request) -> str | None:
    """Извлекает omd_key из request напрямую (не через FastAPI Depends)."""
    # Query params
    omd_key = request.query_params.get("omd_key")
    if omd_key:
        return omd_key
    token = request.query_params.get("token")
    if token:
        return token
    # Headers
    for header_name in ("X-OMD-Key", "X-OMD-Token", "Token"):
        val = request.headers.get(header_name)
        if val:
            return val
    # Authorization
    authorization = request.headers.get("Authorization")
    if authorization:
        if authorization.startswith("Bearer "):
            return authorization[7:]
        elif authorization.startswith("token:"):
            return authorization[6:]
        elif authorization.startswith("token "):
            return authorization[6:]
        return authorization
    return None

def not_authorized(request: Request):
    # The gateway forwards the original client's Authorization header
    # AND adds its own 'Token' header for the node's API token.
    # We must check if ANY of these tokens match our AI_TOKEN.
    possible_tokens = [
        request.headers.get("X-OMD-Ai-Token"),
        request.headers.get("Token"),
        request.headers.get("X-OMD-Token"),
        request.headers.get("Authorization"),
        request.query_params.get("token")
    ]
    
    for raw_token in possible_tokens:
        if not raw_token:
            continue
            
        token = raw_token
        if token.startswith("token:"):
            token = token[len("token:"):]
        elif token.startswith("Bearer "):
            token = token[7:]
            
        token = token.strip()
        
        if AI_TOKEN and token == AI_TOKEN:
            return False # Authorized!
            
        # Also allow any valid 32-character hexadecimal token (the standard OMD node token format)
        if len(token) == 32 and all(c in '0123456789abcdefABCDEF' for c in token):
            return False # Authorized!
            
    if AI_TOKEN:
        logging.warning(f"Unauthorized request, no valid token found.")
        return True # Not authorized
        
    return False # Authorized if no AI_TOKEN is configured

_ephemeral_image_cache: dict[str, dict] = {}

def store_ephemeral_image(filename: str, data: bytes, ttl: int = 300):
    cleanup_ephemeral_images()
    _ephemeral_image_cache[filename] = {
        "data": data,
        "expires_at": time.time() + ttl
    }

def get_ephemeral_image(filename: str) -> bytes | None:
    cleanup_ephemeral_images()
    item = _ephemeral_image_cache.get(filename)
    if item:
        return item["data"]
    return None

def cleanup_ephemeral_images():
    now = time.time()
    expired = [k for k, v in _ephemeral_image_cache.items() if v["expires_at"] < now]
    for k in expired:
        _ephemeral_image_cache.pop(k, None)

def is_private_mode(request: Request, ctx: user_context.UserContext | None = None) -> bool:
    # 1. Explicit header from gateway or P2P client (e.g. isOwner || isSharedTo)
    pm_header = request.headers.get("X-OMD-Private-Mode")
    if pm_header is not None:
        return pm_header == "1" or pm_header.lower() == "true"

    # 2. If token balance is provided and > 0, it's a paid external subscriber (Public Mode)
    token_balance = float(request.headers.get("x-omd-token-balance", "0.0") or "0.0")
    if token_balance > 0.0:
        return False

    # 3. Check if ctx matches local node owner
    import getpass
    node_owner = SETTINGS.get("NODE_OWNER") or getpass.getuser()
    if ctx and ctx.user_id and ctx.user_id == node_owner:
        return True

    # 4. Localhost connection (direct local embedded client on node)
    client_host = request.client.host if request.client else ""
    if client_host in ("127.0.0.1", "::1", "localhost"):
        # If client is authenticated as someone other than node owner, it's NOT private mode
        if ctx and ctx.user_id and ctx.user_id not in ("anon", "system", "") and ctx.user_id != node_owner:
            return False
        return True

    # 5. Token match without balance
    ai_token = request.headers.get("X-OMD-Ai-Token") or request.headers.get("Token") or ""
    return bool(AI_TOKEN and ai_token == AI_TOKEN)

def get_omd_key(
    request: Request,
    omd_key: str | None = Query(None),
    token: str | None = Query(None),
    x_omd_key: str | None = Header(None, alias="X-OMD-Key"),
    x_omd_token: str | None = Header(None, alias="X-OMD-Token"),
    authorization: str | None = Header(None),
    token_header: str | None = Header(None, alias="Token")
):
    if omd_key:
        logging.info(f"omd_key found in query: {omd_key[:10]}...")
        return omd_key
    if token:
        logging.info(f"omd_key (token) found in query: {token[:10]}...")
        return token
    if token_header:
        logging.info(f"omd_key found in Token header: {token_header[:10]}...")
        return token_header
    if x_omd_key:
        logging.info(f"omd_key found in X-OMD-Key header: {x_omd_key[:10]}...")
        return x_omd_key
    if x_omd_token:
        logging.info(f"omd_key found in X-OMD-Token header: {x_omd_token[:10]}...")
        return x_omd_token
    if authorization:
        # Handle "Bearer <token>", "token:<token>", "token <token>" or just "<token>"
        logging.info(f"omd_key found in Authorization header: {authorization[:20]}...")
        auth_val = authorization.strip()
        if auth_val.startswith("Bearer "):
            return auth_val[7:].strip()
        if auth_val.startswith("token:"):
            return auth_val[6:].strip()
        if auth_val.startswith("token "):
            return auth_val[6:].strip()
        return auth_val
    
    # Check cookies
    cookie_token = request.cookies.get("omd_key")
    if cookie_token:
        logging.info(f"omd_key found in cookie: {cookie_token[:10]}...")
        return cookie_token

    logging.warning("No omd_key found in request")
    return None

@app.get("/search")
async def search(
    request: Request,
    q: str | None = Query(None),
    limit: int | None = Query(None),
    lang: str | None = Query(None),
    omd_key: str | None = Depends(get_omd_key)
):
    if not q:
        q = request.headers.get("q")
    if not q:
        raise HTTPException(status_code=422, detail="q is required in query or headers")

    if limit is None:
        header_limit = request.headers.get("limit")
        default_limit = int(SETTINGS.get("SEARCH_TOP_K", "20"))
        limit = int(header_limit) if header_limit and header_limit.isdigit() else default_limit

    if not lang:
        lang = request.headers.get("lang") or "en"

    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    ctx = get_ctx(omd_key)
    results = search_node.search(q, limit, ctx=ctx)
    return results
# CORS middleware already added at line 34

@app.on_event("startup")
async def startup_event():
    logging.info("[api] Startup complete")

# ==== Модели ввода ====

class ChatInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"
    settings: dict | None = None
    history: list | None = None
    knowledge: list | None = None
    prompt_id: str | None = None

class ChatStreamInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"
    history: list | None = None
    settings: dict | None = None
    knowledge: list | None = None
    prompt_id: str | None = None
    chat_summary: str | None = None
    total_message_count: int | None = None
    image_delivery: str | None = None
    client: str | None = None

class ImportInput(BaseModel):
    omd_key: str
    url_or_path: str
    collection: str = "user"

class MemorizeInput(BaseModel):
    omd_key: str
    text: str

class RecognizeInput(BaseModel):
    omd_key: str
    prompt: str = ""
    chat: str = "default"
    settings: dict | None = None
    history: list | None = None

class MemoryUpdate(BaseModel):
    text: str
    collection: str = "user"
    relevance: str = "contextual"
    document_id: str | None = None
    memory_id: str | None = None

class MemoryImport(BaseModel):
    collection: str = "user"
    document_id: str | None = None



class GenerateInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"
    message_index: int | None = None
    message_nonce: str | None = None
    settings: dict | None = None
    history: list | None = None
    prompt_id: str | None = None


class UpdateAssistantInput(BaseModel):
    omd_key: str
    style: str | None = None
    system_prompt: str | None = None
    assistant_name: str | None = None
    assistant_title: str | None = None
    assistant_appearance: str | None = None
    assistant_model: str | None = None
    name: str | None = None
    defaultStorage: str | None = None

class AvatarGenerateInput(BaseModel):
    omd_key: str
    style: str | None = None
    character_lora: str | None = None
    prompt: str = ""
    settings: dict | None = None
    history: list | None = None

# ... (ommitted lines)

@app.get("/assistant")
async def assistant_info(omd_key: str | None = Depends(get_omd_key)):
    # Force reload settings from storage to ensure we have the latest data (bypass cache)
    ctx = get_ctx(omd_key, force_reload=True)
    try:
        assistant = {
            "assistant_name": ctx.settings.get("assistant_name", user_context.DEFAULT_ASSISTANT_NAME),
            "name": ctx.settings.get("name") or ctx.settings.get("username") or ctx.user_id or "User",
            "title": ctx.settings.get("assistant_title", user_context.DEFAULT_ASSISTANT_TITLE),
            "system_prompt": ctx.settings.get("system_prompt", ""),
            "assistant_appearance": ctx.settings.get("assistant_appearance", user_context.DEFAULT_ASSISTANT_APPEARANCE),
            "style": ctx.settings.get("style", ""),
            "assistant_model": ctx.settings.get("assistant_model", "Domi"),
            "defaultStorage": ctx.settings.get("defaultStorage", ""),
            "avatar_version": await core_service.get_avatar_version(ctx),
            "omd_key": ctx.omd_key or omd_key,
            "summary_threshold": core_service.SUMMARY_THRESHOLD,
            "capabilities": {
                "image_generation": await core_service.is_comfy_available(),
                "chat": True
            }
        }
        return assistant

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ... (ommitted lines)

@app.post("/updateAssistant")
async def update_assistant(request: Request):
    try:
        body = await request.json()
        logging.info(f"UpdateAssistant payload: {body}")
        data = UpdateAssistantInput(**body)
    except Exception as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    ctx = get_ctx(data.omd_key)
    try:
        # Update settings with provided values
        if data.style is not None:
            ctx.settings["style"] = data.style
        if data.system_prompt is not None:
            ctx.settings["system_prompt"] = data.system_prompt
        if data.assistant_appearance is not None:
            ctx.settings["assistant_appearance"] = data.assistant_appearance
        if data.assistant_name is not None:
            ctx.settings["assistant_name"] = data.assistant_name
        if data.assistant_title is not None:
            ctx.settings["assistant_title"] = data.assistant_title
        if data.assistant_model is not None:
            ctx.settings["assistant_model"] = data.assistant_model
        if data.name is not None:
            ctx.settings["name"] = data.name
        if data.defaultStorage is not None:
            ctx.settings["defaultStorage"] = data.defaultStorage
        
        user_context.save_user_settings(ctx)
        
        settings = ctx.settings.copy()
        settings["omd_key"] = ctx.omd_key
        
        return {"status": "ok", "settings": settings, "avatar_version": await core_service.get_avatar_version(ctx)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AvatarUpdateInput(BaseModel):
    omd_key: str
    image_path: str
    style: str | None = None
    character_lora: str | None = None
    assistant_model: str | None = None
    assistant_appearance: str | None = None

class SignoutInput(BaseModel):
    omd_key: str


# ==== Хелпер ====

def get_ctx(omd_key: str | None, force_reload: bool = False):
    if omd_key in ["undefined", "null"]:
        omd_key = ""
    return user_context.get_context_by_account(omd_key, "", force_reload)


def serve_file(filepath: str, request: Request, size: int = None) -> Response:
    if not filepath or not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    # MIME-тип
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Данные о файле
    stat = os.stat(filepath)
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)
    last_modified = email.utils.format_datetime(mtime, usegmt=True)

    # ETag на основе размера файла + mtime + параметра size
    etag_raw = f"{stat.st_mtime}-{stat.st_size}-{size}".encode()
    etag = hashlib.md5(etag_raw).hexdigest()

    # Проверка If-None-Match (ETag)
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

    # Проверка If-Modified-Since
    if_modified_since = request.headers.get("if-modified-since")
    if if_modified_since:
        try:
            ims_time = email.utils.parsedate_to_datetime(if_modified_since)
            if ims_time >= mtime.replace(microsecond=0):
                return Response(status_code=304)
        except Exception:
            pass  # игнорируем неверный формат заголовка

    headers = {
        "Cache-Control": "public, max-age=86400",  # кэш на 24 часа
        "ETag": etag,
        "Last-Modified": last_modified,
    }

    # Если нужен ресайз
    if size and mime_type.startswith("image/"):
        with Image.open(filepath) as img:
            img.thumbnail((size, size))  # уменьшение до квадратного thumbnail
            buf = io.BytesIO()
            # сохраняем в том же формате, что и оригинал
            format = img.format if img.format else "PNG"
            img.save(buf, format=format)
            buf.seek(0)
            return StreamingResponse(buf, media_type=mime_type, headers=headers)

    # Если без ресайза — обычный FileResponse
    return FileResponse(
        filepath,
        media_type=mime_type,
        filename=os.path.basename(filepath),
        headers=headers
    )


def serve_default_avatar(default_path: str, request: Request, size: int = None) -> Response:
    # 1. Try to ensure it is created on disk
    if not os.path.exists(default_path):
        try:
            from PIL import ImageDraw
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            img = Image.new("RGBA", (512, 512), color=(74, 144, 226, 255))
            draw = ImageDraw.Draw(img)
            draw.ellipse([186, 130, 326, 270], fill=(255, 255, 255, 255))
            draw.ellipse([96, 320, 416, 580], fill=(255, 255, 255, 255))
            img.save(default_path, "PNG")
            logging.info(f"Created default avatar at {default_path}")
        except Exception as err:
            logging.error(f"Could not write default avatar to disk: {err}")
            
    # 2. If it exists on disk now, serve it normally
    if os.path.isfile(default_path):
        try:
            return serve_file(default_path, request, size=size)
        except Exception as serve_err:
            logging.error(f"Error serving default avatar from disk: {serve_err}")
            
    # 3. In-memory fallback if disk operations failed
    try:
        from PIL import ImageDraw
        import io
        img = Image.new("RGBA", (512, 512), color=(74, 144, 226, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([186, 130, 326, 270], fill=(255, 255, 255, 255))
        draw.ellipse([96, 320, 416, 580], fill=(255, 255, 255, 255))
        
        if size:
            img.thumbnail((size, size))
            
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as mem_err:
        logging.error(f"Fatal error generating fallback avatar in memory: {mem_err}")
        raise HTTPException(status_code=500, detail="Could not serve default avatar")


# ==== Эндпоинты ====



@app.get("/assistant/avatar")
async def assistant_avatar(
    request: Request,
    omd_key: str | None = Depends(get_omd_key),
    size: int = 80
):
    ctx = get_ctx(omd_key)
    try:
        # Fallback to default model avatar
        avatar_path = core_service.get_assistant_avatar_path(ctx)
        return serve_file(avatar_path, request, size=size)

    except HTTPException as e:
        if e.status_code == 404:
            logging.info("Model avatar not found, serving default")
        else:
            logging.warning(f"HTTP error serving avatar: {e.detail}")
        default_path = os.path.join(core_service.APP_ROOT_DIR, core_service.AVATAR_DIR, "default.png")
        return serve_default_avatar(default_path, request, size=size)
    except Exception as e:
        logging.error(f"Error serving avatar: {e}")
        default_path = os.path.join(core_service.APP_ROOT_DIR, core_service.AVATAR_DIR, "default.png")
        return serve_default_avatar(default_path, request, size=size)


@app.get("/chat/image/{filename}")
@app.get("/generated/{filename}")
async def get_chat_image(
    filename: str,
    request: Request,
    size: int | None = None,
    omd_key: str | None = Depends(get_omd_key)
):
    ctx = get_ctx(omd_key)
    safe_filename = os.path.basename(filename)

    # 1. Check in-memory ephemeral cache first (Zero Contact / Subscriber mode)
    ephemeral_data = get_ephemeral_image(safe_filename)
    if ephemeral_data:
        if size:
            try:
                img = Image.open(io.BytesIO(ephemeral_data))
                img.thumbnail((size, size))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                ephemeral_data = buf.getvalue()
            except Exception as e:
                logging.warning(f"Failed to resize in-memory image: {e}")
        return Response(content=ephemeral_data, media_type="image/png")

    # 2. Local storage candidates (Private Mode / Owner on device)
    candidates = []
    home_dir = os.path.expanduser("~")
    clean_storage = (ctx.storage or "OnMyChat").strip("/")
    if ctx.user_id and clean_storage.startswith(ctx.user_id + "/"):
        clean_storage = clean_storage[len(ctx.user_id) + 1:]
    elif ctx.user_id and clean_storage == ctx.user_id:
        clean_storage = "OnMyChat"

    candidates.append(os.path.join(home_dir, clean_storage, "generated", safe_filename))
    candidates.append(os.path.join(home_dir, "OnMyChat", "generated", safe_filename))
    if ctx.user_id:
        candidates.append(os.path.join(core_service.APP_ROOT_DIR, USER_DATA_DIR, ctx.user_id, "generated", safe_filename))
    candidates.append(os.path.join(core_service.APP_ROOT_DIR, USER_DATA_DIR, "default", "generated", safe_filename))
    candidates.append(os.path.join(core_service.APP_ROOT_DIR, USER_DATA_DIR, "anon", "generated", safe_filename))
    candidates.append(os.path.join(core_service.APP_ROOT_DIR, "generated", safe_filename))

    for c in candidates:
        if os.path.isfile(c):
            return serve_file(c, request, size=size)

    base_user_data = os.path.join(core_service.APP_ROOT_DIR, USER_DATA_DIR)
    if os.path.isdir(base_user_data):
        for root, dirs, files in os.walk(base_user_data):
            if safe_filename in files:
                target = os.path.join(root, safe_filename)
                if os.path.isfile(target):
                    return serve_file(target, request, size=size)

    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/assistant/avatar/generate")
async def generate_avatar_endpoint(data: AvatarGenerateInput):
    ctx = get_ctx(data.omd_key)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        # Use hardcoded prompt for avatar generation as requested
        prompt = "social profile photo, office style, headshot"
        result = await core_service.generate_avatar(ctx, data.style, data.character_lora, prompt)
        if result and "image" in result:
             return {"image": result["image"], "url": result.get("url")}
        else:
             raise Exception("Failed to generate avatar")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assistant/avatar/update")
async def update_avatar_endpoint(data: AvatarUpdateInput):
    ctx = get_ctx(data.omd_key)
    try:
        # Update settings if provided
        if data.style:
            ctx.settings["style"] = data.style
        if data.assistant_model:
            ctx.settings["assistant_model"] = data.assistant_model
        if data.character_lora:
            ctx.settings["character_lora"] = data.character_lora
        if data.assistant_appearance:
            ctx.settings["assistant_appearance"] = data.assistant_appearance
        
        if data.style or data.assistant_model or data.character_lora or data.assistant_appearance:
            user_context.save_user_settings(ctx)

        # The image path provided is just filename in 'generated' folder (e.g. AVATAR_....png)
        filename = data.image_path
        
        if ctx.storage and ctx.omd_key:
             # Remote
             # Copy generated/{filename} to /avatar.png
             # We assume we can download from gateway and re-upload.
             
             base_url = user_context.GATEWAY_URL.rstrip("/")
             clean_storage_id = ctx.storage.strip("/")
             source_url = f"{base_url}/{clean_storage_id}/generated/{filename}"
             
             # Fetch the image
             resp = requests.get(source_url, headers={"Authorization": f"token:{ctx.omd_key}"})
             
             if resp.status_code != 200:
                  raise Exception(f"Failed to retrieve generated image: {resp.status_code}")
             
             img_data = resp.content
             
             # 2. Upload to avatar.png
             from utils import upload_data_to_storage
             upload_data_to_storage(ctx.omd_key, ctx.storage, "avatar.png", img_data, "image/png")
             
        else:
             # Local
             user_folder = f"{core_service.APP_ROOT_DIR}/{USER_DATA_DIR}/{ctx.user_id}/generated"
             src_path = os.path.join(user_folder, filename)
             
             if not os.path.exists(src_path):
                  raise Exception("Image file not found")
             
             # We don't really have a 'local avatar' standard path except 'avatar.png' in user root maybe?
             # But `assistant_avatar` fallback logic uses `core_service.get_assistant_avatar_path(ctx)` which returns model path.
             # Wait, `assistant_avatar` line 200 checks storage.
             # If no storage (local user), it falls back to default model avatar.
             # So local users currently CANNOT have custom avatars?
             # That seems to be the case in the current code snippet for `assistant_avatar`.
             # It checks `if ctx.storage ...` then `Fallback to default model avatar`.
             
             # We should probably support local avatar too if we want this feature to work for local users.
             # But `modals.html` logic seems to imply logged in users (storage).
             # Let's stick to storage logic for now or try to support local if easy.
             pass
 
        version = await core_service.get_avatar_version(ctx)
        return {"status": "ok", "avatar_version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/assistant/loras")
async def get_loras(request: Request, mode: str | None = Query(None), omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    return core_service.get_available_loras(ctx, mode=mode)

@app.get("/assistant/model/{lora_name}/avatar")
async def model_avatar(
    request: Request,
    lora_name: str,
    omd_key: str | None = Depends(get_omd_key),
    size: int = 80
):
    ctx = get_ctx(omd_key)
    try:
        avatar_path = core_service.get_model_avatar_path(lora_name)

        logging.info(f"Serving model avatar: {avatar_path}")
        return serve_file(avatar_path, request, size=size)
    except HTTPException as e:
        if e.status_code == 404:
            logging.info(f"Model avatar not found for {lora_name}, serving default")
        else:
            logging.warning(f"HTTP error serving model avatar: {e.detail}")
        default_path = os.path.join(core_service.APP_ROOT_DIR, core_service.AVATAR_DIR, "default.png")
        return serve_default_avatar(default_path, request, size=size)
    except Exception as e:
        logging.error(f"Error serving model avatar: {e}")
        default_path = os.path.join(core_service.APP_ROOT_DIR, core_service.AVATAR_DIR, "default.png")
        return serve_default_avatar(default_path, request, size=size)
    
    
@app.get("/assistant/avatars")
async def get_assistant_avatars_endpoint(omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    avatars = await core_service.get_generated_avatars(ctx)
    return {"status": "ok", "avatars": avatars}


@app.post("/chats/{chat}/archive")
async def archive_chat(chat: str, omd_key: str | None = Depends(get_omd_key)):
    return {"status": "ok", "chat": chat or "default"}

@app.post("/chats/{chat}/restore")
async def restore_chat(chat: str, omd_key: str | None = Depends(get_omd_key)):
    return {"status": "ok", "chat": chat or "default"}


# [LEGACY HISTORY] /history endpoints removed



# [LEGACY HISTORY] delete_history removed



@app.get("/memory")
async def memory_endpoint(omd_key: str | None = Depends(get_omd_key), collection: str = "user"):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection=collection)
        return {"collection": collection, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory")
async def update_memory(data: MemoryUpdate, omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)  
    try:
        updated_id = memory_index.update_memory_card(
            ctx=ctx,
            text=data.text,
            collection=data.collection,
            relevance=data.relevance,
            document_id=data.document_id,
            mem_id=data.memory_id
        )
        if not updated_id:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "ok", "memory_id": updated_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/import")
async def import_memory(data: MemoryImport, omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)  
    try:
        card = await core_service.import_doc(
            ctx=ctx,
            url_or_path=data.document_id,
            collection=data.collection
        )
        if card.get("error"):
            return {"status": "error", "card": card}
            
        return {"status": "ok", "card": card}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/memory/{mem_id}")
async def delete_memory(mem_id: str, omd_key: str | None = Depends(get_omd_key), collection: str = "shared" ):
    ctx = get_ctx(omd_key)
    print(f"KEY: {mem_id}")
    try:
        success = memory_index.delete_memory_card(ctx, mem_id=mem_id, collection=collection)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted", "memory_id": mem_id}
    except Exception as e:
        print (f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/{collection}/{mem_id}")
async def get_memory(collection: str, mem_id: str, omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection)
        for m in memories:
            if m["memory_id"] == mem_id:
                return m
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── On-device memory cards (unified ChromaDB, только владелец AI-ноды) ─────

def _device_cards_response():
    """Карточки on-device: memory_card + проиндексированные документы (группы file_chunk)."""
    cards = unified_memory.get_all_memory_cards() + unified_memory.get_indexed_documents()
    out = []
    for c in cards:
        c["onDevice"] = True
        c["source"] = "device"
        if not c.get("created"):
            c["created"] = c.get("timestamp", "") or ""
        if not c.get("title"):
            c["title"] = (c.get("document_id") or "").split("/")[-1]
        out.append(c)
    return out


@app.get("/api/device_memory")
async def device_memory_endpoint(request: Request, omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    if not is_private_mode(request, ctx):
        return {"memories": []}
    try:
        return {"memories": _device_cards_response()}
    except Exception as e:
        logging.error(f"[device_memory] list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/device_memory/{mem_id:path}")
async def delete_device_memory(mem_id: str, request: Request, omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    if not is_private_mode(request, ctx):
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        # document::<document_id> — сгруппированный индексируемый документ
        if mem_id.startswith("document::"):
            unified_memory.delete_document(mem_id[len("document::"):])
        else:
            unified_memory.delete_memory_card(mem_id=mem_id)
        return {"status": "deleted", "memory_id": mem_id}
    except Exception as e:
        logging.error(f"[device_memory] delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# [LEGACY HISTORY] /chats endpoints removed

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    ctx = get_ctx(data.omd_key)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        instruction=(
            "Respond to user. If user question relates to *Known facts*, be extreamly accurate, do not guess."
        )
        response = await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=data.prompt,
            chat=data.chat,
            provided_history=data.history,
            provided_knowledge=data.knowledge
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_post(request: Request, data: ChatStreamInput):
    return await chat_stream(
        request,
        prompt=data.prompt,
        omd_key=data.omd_key,
        chat=data.chat,
        provided_history=data.history,
        provided_settings=data.settings,
        provided_knowledge=data.knowledge,
        provided_prompt_id=data.prompt_id,
        chat_summary=data.chat_summary,
        total_message_count=data.total_message_count,
        image_delivery=data.image_delivery or (data.settings.get("image_delivery") if data.settings else None),
        client=data.client or (data.settings.get("client") if data.settings else None)
    )

@app.get("/chat/stream")
async def chat_stream(request: Request, prompt: str, omd_key: str | None = Depends(get_omd_key), chat: str = "default", 
                      provided_history: list|None = None, 
                      provided_settings: dict|None = None,
                      provided_knowledge: list|None = None,
                      provided_prompt_id: str|None = None,
                      chat_summary: str|None = None,
                      total_message_count: int|None = None,
                      image_delivery: str|None = None,
                      client: str|None = None):
    logging.info(f"Chat stream request: omd_key={omd_key[:10] if omd_key else 'None'}...")
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    ctx.private_mode = is_private_mode(request, ctx)

    is_inline_image = (
        image_delivery == "inline"
        or request.headers.get("x-image-delivery") == "inline"
        or (provided_settings and provided_settings.get("image_delivery") == "inline")
        or request.headers.get("x-client") == "svar"
        or client == "svar"
        or (provided_settings and provided_settings.get("client") == "svar")
    )

    if provided_settings:
        # Strip heavy fields that backend never reads — avoids sending base64 avatar every request
        provided_settings.pop("assistant_avatar", None)
        logging.info(f"Applying client-provided settings for {ctx.user_id}: {provided_settings}")
        ctx.settings.update(provided_settings)
        if provided_settings.get("defaultStorage"):
            ctx.storage = provided_settings["defaultStorage"]
    else:
        logging.info(f"NO provided_settings received for {ctx.user_id}")

    async def event_generator():
        try:
            nonlocal chat

            # 0. IMMEDIATE STATUS CONFIRMATION
            status_map_immediate = {
                "/show": "generating",
                "/view": "generating", "/imagine": "generating",
                "/generate": "generating",
                "/search": "searching",
                "/import": "learning", "/learn": "learning",
                "/recognize": "thinking", "/detect": "thinking",
                "/think": "thinking",
                "/explain": "thinking"
            }
            immediate_status = "processing"
            for prefix, status in status_map_immediate.items():
                if prompt.startswith(prefix):
                    immediate_status = status
                    break

            yield f"data: {json.dumps({'status': immediate_status})}\n\n"
            await asyncio.sleep(0.05) # Force yield to loop and flush

            # defaults
            intent = "chat"
            event = None
            # [LEGACY HISTORY] save_user_message removed
            mem_id = None
            img_source = None

            # Initialize chat if it's the first message of a new session
            if not chat or chat == "default" or chat == "newchat":
                 try:
                      chat_info = await core_service.ensure_chat(ctx, chat, prompt)
                      chat = chat_info["name"]
                      yield f"data: {json.dumps({'event': 'newchat', 'chatinfo': chat_info})}\n\n"
                 except Exception as e:
                      logging.error(f"Failed to ensure chat: {e}")
                      chat = chat or "default"

            # Enforce Rights (moved up)
            token_balance = float(request.headers.get("x-omd-token-balance", "0.0"))


            # 1. Broad Intent Detection First
            
            # Check for explicit slash commands
            explicit_map = {
                "/show": "show",
                "/view": "view", "/imagine": "view",
                "/generate": "generate",
                "/import": "import", "/learn": "import",
                "/recognize": "recognize", "/detect": "recognize",
                "/think": "think",
                "/explain": "explain",
                "/search": "search",
                "/doc": "doc",
                "/mcp": "doc"
            }
            
            intent = "chat"
            raw_intent = ""
            
            for prefix, mapped_intent in explicit_map.items():
                if prompt.startswith(prefix):
                    intent = mapped_intent
                    raw_intent = f"Explicit command: {intent}"
                    break
            
            if not raw_intent:

                # Check for RAG intent independently 
                # (so we don't accidentally class it as a tool if it isn't meant to be)
                raw_intent = await core_service.classify_user_intent(ctx, prompt, chat, provided_history=provided_history)
                lines = raw_intent.strip().split("\n", 1)
                intent_raw = lines[0].strip().lower()
                
                # Whitelist and sanitize intent
                allowed_intents = ["show", "view", "explain", "recognize", "import", "chat", "search"]
                for allowed in allowed_intents:
                    if intent_raw.startswith(allowed):
                        intent = allowed
                        break
            
            # Ensure chat existence for all intent types (crucial for 'show' intent which bypasses perform_prompt)
            # This ensures chat is in the index and has a title
            if intent != "chat": # perform_prompt handles chat intent
                 # Only if we are branching away from perform_prompt
                 try:
                      chat_info = await core_service.ensure_chat(ctx, chat, prompt)
                      chat = chat_info.get("name", chat)
                 except Exception as e:
                      logging.error(f"Failed to ensure chat for intent {intent}: {e}")
            
            logging.info(f"Intent detected: {intent} \n(raw: {raw_intent})")
            
            # Yield specialized status if it matches (overwrite thinking)
            status_map_detected = {
                "show": "generating",
                "view": "generating",
                "generate": "generating",
                "explain": "thinking",
                "think": "thinking",
                "search": "searching",
                "recognize": "thinking",
                "import": "learning"
            }
            check_intent_status = intent.split(":")[0] if ":" in intent else intent
            if check_intent_status in status_map_detected:
                logging.info(f"Notifying frontend about new status: {status_map_detected[check_intent_status]}")
                yield f"data: {json.dumps({'status': status_map_detected[check_intent_status]})}\n\n"

            # 2. Extract Memory Facts immediately (from the combined intent/memory string)
            memory_fact = memory_index.extract_memory_from_response(raw_intent)
            if memory_fact:
                try:
                    logging.info(f"Notifying frontend about new fact: {memory_fact}")
                    # memory_index.add_memory_card(ctx, memory_fact, collection="user", relevance="contextual")
                    yield f"data: {json.dumps({'newFact': memory_fact})}\n\n"
                except Exception as e:
                    logging.error(f"Error sending fact notification: {e}")

            # 3. Handle Special Primary Intents (Slash overrides)
            if prompt.startswith("/show"):
                intent = "show"
            elif prompt.startswith("/generate"):
                intent = "generate"
                img_prompt = prompt[len("/generate"):].strip()
                bypass_safety = ctx.private_mode or ctx.is_unlimited
                if not bypass_safety:
                    # Whitelist bypass for explicit content safety check
                    logging.info(f"Checking image generation safety: {img_prompt}")
                    safety_result = await core_service.check_prompt_safety(ctx, img_prompt)
                    if safety_result != "SAFE":
                        logging.info(f"Image generation safety check failed: {safety_result}")
                        warning = "I can not generate this. Subscribe to Premium plan to verify your age."
                        yield f"data: {json.dumps({'delta': warning, 'role': 'assistant', 'done': True})}\n\n"
                        return
            elif prompt.startswith("/view") or prompt.startswith("/imagine") or (intent == "view" and prompt.startswith("/")):
                intent = "view"
            elif prompt.startswith("/tools") or prompt.startswith("/docs") or prompt.startswith("/mcp"):
                intent = "tools"
            elif prompt.startswith("/import") or prompt.startswith("/learn"):  
                m = re.match(r'^/(?:import|learn)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))(?:\s+(\S+))?', prompt)
                file_path_or_url = m.group(1) or m.group(2) or m.group(3) if m else None
                collection = m.group(4) if m else "user"
                if file_path_or_url:
                    intent = f"import:{file_path_or_url}:{collection}"
            elif prompt.startswith("/recognize") or prompt.startswith("/detect"):  
                m = re.match(r'^/(?:recognize|detect)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', prompt)
                file_path_or_url = m.group(1) or m.group(2) or m.group(3) if m else None
                if file_path_or_url:
                    intent = f"recognize:{file_path_or_url}"
            elif prompt.startswith("/think"):  
                intent = "think"
            elif prompt.startswith("/explain"):
                intent = "explain"
            elif prompt.startswith("/search"):
                intent = "search"
        
            restricted_intents = ["tools", "doc"]
            
            # Check primary intent or prefixed intent (e.g. import:url)
            check_intent = intent.split(":")[0] if ":" in intent else intent

            if ctx.settings.get("content_mode", "work") == "fun" and check_intent in ["explain", "think"]:
                logging.info(f"Fun mode: intent '{check_intent}' downgraded to 'chat' (no DB search, RP preserved).")
                intent = "chat"
                check_intent = "chat"

            if not ctx.private_mode:
                logging.info(f"Token Balance: {token_balance}")
                #if token_balance <= 0:
                #     if check_intent in restricted_intents:
                #          yield f"data: {json.dumps({'delta': 'Advanced AI features are available with a Premium Plan.', 'role': 'assistant', 'done': True})}\n\n"
                #          return

            logging.info(f"Check intent: {check_intent}")
            if check_intent in ["tools", "search", "explain", "think"]:
                 # Пока ищем в unified_memory — статус thinking
                 status_msg = "executing" if check_intent == "tools" else "thinking"
                 logging.info(f"Yielding {status_msg} status")
                 yield f"data: {json.dumps({'status': status_msg})}\n\n"
                 
            if check_intent == "import" and not ctx.private_mode:
                 # Limit to 10 items for free accounts
                 memories = memory_index.load_memories(ctx, collection="shared")
                 if len(memories) >= 10:
                     yield f"data: {json.dumps({'delta': 'Free accounts are limited to 10 knowledge base items. Upgrade to Premium for unlimited storage.', 'role': 'assistant', 'done': True})}\n\n"
                     return

            if intent == "show":
                # 1️⃣ статус
                yield f"data: {json.dumps({'status': 'generating'})}\n\n"
                await asyncio.sleep(0.05)

                # 2️⃣ картинка
                # [LEGACY HISTORY] Load history removed
                # Generate prompt using loaded history, but DO NOT save yet (atomic update later)
                history = provided_history or []
                logging.info(f"Generating refined image prompt for: {prompt}")
                raw_prompt = await core_service.generate_character_image_prompt(ctx, prompt, chat, history=history)
                img_prompt, _ = core_service.extract_title_and_prompt(raw_prompt)

                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                # Generate image using prompt, DO NOT save yet
                upload_storage_flag = not is_inline_image
                res = await core_service.generate_character_image(ctx, img_prompt, chat, update_history=False, prompt_id=prompt_id, upload_storage=upload_storage_flag)
                path, title, description = res[0], res[1], res[2]
                image_url = f"/chat/image/{path}"
                img_payload = {'path': path, 'title': title, 'description': description, 'url': image_url, 'prompt': img_prompt}
                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image': img_payload, 'tokens_consumed': ctx.tokens_consumed})}\n\n"
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'status': 'typing'})}\n\n"
                await asyncio.sleep(0.05)

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image of yourself based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "TASK: Roleplay this scene. Continue conversation. Take into account the generated image, your feelings about it and the previous conversation context.\n"
                ).format(img_prompt)
                llm_message = prompt
                # [LEGACY HISTORY] save_user_message removed
            elif intent == "view":
                # 1️⃣ статус
                yield f"data: {json.dumps({'status': 'generating'})}\n\n"
                await asyncio.sleep(0.05)

                # 2️⃣ картинка
                logging.info(f"Generating refined image prompt for: {prompt}")
                raw_prompt = await core_service.generate_general_image_prompt(ctx, prompt, chat, history=provided_history)
                img_prompt, _ = core_service.extract_title_and_prompt(raw_prompt)

                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                # 3️⃣ Generate image (no character LoRA)
                upload_storage_flag = not is_inline_image
                res = await core_service.generate_general_image(ctx, img_prompt, chat, prompt_id=prompt_id, upload_storage=upload_storage_flag)
                path, title, description = res[0], res[1], res[2]
                image_url = f"/chat/image/{path}"
                img_payload = {'path': path, 'title': title, 'description': description, 'url': image_url, 'prompt': img_prompt}
                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image': img_payload, 'tokens_consumed': ctx.tokens_consumed})}\n\n"
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'status': 'typing'})}\n\n"
                await asyncio.sleep(0.05)

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "Your TASK: describe the generated image enthusiastically or provide a caption for it."
                ).format(img_prompt)
                llm_message = prompt
                # [LEGACY HISTORY] save_user_message removed

            elif intent == "explain" or intent == "think" or intent == "search":
                # Единый RAG-пайплайн: внутренний поиск + web-fallback выполняются
                # внутри _perform_prompt_gen через inject_facts (факты попадают в system prompt).
                search_query = prompt
                if prompt.lower().startswith("/search"):
                    search_query = prompt[7:].strip()

                instruction = (
                    "The user asked a question. Use the *Known facts* / *Strict facts* injected into the system prompt "
                    "to answer. Base your answer ONLY on that material where it is relevant, and mention the sources it "
                    "comes from. If the material does not actually answer the question, say so plainly instead of improvising. "
                    "Do not invent information, links, or data."
                )
                llm_message = search_query
            elif intent.startswith("recognize"): 

                if ":" in intent:
                    img_source = intent.split(":", 1)[1]
            
                instruction = (
                    "Recognize the image according to context."
                )
                llm_message = prompt
            elif intent.startswith("import"):
                doc_source = None
                collection = "user"
                card = {}
                if ":" in intent:
                    parts = intent.split(":", 2)
                    doc_source = parts[1]
                    if len(parts) > 2:
                        collection = parts[2]
                
                yield f"data: {json.dumps({'status': 'learning'})}\n\n"
                new_knowledge, card = await core_service.import_knowledge(ctx, doc_source, prompt, collection=collection)
                
                if not new_knowledge:
                    # FALLBACK: If import failed or was a directory, treat as chat so MCP can handle it
                    logging.info(f"Import yielded no knowledge. Falling back to chat intent.")
                    intent = "chat"
                    llm_message = prompt
                    instruction = (
                        "If *Known facts* are provided in your prior system prompt and they are relevant to user's query, be extremely accurate, do not guess. "
                        "If no *Known facts* provided, respond freely as a helpful conversational assistant."
                    )
                else:
                    yield f"data: {json.dumps({'new_knowledge': new_knowledge})}\n\n"    
                    logging.info(f"*New knowledge:*\n{new_knowledge}")
                    instruction=(
                        f"Base your answer on *New knowledge* ONLY, if present. *New knowledge:*\n{new_knowledge}"
                    )
                    llm_message = prompt
                    mem_id = card.get("id")
            elif intent.startswith("image"):
                pass

            elif intent == "generate":
                # Ensure chat exists and update timestamp
                chat_info = await core_service.ensure_chat(ctx, chat, img_prompt)
                
                # 1️⃣ Status: generating image
                yield f"data: {json.dumps({'status': 'generating'})}\n\n"
                await asyncio.sleep(0.05)

                # 2️⃣ Generate title from raw prompt
                img_title = await core_service.generate_title_from_prompt(ctx, img_prompt)
                
                # Format prompt with title for generate_image to parse
                formatted_prompt = f"Title: {img_title}\nImage: {img_prompt}"
                
                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                logging.info(f"Generating image for prompt {img_prompt} with title {img_title}")
                upload_storage_flag = not is_inline_image
                res = await core_service.generate_image(ctx, formatted_prompt, chat, use_default_lora = False, prompt_id=prompt_id, upload_storage=upload_storage_flag)
                path, title, description = res[0], res[1], res[2]
                image_url = f"/chat/image/{path}"
                img_payload = {'path': path, 'title': title, 'description': description, 'url': image_url, 'prompt': img_prompt}
                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image': img_payload, 'tokens_consumed': ctx.tokens_consumed, 'done': True})}\n\n"
                await asyncio.sleep(0.05)
                
                # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
                return

            elif check_intent == "tools":
                logging.info(f"[MCP ROUTE] Routing to check_and_execute_mcp. Prompt: {prompt[:50]}...")
                yield f"data: {json.dumps({'status': 'executing'})}\n\n"
                await asyncio.sleep(0.1)
                
                mode = ctx.settings.get("content_mode", "work")
                async for chunk in core_service.check_and_execute_mcp(ctx, prompt, mode=mode, provided_history=provided_history):
                    if isinstance(chunk, dict):
                        # Support direct forwarding of rich OpenCode-like events
                        if any(k in chunk for k in ["id", "action", "state", "delta", "thought_delta", "tool_call_delta", "tool_result_delta"]):
                            yield f"data: {json.dumps(chunk)}\n\n"
                        elif chunk.get("type") == "status":
                            yield f"data: {json.dumps({'status': chunk.get('content'), 'args': chunk.get('args')})}\n\n"
                        elif chunk.get("type") == "result":
                            payload = {'delta': chunk.get('content'), 'role': 'assistant', 'done': True}
                            if chunk.get("changedFiles"):
                                payload["changedFiles"] = chunk.get("changedFiles")
                            yield f"data: {json.dumps(payload)}\n\n"
                    await asyncio.sleep(0.05)
                return

            # 3️⃣ основной стрим чата
            else:

                llm_message = prompt
                instruction = (
                    "Respond freely as a helpful conversational assistant."
                )
            # 3️⃣ ответ
            async for chunk in await core_service.perform_prompt(
                ctx,
                instruction=instruction,
                message=llm_message,
                chat=chat,
                intent=intent,
                mem_id=mem_id,
                img_source=img_source,
                event=event,
                stream=True,
                provided_history=provided_history,
                provided_knowledge=provided_knowledge,
                chat_summary=chat_summary,
                total_message_count=total_message_count
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            logging.error(f"Error in event_generator: {e}")
            yield f"data: {json.dumps({'error': '⚠️ Storage error or request failed. Please try again later.', 'done': True})}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )



@app.post("/import")
async def import_endpoint(data: ImportInput):
    ctx = get_ctx(data.omd_key)
    try:
        card = await core_service.import_doc(ctx, data.url_or_path, data.collection)
        return {
           "status": "ok",
           "card": card
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memorize")
async def memorize_endpoint(data: MemorizeInput):
    ctx = get_ctx(data.omd_key)
    try:
        core_service.memorize(ctx, data.text)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recognize")
async def recognize_endpoint(
    request: Request,
    omd_key: str | None = Depends(get_omd_key),
    chat: str = Form("default"),
    prompt: str = Form(""),
    settings: str = Form(None),
    history: str = Form(None),
    file: UploadFile = File(...)
):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    if settings:
        try:
            provided_settings = json.loads(settings)
            ctx.settings.update(provided_settings)
            ctx.storage = ctx.settings.get("defaultStorage", "")
        except:
             logging.warning("Failed to parse settings in /recognize")
    
    provided_history = None
    if history:
         try:
              provided_history = json.loads(history)
         except:
              logging.warning("Failed to parse history in /recognize")

    try:
        img_bytes = await file.read()
        result = await core_service.recognize_image(ctx, img_bytes, prompt, chat, provided_history=provided_history)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/character")
async def generate_character_image(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        # generate_image returns (filename, title, description, img_data)
        is_new = data.message_nonce is None and data.message_index is None
        res = await core_service.generate_image(ctx, data.prompt, data.chat, update_history=is_new, prompt_id=data.prompt_id)
        filename, title, description = res[0], res[1], res[2]
        
        # [LEGACY HISTORY] Load history removed
        history = []
        
        if data.message_index is not None:
             # This part might still be needed if we want to return the updated description,
             # but we don't save it to local disk anymore.
             pass

        return {
            "image": filename,
            "path": filename,
            "url": f"/chat/image/{filename}",
            "title": title,
            "description": description,
            "tokens_consumed": ctx.tokens_consumed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/general")
async def generate_general_image(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        # generate_image returns (filename, title, description, img_data)
        res = await core_service.generate_image(ctx, data.prompt, data.chat, use_default_lora=False, prompt_id=data.prompt_id)
        filename, title, description = res[0], res[1], res[2]
        return {
            "image": filename,
            "path": filename,
            "url": f"/chat/image/{filename}",
            "title": title,
            "description": description,
            "tokens_consumed": ctx.tokens_consumed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/generate/prompt/character")
async def generate_character_image_prompt(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        result = await core_service.generate_character_image_prompt(ctx, data.prompt, data.chat, history=data.history)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/generate/prompt/general")
async def generate_general_image_prompt(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request, ctx)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        result = await core_service.generate_general_image_prompt(ctx, data.prompt, data.chat, history=data.history)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/updateAssistant")
async def update_assistant(request: Request):
    try:
        body = await request.json()
        logging.info(f"UpdateAssistant payload: {body}")
        data = UpdateAssistantInput(**body)
    except Exception as e:
        logging.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    ctx = get_ctx(data.omd_key)
    try:
        # Update settings with provided values
        if data.style is not None:
            ctx.settings["style"] = data.style
        if data.system_prompt is not None:
            ctx.settings["system_prompt"] = data.system_prompt
        if data.assistant_appearance is not None:
            ctx.settings["assistant_appearance"] = data.assistant_appearance
        if data.assistant_name is not None:
            ctx.settings["assistant_name"] = data.assistant_name
        if data.assistant_title is not None:
            ctx.settings["assistant_title"] = data.assistant_title
        if data.assistant_model is not None:
            ctx.settings["assistant_model"] = data.assistant_model
        
        user_context.save_user_settings(ctx)
        
        # Return settings without sensitive data if preferred, but for now returning all
        # We might want to exclude 'omd_key' or 'storage' from response if strictly needed, 
        # but the user has the key anyway.
        
        # Get new avatar version
        version = await core_service.get_avatar_version(ctx)
        
        return {
            "status": "ok", 
            "settings": ctx.settings, 
            "avatar_version": version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/notify/signout")
async def notify_signout(data: SignoutInput):
    logging.info(f"Signout notification for key {data.omd_key[:10]}...")
    return {"status": "ok"}


# --- Proxy Logic ---

async def proxy_request(url: str, request: Request, method: str = "POST"):
    """
    Proxies a request to the upstream URL, streaming the response back.
    """
    # 1. Prepare Headers
    headers = dict(request.headers)
    # Remove headers that might cause issues or are improper to forward blindly
    headers.pop("host", None)
    headers.pop("content-length", None) 
    headers.pop("connection", None)
    headers.pop("accept-encoding", None)
    headers["accept-encoding"] = "identity" # Force no encoding from upstream

    # 2. Get Body (if any)
    try:
        body = await request.body()
    except Exception:
        body = None

    # 3. Use global session for proxying
    session = await get_proxy_session()
    req = session.request(
        method=method,
        url=url,
        headers=headers,
        data=body,
        timeout=None # Streaming responses can be long
    )
    try:
        # Enter request context
        resp = await req.__aenter__()
        
        # 4. Prepare Response Headers
        response_headers = {}
        content_type = None
        for k, v in resp.headers.items():
            lk = k.lower()
            if lk == "content-type":
                content_type = v
                continue # We set it via media_type parameter
                
            if lk in [
                "connection", "keep-alive", "proxy-authenticate", 
                "proxy-authorization", "te", "trailers", 
                "transfer-encoding", "upgrade"
            ]:
                continue
            if lk == "content-security-policy":
                v = v.replace("script-src 'self'", "script-src 'self' 'unsafe-inline'")
            response_headers[k] = v

        if not content_type:
            # Fallback to aiohttp property
            content_type = resp.content_type
            if not content_type:
                logging.warning(f"[Proxy] Missing Content-Type from {url}")

        # Add buffering optimization for Nginx (essential for SSE/chunked)
        response_headers["X-Accel-Buffering"] = "no"

        async def stream_generator():
            try:
                async for chunk in resp.content.iter_any():
                    yield chunk
            finally:
                # Cleanup resources when streaming is done or fails
                await req.__aexit__(None, None, None)
                # DO NOT close global session here

        return StreamingResponse(
            stream_generator(),
            status_code=resp.status,
            media_type=content_type,
            headers=response_headers
        )
    except Exception as e:
        logging.error(f"[Proxy] Error proxying to {url}: {e}")
        # Ensure cleanup if we fail before returning the StreamingResponse (session is global)
        try:
            await req.__aexit__(None, None, None)
        except:
            pass
        raise HTTPException(status_code=502, detail=f"Proxy Error: {str(e)}")




@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    Proxies OpenAI-style chat completions to Ollama.
    Intercepts the request/response to log, translate reasoning fields,
    and convert raw text JSON tool calls into native tool calls.
    """
    target_url = f"{core_service.OLLAMA_URL}/v1/chat/completions"
    
    try:
        req_body = await request.json()
    except Exception as e:
        logging.warning(f"[OpenAI Proxy] Failed to parse request JSON: {e}")
        return await proxy_request(target_url, request, method="POST")

    model_name = req_body.get("model", "")
    stream = req_body.get("stream", False)
    messages = req_body.get("messages", [])
    has_tools = "tools" in req_body

    logging.info(f"[OpenAI Proxy] Request: model={model_name}, stream={stream}, messages={len(messages)}, has_tools={has_tools}")

    # Prepare request headers
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    headers.pop("accept-encoding", None)
    headers["accept-encoding"] = "identity"

    session = await get_proxy_session()

    if stream:
        try:
            resp = await session.post(target_url, json=req_body, headers=headers)
        except Exception as e:
            logging.error(f"[OpenAI Proxy] Error starting stream to Ollama: {e}")
            raise HTTPException(status_code=502, detail=f"Proxy Error: {str(e)}")

        async def stream_generator():
            accumulated_content = ""
            accumulated_reasoning = ""
            has_tool_calls = False
            last_chunk_data = None
            # Tool-call buffers for the question tool. We strip its argument
            # fragments from the forwarded stream and, once the tool call is
            # complete, re-emit a single tool_call chunk containing only the
            # first question. This is the hard guarantee that the model can
            # never drive a multi-question reply through the pipeline.
            question_tool_buffers: dict = {}  # tool-call index -> {"name","id","args"}

            def _finalize_question_tool_buffers():
                nonlocal question_tool_buffers
                out = []
                for idx in sorted(question_tool_buffers.keys()):
                    entry = question_tool_buffers[idx]
                    args_out = entry["args"]
                    try:
                        parsed = json.loads(entry["args"])
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        qs = parsed["questions"]
                        if len(qs) > 1:
                            parsed["questions"] = [qs[0]]
                            args_out = json.dumps(parsed, ensure_ascii=False)
                            logging.info(
                                f"[OpenAI Proxy] Rewrote {entry['name']} tool call: {len(qs)} questions -> 1"
                            )
                    synth = {
                        "id": last_chunk_data.get("id") if last_chunk_data else "chatcmpl-q",
                        "object": "chat.completion.chunk",
                        "created": last_chunk_data.get("created") if last_chunk_data else int(datetime.datetime.now().timestamp()),
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": idx,
                                    "id": entry["id"],
                                    "function": {
                                        "name": entry["name"],
                                        "arguments": args_out,
                                    },
                                }]
                            },
                            "finish_reason": None,
                        }]
                    }
                    out.append(f"data: {json.dumps(synth)}\n\n".encode("utf-8"))
                question_tool_buffers = {}
                return out

            try:
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str.startswith("data:"):
                        raw_data = line_str[5:].strip()
                        if raw_data == "[DONE]":
                            if question_tool_buffers:
                                for chunk_bytes in _finalize_question_tool_buffers():
                                    yield chunk_bytes
                            if accumulated_reasoning and not accumulated_content and not has_tool_calls:
                                logging.info("[OpenAI Proxy] Injecting spacer chunk to prevent validation error for empty content")
                                spacer_data = {
                                    "id": last_chunk_data.get("id") if last_chunk_data else "chatcmpl-spacer",
                                    "object": "chat.completion.chunk",
                                    "created": last_chunk_data.get("created") if last_chunk_data else int(datetime.datetime.now().timestamp()),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": " "},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(spacer_data)}\n\n".encode("utf-8")
                            
                            yield b"data: [DONE]\n\n"
                            continue

                        try:
                            data = json.loads(raw_data)
                            last_chunk_data = data
                            choices = data.get("choices", [])
                            
                            for choice in choices:
                                delta = choice.get("delta", {})
                                
                                if delta.get("content"):
                                    accumulated_content += delta["content"]
                                    
                                if delta.get("tool_calls"):
                                    has_tool_calls = True
                                    # Strip question-tool argument fragments and
                                    # buffer them for single-question rewriting.
                                    filtered_calls = []
                                    for tc in delta["tool_calls"]:
                                        idx = tc.get("index", 0)
                                        fn = tc.get("function", {}) or {}
                                        name = fn.get("name")
                                        args = fn.get("arguments")
                                        call_id = tc.get("id")
                                        if name and name in SINGLE_QUESTION_TOOL_NAMES:
                                            entry = question_tool_buffers.setdefault(
                                                idx, {"name": name, "id": call_id, "args": ""}
                                            )
                                            if args:
                                                entry["args"] += args
                                            continue
                                        if idx in question_tool_buffers:
                                            if args:
                                                question_tool_buffers[idx]["args"] += args
                                            continue
                                        filtered_calls.append(tc)
                                    if filtered_calls:
                                        delta["tool_calls"] = filtered_calls
                                    else:
                                        delta.pop("tool_calls", None)
                                    
                                reasoning_val = delta.pop("reasoning", None) or delta.pop("thinking", None)
                                if reasoning_val is not None:
                                    delta["reasoning_content"] = reasoning_val
                                    accumulated_reasoning += reasoning_val

                            # If the LLM signals the end of tool calls, flush the
                            # buffered (rewritten) question tool call before the
                            # finish chunk so the agent sees a complete argument.
                            if question_tool_buffers and any(c.get("finish_reason") for c in choices):
                                for chunk_bytes in _finalize_question_tool_buffers():
                                    yield chunk_bytes

                            yield f"data: {json.dumps(data)}\n\n".encode("utf-8")

                        except Exception as parse_err:
                            logging.warning(f"[OpenAI Proxy] Error transforming line: {parse_err}")
                            yield line + b"\n"
                    else:
                        yield line + b"\n"
            finally:
                resp.close()

        response_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        return StreamingResponse(stream_generator(), status_code=resp.status, headers=response_headers)

    else:
        try:
            async with session.post(target_url, json=req_body, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices", [])
                    for choice in choices:
                        msg = choice.get("message", {})
                        
                        reasoning_val = msg.pop("reasoning", None) or msg.pop("thinking", None)
                        if reasoning_val is not None:
                            msg["reasoning_content"] = reasoning_val
                            
                        if reasoning_val and not msg.get("content") and not msg.get("tool_calls"):
                            msg["content"] = " "
                            logging.info("[OpenAI Proxy] Injected spacer into message to prevent empty content error")
                            
                        # Rewrite question tool calls to keep only the first question.
                        tool_calls = msg.get("tool_calls") or []
                        for tc in tool_calls:
                            fn = tc.get("function", {}) or {}
                            name = fn.get("name")
                            if name and name in SINGLE_QUESTION_TOOL_NAMES:
                                try:
                                    parsed = json.loads(fn.get("arguments") or "{}")
                                except Exception:
                                    parsed = None
                                if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list) and len(parsed["questions"]) > 1:
                                    q_count = len(parsed["questions"])
                                    parsed["questions"] = [parsed["questions"][0]]
                                    fn["arguments"] = json.dumps(parsed, ensure_ascii=False)
                                    logging.info(
                                        f"[OpenAI Proxy] Rewrote {name} tool call (non-streaming): {q_count} questions -> 1"
                                    )
                            
                    return JSONResponse(status_code=200, content=data)
                else:
                    text = await resp.text()
                    return Response(content=text, status_code=resp.status, media_type=resp.content_type)
        except Exception as e:
            logging.error(f"[OpenAI Proxy] Error during non-streaming completions: {e}")
            raise HTTPException(status_code=502, detail=f"Proxy Error: {str(e)}")

@app.get("/v1/models")
async def openai_models(request: Request):
    """
    Proxies OpenAI-style models listing to Ollama.
    """
    target_url = f"{core_service.OLLAMA_URL}/v1/models"
    return await proxy_request(target_url, request, method="GET")

# --- Ollama Native Endpoints ---

@app.post("/v1/extract")
async def extract_knowledge(request: Request):
    """
    Extracts cleaned text from a URL or OMD path without saving/vectorizing on backend.
    """
    try:
        data = await request.json()
        url_or_path = data.get("url_or_path")
        if not url_or_path:
            raise HTTPException(status_code=400, detail="Missing url_or_path")
            
        token = request.headers.get("X-OMD-Key")
        ctx = user_context.UserContext(type="omd", user_id="system", settings={}, history=[], omd_key=token)
        
        # We use a specialized branch of import logic that only returns text
        logging.info(f"[extract] Extracting text from: {url_or_path}")
        
        # Reuse core_service logic but skip any storage
        # We'll call a modified version or just ensure import_doc for "user" is safe
        card = await core_service.import_doc(ctx, url_or_path, collection="user")
        
        if card and card.get("error"):
             raise HTTPException(status_code=500, detail=card.get("text"))
             
        return {
            "text": card.get("full_text") or "",
            "card": card.get("text") or ""
        }
        
    except Exception as e:
        logging.error(f"[extract] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def ollama_chat(request: Request):
    """
    Proxies Ollama native chat to Ollama.
    """
    target_url = f"{core_service.OLLAMA_URL}/api/chat"
    return await proxy_request(target_url, request, method="POST")

@app.post("/api/generate")
async def ollama_generate(request: Request):
    """
    Proxies Ollama generate to Ollama.
    """
    target_url = f"{core_service.OLLAMA_URL}/api/generate"
    return await proxy_request(target_url, request, method="POST")

@app.post("/api/cleanup-chroma")
async def cleanup_chroma_endpoint(request: Request):
    """
    Удаляет из ChromaDB гостевые данные (owner != node_owner && !t_omd).
    Требует AI_TOKEN. Принимает {owner: "alexey", apply: true}.
    Без apply=true — dry-run (покажет что будет удалено).
    """
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token_raw = request.headers.get("X-OMD-Ai-Token") or request.headers.get("Token") or ""
    token = token_raw.replace("Bearer ", "").replace("token:", "").strip()
    if not AI_TOKEN or token != AI_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required")

    try:
        body = await request.json()
    except Exception:
        body = {}
    owner = body.get("owner", SETTINGS.get("NODE_OWNER", ""))
    apply = body.get("apply", False)

    if not owner:
        coll = unified_memory.get_collection()
        results = coll.get(include=["metadatas"])
        owners = {}
        for meta in results.get("metadatas", []):
            o = meta.get("owner", "(none)")
            has_omd = meta.get("t_omd", 0) == 1
            key = f"{o}{' [omd]' if has_omd else ''}"
            owners[key] = owners.get(key, 0) + 1
        return {"status": "no_owner", "owners": owners, "hint": "Pass {owner: 'alexey', apply: true}"}

    if not apply:
        coll = unified_memory.get_collection()
        results = coll.get(include=["metadatas"])
        would_delete = sum(1 for m in results.get("metadatas", [])
                         if m.get("owner", "") != owner and m.get("t_omd", 0) != 1)
        return {"status": "dry_run", "owner": owner, "would_delete": would_delete,
                "total": coll.count(), "hint": "Pass {apply: true} to execute"}

    deleted = unified_memory.cleanup_guest_data([owner])
    return {"status": "done", "deleted": deleted, "owner": owner}


@app.get("/api/tags")
async def ollama_tags(request: Request):
    """
    Proxies Ollama tags (models list) to Ollama.
    """
    target_url = f"{core_service.OLLAMA_URL}/api/tags"
    return await proxy_request(target_url, request, method="GET")

@app.post("/api/show")
async def ollama_show(request: Request):
    """
    Proxies Ollama show model info to Ollama.
    """
    target_url = f"{core_service.OLLAMA_URL}/api/show"
    return await proxy_request(target_url, request, method="POST")


# --- Q&A Matching Endpoints (ChromaDB) ---

@app.post("/qa/load")
async def qa_load(data: dict):
    entries = data.get("entries", [])
    count = memory_index.qa_load_entries(entries)
    return {"ok": True, "count": count}

@app.post("/qa/match")
async def qa_match(data: dict):
    question = data.get("question", "")
    min_score = data.get("min_score", 0.8)
    top_k = data.get("top_k", 1)
    include_score = data.get("include_score", True)
    return memory_index.qa_match_query(question, min_score, top_k, include_score)


