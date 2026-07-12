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
import requests
import aiohttp
import asyncio
import json
import subprocess
import numpy as np
from typing import List, Optional

import core_service
import user_context

# Create a global session for proxying to avoid socket exhaustion
_proxy_session = None

# Track pending questions per OpenCode session.
# Populated when a question.asked SSE event arrives; consumed when the
# user's answer arrives via a separate POST. The SSE loop itself picks up
# the answer and proxies it to /question/{requestID}/reply, keeping the
# original SSE stream alive throughout.
# Structure: { session_id: { "requestID": str, "answer": str | None } }
_pending_questions: dict[str, dict] = {}
_pending_questions_metadata: dict[str, dict] = {}
_active_session_directories: dict[str, str] = {}
# Tracks already-processed question requestIDs per session to prevent duplicate
# question.asked events when SSE reconnections replay history.
_processed_question_ids: dict[str, set[str]] = {}

# Track pending permission metadata for reply/reject endpoints.
_pending_permissions_metadata: dict[str, dict] = {}

# Track child sessions already saved to .knowledge/ to prevent duplicates
_saved_knowledge_ids: set[str] = set()

# Map of frontend nonces (mr...) to backend OpenCode message IDs (msg_...)
_nonce_to_msg_id: dict[str, str] = {}



async def get_proxy_session():
    global _proxy_session
    if _proxy_session is None or _proxy_session.closed:
        _proxy_session = aiohttp.ClientSession()
    return _proxy_session

import memory_index
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


# Search Node Initialization
try:
    from search_node import SearchNode
    AI_TOKEN = SETTINGS.get("AI_TOKEN", "") # Ensure this key exists in config or is empty
    search_node = SearchNode(storage_path=BASE_INDEX_DIR, model=memory_index.get_model(), token=AI_TOKEN)
    logging.info("[api] SearchNode initialized in BASE_INDEX_DIR")
    
    # Run legacy migration on the server side (disabled as legacy files are obsolete)
    # memory_index.migrate_legacy_data()
except ImportError as e:
    logging.error(f"[api] SearchNode initialization failed: Missing dependency - {e}. Please run 'pip install chromadb' in the venv.")
    search_node = None
except Exception as e:
    logging.error(f"[api] Error initializing SearchNode: {e}")
    search_node = None

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
         raise HTTPException(status_code=422, detail="path is required in query or headers")

    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    result = search_node.delete_path(path)
    return result

@app.get("/api/urls/move")
async def move_url(request: Request):
    src = request.query_params.get("src") or request.headers.get("src")
    target = request.query_params.get("target") or request.headers.get("target")
        
    if not src or not target:
         raise HTTPException(status_code=422, detail="src and target are required in query or headers")

    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    result = search_node.move_path(src, target)
    return result

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

def is_private_mode(request: Request) -> bool:
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


class SessionKillInput(BaseModel):
    directory: str | None = None


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
            "name": ctx.settings.get("name", "User"),
            "title": ctx.settings.get("assistant_title", user_context.DEFAULT_ASSISTANT_TITLE),
            "system_prompt": ctx.settings.get("system_prompt", ""),
            "assistant_appearance": ctx.settings.get("assistant_appearance", user_context.DEFAULT_ASSISTANT_APPEARANCE),
            "style": ctx.settings.get("style", ""),
            "assistant_model": ctx.settings.get("assistant_model", "Domi"),
            "defaultStorage": ctx.settings.get("defaultStorage", ""),
            "avatar_version": await core_service.get_avatar_version(ctx),
            "omd_key": ctx.omd_key or omd_key
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

async def _force_kill_session(session_id: str, directory: str | None = None) -> str:
    """
    Forcefully kills all tasks and connections for a session.
    Returns status string: 'killed', 'no_active_task'
    """
    sid = str(session_id)
    had_work = False
    
    if not directory:
        directory = _active_session_directories.get(sid)
        
    # 1. Cancel the POST task
    if sid in active_tasks:
        had_work = True
        task = active_tasks.pop(sid, None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
    
    # 2. Cancel the SSE read task and close its session
    if sid in session_tasks:
        had_work = True
        info = session_tasks.pop(sid, None)
        if info:
            read_task = info.get("read_task")
            if read_task and not read_task.done():
                read_task.cancel()
                try:
                    await asyncio.wait_for(read_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            # Force-close the SSE session to break any stuck connection
            sse_session = info.get("sse_session")
            if sse_session and not sse_session.closed:
                try:
                    await sse_session.close()
                except Exception:
                    pass
    
    # 4. Clean up pending questions & session directories
    pq = _pending_questions.pop(sid, None)
    if pq and pq.get("requestID"):
        _pending_questions_metadata.pop(str(pq["requestID"]), None)
        
    to_remove = [req_id for req_id, meta in _pending_questions_metadata.items() if meta.get("session_id") == sid]
    for req_id in to_remove:
        _pending_questions_metadata.pop(req_id, None)
        
    # Clean up pending permission metadata for this session
    perm_to_remove = [req_id for req_id, meta in _pending_permissions_metadata.items() if meta.get("session_id") == sid]
    for req_id in perm_to_remove:
        _pending_permissions_metadata.pop(req_id, None)
    _active_session_directories.pop(sid, None)
    _processed_question_ids.pop(sid, None)
    
    # 4. Reset the global proxy session to break all lingering HTTP connections
    global _proxy_session
    if _proxy_session and not _proxy_session.closed:
        try:
            await _proxy_session.close()
        except Exception:
            pass
        _proxy_session = None  # Will be recreated on next get_proxy_session() call
        logging.info(f"[OpenCode Proxy] Global proxy session forcefully reset for session {sid}")
    
    # 5. Try to abort on the OpenCode side too
    try:
        async with aiohttp.ClientSession() as cleanup_session:
            abort_url = f"{core_service.CODE_BASE_URL}/session/{sid}/abort"
            if directory:
                import urllib.parse
                resolved_dir = resolve_session_directory(directory)
                abort_url += f"?directory={urllib.parse.quote(resolved_dir)}"
            try:
                await cleanup_session.post(abort_url, timeout=aiohttp.ClientTimeout(total=2))
            except Exception:
                pass
    except Exception:
        pass
    
    return "killed" if had_work else "no_active_task"


@app.post("/code/sessions/{session_id}/cancel")
async def cancel_session_task(session_id: str, payload: SessionKillInput | None = None):
    """
    Cancels an active OpenCode session task.
    Also resets the global proxy session to break any stuck connections.
    """
    directory = payload.directory if payload else None
    return {"status": await _force_kill_session(session_id, directory=directory)}


@app.post("/code/sessions/{session_id}/kill")
async def kill_session_task(session_id: str, payload: SessionKillInput | None = None):
    """
    Forcefully kills an OpenCode session.
    More aggressive than /cancel: cancels all tasks, closes SSE connections,
    resets the global proxy session, and attempts OpenCode-side cleanup.
    """
    directory = payload.directory if payload else None
    status = await _force_kill_session(session_id, directory=directory)
    return {"status": status}

@app.get("/assistant/loras")
async def get_loras(request: Request, mode: str | None = Query(None), omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    ctx.private_mode = is_private_mode(request)
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
        provided_prompt_id=data.prompt_id
    )

@app.get("/chat/stream")
async def chat_stream(request: Request, prompt: str, omd_key: str | None = Depends(get_omd_key), chat: str = "default", 
                      provided_history: list|None = None, 
                      provided_settings: dict|None = None,
                      provided_knowledge: list|None = None,
                      provided_prompt_id: str|None = None):
    logging.info(f"Chat stream request: omd_key={omd_key[:10] if omd_key else 'None'}...")
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    ctx.private_mode = is_private_mode(request)

    if provided_settings:
        logging.info(f"Applying client-provided settings for {ctx.user_id}")
        ctx.settings.update(provided_settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")

    async def event_generator():
        try:
            nonlocal chat

            # 0. IMMEDIATE STATUS FOR SLASH COMMANDS
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
            for prefix, status in status_map_immediate.items():
                if prompt.startswith(prefix):
                    yield f"data: {json.dumps({'status': status})}\n\n"
                    await asyncio.sleep(0.1) # Force yield to loop and flush
                    break

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


            # perform commands
            if prompt.startswith("/mode"):
                args = prompt[len("/mode"):].strip().split(maxsplit=1)
                new_mode = "work"

                if args:
                    if args[0].lower() == "fun":
                        new_mode = "fun"
                    elif args[0].lower() == "work":
                        new_mode = "work"

                llm_message = "get ready to play" if new_mode == "fun" else "calm down for now"

                if len(args) > 1:
                    llm_message = args[1].strip()
            

                ctx.settings["content_mode"] = new_mode
                logging.info(f"User: {ctx.user_id} switched mode to {new_mode}")
                user_context.save_user_settings(ctx)
                instruction = (
                    "User has switched mode to '{}'.\nPlease, act accordingly."
                ).format(new_mode)
                
                event = 'reload_chats'
            elif prompt.startswith("/reset"):
                subprocess.run(["pkill", "-f", "opencode web"], capture_output=True)
                yield f"data: {json.dumps({'delta': 'Done.', 'role': 'assistant', 'done': True})}\n\n"
                return
            else:
                # 1. Broad Intent Detection First
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
                    effective_fun = ctx.private_mode and ctx.settings.get("content_mode", "work") == "fun"
                    if not effective_fun:
                        logging.info(f"Checking image generation safety: {img_prompt}")
                        safety_result = await core_service.check_prompt_safety(ctx, img_prompt)
                        if safety_result != "SAFE":
                            logging.info(f"Image generation safety check failed: {safety_result}")
                            if ctx.private_mode:
                                warning = "I can not generate this in work mode. Switch to fun mode with /mode fun"
                            else:
                                warning = "I can not generate this. Subscribe to Premium plan to enable fun mode."
                            yield f"data: {json.dumps({'delta': warning, 'role': 'assistant', 'done': True})}\n\n"
                            return
                elif prompt.startswith("/view") or prompt.startswith("/imagine") or (intent == "view" and prompt.startswith("/")):
                    intent = "view"
                elif prompt.startswith("/tools"):
                    # Provide an immediate, reliable list of tools
                    mode = ctx.settings.get("content_mode", "work")
                    tools_list = await core_service.list_supported_tools(ctx, mode=mode)

                    # [LEGACY HISTORY] history saving removed - handled by frontend/OrbitDB

                    yield f"data: {json.dumps({'delta': tools_list, 'role': 'assistant', 'done': True})}\n\n"
                    return
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

            if ctx.settings.get("content_mode", "work") == "fun" and check_intent in ["tools", "import", "search", "explain", "think", "doc"]:
                 yield f"data: {json.dumps({'delta': 'Tools and advanced commands are not supported in fun mode.', 'role': 'assistant', 'done': True})}\n\n"
                 return

            if not ctx.private_mode:
                logging.info(f"Token Balance: {token_balance}")
                if token_balance <= 0:
                     if check_intent in restricted_intents:
                          yield f"data: {json.dumps({'delta': 'Advanced AI features are available with a Premium Plan.', 'role': 'assistant', 'done': True})}\n\n"
                          return

            logging.info(f"Check intent: {check_intent}")
            if check_intent in ["tools", "search"]:
                 status_msg = "searching" if check_intent == "search" else "executing"
                 logging.info(f"Yielding {status_msg} status")
                 yield f"data: {json.dumps({'status': status_msg})}\n\n"
                 
            if check_intent == "import" and not ctx.private_mode:
                 # Limit to 10 items for free accounts
                 memories = memory_index.load_memories(ctx, collection="shared")
                 if len(memories) >= 10:
                     yield f"data: {json.dumps({'delta': 'Free accounts are limited to 10 knowledge base items. Upgrade to Premium for unlimited storage.', 'role': 'assistant', 'done': True})}\n\n"
                     return

            if intent == "show":
                # 2️⃣ картинка
                # [LEGACY HISTORY] Load history removed
                # Generate prompt using loaded history, but DO NOT save yet (atomic update later)
                history = provided_history or []
                logging.info(f"Generating refined image prompt for: {prompt}")
                img_prompt = await core_service.generate_character_image_prompt(ctx, prompt, chat, history=history)

                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                # Generate image using prompt, DO NOT save yet
                path, title, description = await core_service.generate_character_image(ctx, img_prompt, chat, update_history=False, prompt_id=prompt_id)
                
                # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
                

                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image':{'path': path, 'title': title, 'description': description}, 'tokens_consumed': ctx.tokens_consumed})}\n\n"
                

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image of yourself based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "TASK: Roleplay this scene. Continue conversation. Take into account the generated image, your feelings about it and the previous conversation context.\n"
                ).format(img_prompt)
                llm_message = "Please describe the image or roleplay as requested."
                # [LEGACY HISTORY] save_user_message removed
            elif intent == "view":
                # 1️⃣ статус

                # 2️⃣ картинка
                logging.info(f"Generating refined image prompt for: {prompt}")
                img_prompt = await core_service.generate_general_image_prompt(ctx, prompt, chat, history=provided_history)

                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                # 3️⃣ Generate image (no character LoRA)
                
                path, title, description = await core_service.generate_general_image(ctx, img_prompt, chat, prompt_id=prompt_id)
                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image':{'path': path, 'title': title, 'description': description}, 'tokens_consumed': ctx.tokens_consumed})}\n\n"

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "Your TASK: describe the generated image enthusiastically or provide a caption for it."
                ).format(img_prompt)
                llm_message = "Please describe the image as requested."
                # [LEGACY HISTORY] save_user_message removed

            elif intent == "explain" or intent == "think":    
                instruction=(
                    "If Known facts are provided and they are relevant to user's query, you must strictly base your response only on them. "
                    "Do not invent or speculate. If no *Strict facts* are provided, do not guess, clearly separate what is factual from what is uncertain, and explicitly state the limitations."
                    "If no relevant Known facts are provided, respond freely as a helpful conversational assistant."
                )
                llm_message = prompt
            elif intent == "search":
                # Extract query from prompt (remove /search prefix if present)
                search_query = prompt
                if prompt.lower().startswith("/search"):
                    search_query = prompt[7:].strip()
                
                # 1. First search internal memory and indexed files
                search_results = await core_service.search_memory_tool(ctx, search_query)
                
                # 2. Fall back to web search if nothing found internally
                if "No relevant knowledge or files found." in search_results:
                    logging.info(f"No internal results for '{search_query}'. Falling back to web search.")
                    web_results = await core_service.search_web(ctx, search_query)
                    search_results = f"Web Search Results:\n{web_results}"
                
                # [LEGACY HISTORY] Load history removed
                history = provided_history or []
                
                instruction = (
                    f"The user asked to search for information. Here are the REAL search results (from internal knowledge or web):\n\n"
                    f"{search_results}\n\n"
                    "Summarize these results for the user in a helpful way. "
                    "CRITICAL: Use ONLY the data provided above. Do NOT invent links or information."
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
                    doc_source = parts[1] if len(parts) > 1 else None
                    collection = parts[2] if len(parts) > 2 else "user"
                if doc_source:
                    card = await core_service.import_doc(ctx, doc_source, collection=collection)   
                try:
                    if provided_knowledge is not None:
                        knowledge = provided_knowledge
                    else:
                        knowledge = memory_index.load_memories(ctx)
                except Exception as e:
                    logging.error(f"Failed to load knowledge: {e}")
                    knowledge = []
                
                new_knowledge = ""     
                if card: 
                    new_knowledge = card.get("text")
                
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
                
                # 1️⃣ статус

                # 2️⃣ Generate title from raw prompt
                img_title = await core_service.generate_title_from_prompt(ctx, img_prompt)
                
                # Format prompt with title for generate_image to parse
                formatted_prompt = f"Title: {img_title}\nImage: {img_prompt}"
                
                # Calculate prompt_id if not provided
                prompt_id = provided_prompt_id or ("p_" + core_service.hash_string(img_prompt + ctx.settings.get("style", "")))

                logging.info(f"Generating image for prompt {img_prompt} with title {img_title}")
                path, title, description = await core_service.generate_image(ctx, formatted_prompt, chat, use_default_lora = False, prompt_id=prompt_id)
                yield f"data: {json.dumps({'prompt': img_prompt, 'prompt_id': prompt_id, 'image':{'path': path, 'title': title, 'description': description}, 'tokens_consumed': ctx.tokens_consumed, 'done': True})}\n\n"
                
                # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
                return

            elif check_intent == "doc":
                logging.info(f"[MCP ROUTE] Routing document template operation to check_and_execute_mcp. Prompt: {prompt[:50]}...")
                yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
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
                provided_knowledge=provided_knowledge
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
    ctx.private_mode = is_private_mode(request)
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
        result = await core_service.recognize_image(ctx, img_bytes, prompt, chat)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/character")
async def generate_character_image(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        # generate_image returns (filename, title, description)
        is_new = data.message_nonce is None and data.message_index is None
        filename, title, description = await core_service.generate_image(ctx, data.prompt, data.chat, update_history=is_new, prompt_id=data.prompt_id)
        
        # [LEGACY HISTORY] Load history removed
        history = []
        
        if data.message_index is not None:
             # This part might still be needed if we want to return the updated description,
             # but we don't save it to local disk anymore.
             pass

        return {"image": filename, "description": description, "tokens_consumed": ctx.tokens_consumed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/general")
async def generate_general_image(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request)
    if data.settings:
        ctx.settings.update(data.settings)
        ctx.storage = ctx.settings.get("defaultStorage", "")
    try:
        # generate_image returns (filename, title, description)
        filename, title, description = await core_service.generate_image(ctx, data.prompt, data.chat, use_default_lora=False, prompt_id=data.prompt_id)
        return {"image": filename, "description": description, "tokens_consumed": ctx.tokens_consumed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/generate/prompt/character")
async def generate_character_image_prompt(request: Request, data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    ctx.private_mode = is_private_mode(request)
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
    ctx.private_mode = is_private_mode(request)
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



# ---- OpenCode Integration ----

active_tasks = {}

# Track per-session read tasks and SSE sessions for reliable force-kill
# Structure: { session_id: {"read_task": task, "sse_session": session} }
session_tasks: dict[str, dict] = {}

# Track subagent (child) session IDs detected during the current SSE stream.
# Used by _save_child_knowledge to know which sessions are children.
_child_session_ids: set[str] = set()

def resolve_session_directory(directory: str) -> str:
    if not directory:
        return directory
    import os
    home_dir = os.path.expanduser('~')
    
    # 1. If it starts with home_dir and actually exists, use it!
    if directory.startswith(home_dir) and os.path.exists(directory):
        return directory
        
    parts = [p for p in directory.split('/') if p]
    if not parts:
        return directory
        
    # 2. Try to find a suffix that exists when joined with home_dir
    for i in range(len(parts)):
        subpath = '/'.join(parts[i:])
        test_path = os.path.join(home_dir, subpath)
        if os.path.exists(test_path):
            return test_path
            
    # 3. Fallback to original resolution logic
    if len(parts) > 1 and parts[1] == "root":
        relative_path = '/'.join(parts[2:])
    else:
        relative_path = '/'.join(parts[1:])
    return os.path.join(home_dir, relative_path)


async def _save_child_knowledge(child_sid: str, workspace_dir: str):
    """Fetch a completed child session's messages and save as .knowledge/<topic>.md"""
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return
    knowledge_dir = os.path.join(workspace_dir, '.knowledge')
    os.makedirs(knowledge_dir, exist_ok=True)

    # 1. Fetch session info for the title
    try:
        session = await get_proxy_session()
        async with session.get(f"{core_service.CODE_BASE_URL}/session/{child_sid}") as resp:
            if resp.status != 200:
                logging.warning(f"[Knowledge] Session {child_sid} not found (status {resp.status})")
                return
            data = await resp.json()
            session_info = data.get("session", data)
    except Exception as e:
        logging.error(f"[Knowledge] Failed to fetch session {child_sid}: {e}")
        return

    title = session_info.get("title") or session_info.get("name") or child_sid

    # Sanitize to a safe filename
    safe_name = re.sub(r'[^\w\s-]', '', title).strip().lower()
    safe_name = re.sub(r'[-\s]+', '-', safe_name)[:80]

    # 2. Fetch messages
    try:
        session2 = await get_proxy_session()
        async with session2.get(f"{core_service.CODE_BASE_URL}/session/{child_sid}/message") as resp:
            msgs_data = await resp.json()
    except Exception as e:
        logging.error(f"[Knowledge] Failed to fetch messages for {child_sid}: {e}")
        return

    messages = msgs_data if isinstance(msgs_data, list) else msgs_data.get("messages", [])

    # Don't save if there's no meaningful content
    has_content = any(
        msg.get("role") in ("assistant", "user") and (msg.get("text") or msg.get("content"))
        for msg in messages
    )
    if not has_content:
        logging.info(f"[Knowledge] Child session {child_sid} has no content, skipping")
        return

    # 3. Build the knowledge markdown
    raw_content = ""
    for msg in messages:
        role = msg.get("role", "")
        if role in ("assistant", "user"):
            content = msg.get("text", "") or msg.get("content", "") or ""
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(p.get("text", ""))
                    else:
                        parts.append(str(p))
                content = " ".join(parts)
            if content.strip():
                raw_content += f"{role.upper()}: {content}\n\n"

    # Summarize the knowledge to save tokens for future tasks
    summarized_content = ""
    try:
        ctx = user_context.UserContext(type="system", user_id="system", settings={}, history={})
        model = core_service.get_llm_model(ctx)
        
        prompt = (
            f"You are a knowledge extraction assistant. The following is a raw transcript of a subagent's execution (task: {title}). "
            "Please extract the key findings, solutions, research results, and any important code snippets or terminal commands. "
            "Organize the extracted knowledge logically into a concise markdown document so that other agents can quickly read and understand the outcome without reading the full transcript. "
            "Ignore conversational filler, repeated errors if they were eventually resolved, and unrelated details.\n\n"
            "RAW TRANSCRIPT:\n"
            f"{raw_content[:100000]}" # Limit size to prevent overflow
        )
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": model,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        
        data = await core_service.llm_request(payload)
        if data and "message" in data:
            summarized_content = data["message"]["content"].strip()
    except Exception as e:
        logging.error(f"[Knowledge] Failed to summarize knowledge for {child_sid}: {e}")

    if not summarized_content or len(summarized_content) < 50:
        summarized_content = raw_content

    lines = [f"# {title}\n\n", f"{summarized_content}\n\n", f"---\n*Source: OpenCode session `{child_sid}`*\n"]

    filepath = os.path.join(knowledge_dir, f"{safe_name}.md")
    try:
        with open(filepath, "w") as f:
            f.write("".join(lines))
        logging.info(f"[Knowledge] Saved '{title}' → {filepath}")
    except Exception as e:
        logging.error(f"[Knowledge] Failed to write {filepath}: {e}")


@app.get("/code/sessions")
async def proxy_opencode_sessions_list(request: Request, directory: str = Query(None)):
    target_url = f"{core_service.CODE_BASE_URL}/session"
    if directory:
        import urllib.parse
        query = urllib.parse.urlencode({"directory": directory})
        target_url += f"?{query}"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            return {"sessions": data}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error listing sessions: {e}")
        return {"sessions": []}

@app.get("/code/projects")
async def proxy_opencode_projects_list(request: Request):
    target_url = f"{core_service.CODE_BASE_URL}/project"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            return {"projects": data}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error listing projects: {e}")
        return {"projects": []}

@app.post("/code/sessions")
async def proxy_opencode_sessions_create(request: Request):
    target_url = f"{core_service.CODE_BASE_URL}/session"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    headers.pop("accept-encoding", None)
    
    try:
        body_json = await request.json()
    except Exception:
        body_json = {}

    directory = body_json.pop("directory", None)
    if directory:
        import urllib.parse
        resolved_dir = resolve_session_directory(directory)
        target_url += f"?directory={urllib.parse.quote(resolved_dir)}"
        logging.info(f"[OpenCode Proxy] Resolved directory for session create: {directory} -> {resolved_dir}")
            
    body_data = json.dumps(body_json).encode('utf-8')
    headers["Content-Type"] = "application/json"
    
    try:
        async with session.post(target_url, data=body_data, headers=headers) as resp:
            data = await resp.json()
            logging.info(f"[OpenCode Proxy] Session created successfully. Response: {data}")
            if isinstance(data, dict):
                if "id" not in data and "session_id" in data:
                    data["id"] = data["session_id"]
                elif "id" not in data and "session" in data and isinstance(data["session"], dict) and "id" in data["session"]:
                    data = data["session"]
            return {"session": data}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/code/sessions/children")
async def get_all_session_children(request: Request):
    """
    Returns all known parent-child session mappings.
    Scans OpenCode sessions for parentID on the session object.
    """
    result = {}
    parent_map = {}

    try:
        sess = await get_proxy_session()
        async with sess.get(f"{core_service.CODE_BASE_URL}/session") as resp:
            if resp.status == 200:
                sessions = await resp.json()
                if isinstance(sessions, list):
                    for session in sessions:
                        sid = session.get("id")
                        if not sid:
                            continue
                        # parentID can be flat on the session or nested inside info
                        pid = (session.get("parentID") or
                               session.get("parent_id") or
                               session.get("parentId") or
                               session.get("info", {}).get("parentID") or
                               session.get("info", {}).get("parent_id") or
                               session.get("info", {}).get("parentId"))
                        if pid:
                            sid_s, pid_s = str(sid), str(pid)
                            parent_map[sid_s] = pid_s
                            result.setdefault(pid_s, []).append(sid_s)
    except Exception as e:
        logging.warning(f"[OpenCode Proxy] Failed to scan sessions for parentID: {e}")

    for pid in result:
        result[pid] = sorted(set(result[pid]))

    return {
        "children": result,
        "parentMap": parent_map
    }

@app.get("/code/sessions/{session_id}/children")
async def get_session_children(request: Request, session_id: str):
    """
    Returns child session data for a given parent session.
    Fetches each child session's info from OpenCode.
    """
    try:
        sess = await get_proxy_session()
        async with sess.get(f"{core_service.CODE_BASE_URL}/session") as resp:
            if resp.status != 200:
                return {"children": []}
            sessions = await resp.json()
    except Exception:
        return {"children": []}

    if not isinstance(sessions, list):
        return {"children": []}

    child_ids = []
    for session in sessions:
        sid = session.get("id")
        if not sid:
            continue
        pid = (session.get("parentID") or
               session.get("parent_id") or
               session.get("parentId") or
               session.get("info", {}).get("parentID") or
               session.get("info", {}).get("parent_id") or
               session.get("info", {}).get("parentId"))
        if pid and str(pid) == str(session_id):
            child_ids.append(sid)

    if not child_ids:
        return {"children": []}

    children = []
    for cid in child_ids:
        try:
            async with sess.get(f"{core_service.CODE_BASE_URL}/session/{cid}") as s_resp:
                if s_resp.status == 200:
                    data = await s_resp.json()
                    children.append(data)
                else:
                    children.append({"id": cid})
        except Exception as e:
            logging.error(f"[OpenCode Proxy] Error fetching child session {cid}: {e}")
            children.append({"id": cid})
    return {"children": children}

@app.api_route("/code/sessions/{session_id}", methods=["GET", "DELETE", "PATCH", "POST"])
async def proxy_opencode_session_item(request: Request, session_id: str):
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}"
    if request.method in ["PATCH", "POST"]:
        session = await get_proxy_session()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("connection", None)
        headers.pop("accept-encoding", None)
        headers["accept-encoding"] = "identity"
        
        try:
            body_json = await request.json()
        except Exception:
            body_json = {}
            
        directory = body_json.pop("directory", None)
        if directory:
            import urllib.parse
            resolved_dir = resolve_session_directory(directory)
            target_url += f"?directory={urllib.parse.quote(resolved_dir)}"
            logging.info(f"[OpenCode Proxy] Resolved directory for session update: {directory} -> {resolved_dir}")
                
        body_data = json.dumps(body_json).encode('utf-8')
        headers["Content-Type"] = "application/json"
        
        try:
            async with session.patch(target_url, data=body_data, headers=headers) as resp:
                data = await resp.json()
                return {"session": data}
        except Exception as e:
            logging.error(f"[OpenCode Proxy] Error updating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    return await proxy_request(target_url, request, method=request.method)

@app.post("/code/question/{request_id}/reply")
async def proxy_opencode_question_reply(request: Request, request_id: str):
    try:
        payload = await request.json()
        logging.info(f"[OpenCode Proxy] Forwarding question reply for {request_id}: {payload}")
        
        # INTERCEPT PERMISSION REPLIES SUBMITTED AS QUESTIONS
        if str(request_id) in _pending_permissions_metadata:
            logging.info(f"[OpenCode Proxy] Intercepted question reply as permission reply for {request_id}")
            p_url = f"{core_service.CODE_BASE_URL}/permission/{request_id}/reply"
            
            # Map the answer back to a permission reply payload
            answers = payload.get("answers", [])
            reply_action = "allow"
            if answers and answers[0] and answers[0][0].lower() == "reject":
                reply_action = "reject"
                
            session = await get_proxy_session()
            async with session.post(p_url, json={"reply": reply_action}) as p_resp:
                if p_resp.status not in [200, 201, 204]:
                    err_body = await p_resp.text()
                    logging.error(f"[OpenCode Proxy] Intercepted permission reply failed: {err_body}")
                    raise HTTPException(status_code=p_resp.status, detail=err_body)
                
                # Cleanup
                _pending_permissions_metadata.pop(str(request_id), None)
                return {}
        
        query_params = dict(request.query_params)
        directory = query_params.get("directory")
        if not directory:
            q_meta = _pending_questions_metadata.get(str(request_id))
            if q_meta:
                directory = q_meta.get("directory")
        
        q_url = f"{core_service.CODE_BASE_URL}/question/{request_id}/reply"
        url_params = {}
        if directory:
            url_params["directory"] = directory
        if url_params:
            import urllib.parse
            q_url += "?" + urllib.parse.urlencode(url_params)
            
        session = await get_proxy_session()
        async with session.post(q_url, json=payload) as q_resp:
            if q_resp.status not in [200, 201, 204]:
                err_body = await q_resp.text()
                logging.error(f"[OpenCode Proxy] Question reply failed ({q_resp.status}): {err_body}")
                raise HTTPException(status_code=q_resp.status, detail=err_body)
            
            res = {}
            if q_resp.status != 204:
                try:
                    res = await q_resp.json()
                except Exception:
                    try:
                        text = await q_resp.text()
                        if text:
                            res = {"result": text}
                    except Exception:
                        pass
            return res
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in question reply proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/question/{request_id}/reject")
async def proxy_opencode_question_reject(request: Request, request_id: str):
    try:
        logging.info(f"[OpenCode Proxy] Forwarding question reject for {request_id}")
        
        # INTERCEPT PERMISSION REJECTS SUBMITTED AS QUESTIONS
        if str(request_id) in _pending_permissions_metadata:
            logging.info(f"[OpenCode Proxy] Intercepted question reject as permission reject for {request_id}")
            p_url = f"{core_service.CODE_BASE_URL}/permission/{request_id}/reply"
            session = await get_proxy_session()
            async with session.post(p_url, json={"reply": "reject"}) as p_resp:
                if p_resp.status not in [200, 201, 204]:
                    err_body = await p_resp.text()
                    logging.error(f"[OpenCode Proxy] Intercepted permission reject failed: {err_body}")
                    raise HTTPException(status_code=p_resp.status, detail=err_body)
                
                # Cleanup
                _pending_permissions_metadata.pop(str(request_id), None)
                return {}
        
        query_params = dict(request.query_params)
        directory = query_params.get("directory")
        if not directory:
            q_meta = _pending_questions_metadata.get(str(request_id))
            if q_meta:
                directory = q_meta.get("directory")
                
        q_url = f"{core_service.CODE_BASE_URL}/question/{request_id}/reject"
        url_params = {}
        if directory:
            url_params["directory"] = directory
        if url_params:
            import urllib.parse
            q_url += "?" + urllib.parse.urlencode(url_params)
            
        session = await get_proxy_session()
        async with session.post(q_url) as q_resp:
            if q_resp.status not in [200, 201, 204]:
                err_body = await q_resp.text()
                logging.error(f"[OpenCode Proxy] Question reject failed ({q_resp.status}): {err_body}")
                raise HTTPException(status_code=q_resp.status, detail=err_body)
            
            res = {}
            if q_resp.status != 204:
                try:
                    res = await q_resp.json()
                except Exception:
                    try:
                        text = await q_resp.text()
                        if text:
                            res = {"result": text}
                    except Exception:
                        pass
            return res
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in question reject proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/permission/{permission_id}/reply")
async def proxy_opencode_permission_reply(request: Request, permission_id: str):
    try:
        payload = await request.json()
        logging.info(f"[OpenCode Proxy] Forwarding permission reply for {permission_id}: {payload}")

        query_params = dict(request.query_params)
        directory = query_params.get("directory")
        if not directory:
            p_meta = _pending_permissions_metadata.get(str(permission_id))
            if p_meta:
                directory = p_meta.get("directory")

        p_url = f"{core_service.CODE_BASE_URL}/permission/{permission_id}/reply"
        url_params = {}
        if directory:
            url_params["directory"] = directory
        if url_params:
            import urllib.parse
            p_url += "?" + urllib.parse.urlencode(url_params)

        session = await get_proxy_session()
        async with session.post(p_url, json=payload) as p_resp:
            if p_resp.status not in [200, 201, 204]:
                err_body = await p_resp.text()
                logging.error(f"[OpenCode Proxy] Permission reply failed ({p_resp.status}): {err_body}")
                raise HTTPException(status_code=p_resp.status, detail=err_body)

            res = {}
            if p_resp.status != 204:
                try:
                    res = await p_resp.json()
                except Exception:
                    try:
                        text = await p_resp.text()
                        if text:
                            res = {"result": text}
                    except Exception:
                        pass
            # Clean up stored permission
            _pending_permissions_metadata.pop(str(permission_id), None)
            return res
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in permission reply proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/code/session/{session_id}/permissions/{permission_id}")
async def proxy_opencode_frontend_permission_reply(request: Request, session_id: str, permission_id: str):
    """Endpoint called by the frontend when user clicks Allow Once / Allow Always / Reject."""
    try:
        payload = await request.json()
        response = payload.get("response", "reject")
        logging.info(f"[OpenCode Proxy] Frontend permission reply for {permission_id}: {response}")

        # Pass through directly — opencode accepts "once" / "always" / "reject"
        query_params = dict(request.query_params)
        directory = query_params.get("directory")
        if not directory:
            p_meta = _pending_permissions_metadata.get(str(permission_id))
            if p_meta:
                directory = p_meta.get("directory")

        p_url = f"{core_service.CODE_BASE_URL}/permission/{permission_id}/reply"
        url_params = {}
        if directory:
            import urllib.parse
            p_url += "?" + urllib.parse.urlencode({"directory": directory})

        session = await get_proxy_session()
        async with session.post(p_url, json={"reply": response}) as p_resp:
            if p_resp.status not in [200, 201, 204]:
                err_body = await p_resp.text()
                logging.error(f"[OpenCode Proxy] Frontend permission reply failed ({p_resp.status}): {err_body}")
                raise HTTPException(status_code=p_resp.status, detail=err_body)

            logging.info(f"[OpenCode Proxy] Frontend permission {permission_id} forwarded as {response}, status: {p_resp.status}")
            _pending_permissions_metadata.pop(str(permission_id), None)
            return {"status": "ok", "reply": response}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in frontend permission reply: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/code/permission/{permission_id}/reject")
async def proxy_opencode_permission_reject(request: Request, permission_id: str):
    try:
        logging.info(f"[OpenCode Proxy] Forwarding permission reject for {permission_id}")

        query_params = dict(request.query_params)
        directory = query_params.get("directory")
        if not directory:
            p_meta = _pending_permissions_metadata.get(str(permission_id))
            if p_meta:
                directory = p_meta.get("directory")

        p_url = f"{core_service.CODE_BASE_URL}/permission/{permission_id}/reply"
        url_params = {}
        if directory:
            url_params["directory"] = directory
        if url_params:
            import urllib.parse
            p_url += "?" + urllib.parse.urlencode(url_params)

        session = await get_proxy_session()
        # Reject = reply with "reject"
        async with session.post(p_url, json={"reply": "reject"}) as p_resp:
            if p_resp.status not in [200, 201, 204]:
                err_body = await p_resp.text()
                logging.error(f"[OpenCode Proxy] Permission reject failed ({p_resp.status}): {err_body}")
                raise HTTPException(status_code=p_resp.status, detail=err_body)

            res = {}
            if p_resp.status != 204:
                try:
                    res = await p_resp.json()
                except Exception:
                    try:
                        text = await p_resp.text()
                        if text:
                            res = {"result": text}
                    except Exception:
                        pass
            # Clean up stored permission
            _pending_permissions_metadata.pop(str(permission_id), None)
            return res
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in permission reject proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/code/sessions/{session_id}/message", methods=["POST"])
async def proxy_opencode_prompt(request: Request, session_id: str):
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
    
    # We need to translate the OnMyDisk payload to OpenCode format
    # OnMyDisk: { "prompt": "...", ... }
    # OpenCode: { "parts": [ { "type": "text", "text": "..." } ] }
    try:
        omd_payload = await request.json()
        prompt_text = omd_payload.get("prompt", "")
        agent = omd_payload.get("agent", "build")

        session = await get_proxy_session()
        
        # PENDING QUESTION CHECK: If there is a question awaiting an answer for
        # this session, we reply to the question directly and then start the SSE stream.
        is_question_reply = False
        pending_entry = _pending_questions.pop(str(session_id), None)
        
        if pending_entry:
            is_question_reply = True
            answer = prompt_text.strip().lower()
            rid = pending_entry["requestID"]
            logging.info(f"[OpenCode Proxy] Replying directly to pending question {rid} for session {session_id} with answer: {answer}")
            target_url = f"{core_service.CODE_BASE_URL}/question/{rid}/reply"
            opencode_payload = {"answers": [[answer]]}
        else:
            opencode_payload = {
                "parts": [
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
            if agent:
                opencode_payload["agent"] = agent
            
            # Inject custom assistant personality & settings directly into OpenCode's system parameters
            settings = omd_payload.get("settings", {})
            if settings:
                system_instructions = []
                
                # Map standard OMD settings into OpenCode context guidelines
                assistant_name = settings.get("assistant_name", "").strip()
                if assistant_name:
                    system_instructions.append(f"Your name is {assistant_name}.")
                    
                user_name = settings.get("name", "").strip()
                if user_name:
                    system_instructions.append(f"The user's name is {user_name}.")
                    
                personality = settings.get("system_prompt", "").strip()
                if personality:
                    system_instructions.append(f"Personality & Instructions:\n{personality}")
                    
                if system_instructions:
                    opencode_payload["system"] = "\n\n".join(system_instructions)
                    
            # Bind the global LLM model strictly formatted per OpenCode JSON schema requirements
            code_model = core_service.CODE_MODEL
            if "/" in code_model:
                provider_id, model_id = code_model.split("/", 1)
            else:
                provider_id = "ollama"
                model_id = code_model

            opencode_payload["model"] = {
                "providerID": provider_id,
                "modelID": model_id
            }
        
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        headers.pop("connection", None)
        headers.pop("accept-encoding", None)
        headers["Content-Type"] = "application/json"
        
        directory = omd_payload.get("directory")
        resolved_dir = None
        if directory:
            import urllib.parse
            resolved_dir = resolve_session_directory(directory)
            target_url += f"?directory={urllib.parse.quote(resolved_dir)}"
            logging.info(f"[OpenCode Proxy] Resolved directory for message: {directory} -> {resolved_dir}")
            
        if resolved_dir:
            _active_session_directories[str(session_id)] = resolved_dir
        
        async def stream_generator():
            if prompt_text.strip().startswith("/reset"):
                subprocess.run(["pkill", "-f", "opencode web"], capture_output=True)
                msg = {
                    "type": "message.part.updated",
                    "properties": {
                        "sessionID": str(session_id),
                        "part": {"id": "reset_part", "type": "text", "text": "OpenCode agent restarted."}
                    }
                }
                yield f"data: {json.dumps(msg)}\n\n".encode('utf-8')
                yield b"data: {\"done\": true}\n\n"
                return
                
            read_task = None
            event_stream_closed = False
            try:
                # 0. Yield initial status to inform UI immediately
                yield f"data: {json.dumps({'status': 'thinking'})}\n\n".encode('utf-8')
                await asyncio.sleep(0.05) # Force flush

                # 0.5. Cancel any existing active tasks for this session before starting a new stream.
                # This prevents duplicate SSE streams and state corruption that can cause infinite loops
                # (e.g. when a user answers a question and the old stream keeps processing events).
                existing_post_task = active_tasks.pop(str(session_id), None)
                if existing_post_task and not existing_post_task.done():
                    existing_post_task.cancel()
                    try:
                        await asyncio.wait_for(existing_post_task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass

                existing_session = session_tasks.pop(str(session_id), None)
                if existing_session:
                    existing_read_task = existing_session.get("read_task")
                    if existing_read_task and not existing_read_task.done():
                        existing_read_task.cancel()
                        try:
                            await asyncio.wait_for(existing_read_task, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                    existing_sse_session = existing_session.get("sse_session")
                    if existing_sse_session and not existing_sse_session.closed:
                        try:
                            await existing_sse_session.close()
                        except Exception:
                            pass

                # 1. Connect to OpenCode's Event Stream FIRST to avoid dropping events
                event_url = f"{core_service.CODE_BASE_URL}/event?filter_sessionID={session_id}"
                if resolved_dir:
                    import urllib.parse
                    event_url += f"&directory={urllib.parse.quote(resolved_dir)}"
                class _AsyncNull:
                    async def __aenter__(self): return self
                    async def __aexit__(self, *args): pass
                async with aiohttp.ClientSession() as sse_session:
                    async with _AsyncNull() as _:
                        event_resp = await sse_session.get(event_url, headers={"Accept": "text/event-stream"})
                        if event_resp.status != 200:
                            logging.error(f"[OpenCode Proxy] Failed to connect to event stream: {event_resp.status}")
                            # Fallback: Just fire and return
                            async with session.post(target_url, json=opencode_payload) as resp:
                                result = await resp.json()
                                if isinstance(result, dict):
                                    info = result.get("info")
                                    if isinstance(info, dict):
                                        user_id = info.get("parentID")
                                        ass_id = info.get("id")
                                        if user_id and omd_payload.get("user_nonce"):
                                            _nonce_to_msg_id[omd_payload["user_nonce"]] = user_id
                                            logging.info(f"[OpenCode Proxy] Mapped user nonce in fallback: {omd_payload['user_nonce']} -> {user_id}")
                                        if ass_id and omd_payload.get("assistant_nonce"):
                                            _nonce_to_msg_id[omd_payload["assistant_nonce"]] = ass_id
                                            logging.info(f"[OpenCode Proxy] Mapped assistant nonce in fallback: {omd_payload['assistant_nonce']} -> {ass_id}")
                                yield f"data: {json.dumps(result)}\n\n".encode('utf-8')
                                yield b"data: {\"done\": true}\n\n"
                            return
    
                        logging.info(f"[OpenCode Proxy] STREAM STARTED for Session: {session_id}")
    
                        # 2. Start the POST request in the background NOW
                        async def do_post():
                            try:
                                # Set infinity timeout for agentic tasks
                                timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=None)
                                logging.info(f"[OpenCode Proxy] do_post() -> POST {target_url} payload_keys={list(opencode_payload.keys())}")
                                async with session.post(target_url, json=opencode_payload, timeout=timeout) as resp:
                                    logging.info(f"[OpenCode Proxy] do_post() response: status={resp.status}")
                                    if resp.status not in [200, 201, 204]:
                                        error_text = await resp.text()
                                        logging.error(f"[OpenCode Proxy] Backend error {resp.status}: {error_text}")
                                        return {"error": f"Backend error: {resp.status}", "text": error_text}
                                    
                                    # Handle empty or non-JSON responses safely
                                    res = {}
                                    if resp.status != 204:
                                        try:
                                            res = await resp.json()
                                            logging.info(f"[OpenCode Proxy] do_post() JSON result keys={list(res.keys()) if isinstance(res, dict) else type(res)}")
                                        except Exception:
                                            try:
                                                text = await resp.text()
                                                if text:
                                                    res = {"result": text}
                                                    logging.info(f"[OpenCode Proxy] do_post() text result len={len(text)}")
                                            except Exception:
                                                pass
                                            
                                    if is_question_reply:
                                        # Wait until execution is finished to keep post_task alive
                                        while not terminal_event_received and not event_stream_closed:
                                            await asyncio.sleep(0.1)
                                    return res
                            except asyncio.TimeoutError:
                                logging.error(f"[OpenCode Proxy] POST Task timed out for {session_id}")
                                return {"error": "Request timed out"}
                            except Exception as e:
                                logging.error(f"[OpenCode Proxy] POST Task failed: {e}")
                                return {"error": f"Task failed: {str(e)}"}
                        
                        event_queue = asyncio.Queue()
                        async def read_events(content_reader, queue):
                            nonlocal event_stream_closed
                            try:
                                logging.info("[OpenCode Proxy] read_events started")
                                buffer = b""
                                chunk_count = 0
                                while True:
                                    chunk = await content_reader.read(65536)
                                    if not chunk:
                                        logging.info(f"[OpenCode Proxy] read_events EOF after {chunk_count} chunks")
                                        break
                                    chunk_count += 1
                                    if chunk_count <= 2:
                                        logging.debug(f"[OpenCode Proxy] read_events chunk #{chunk_count}: {chunk[:80]}")
                                    buffer += chunk
                                    while b"\n" in buffer:
                                        line, buffer = buffer.split(b"\n", 1)
                                        await queue.put(line)
                                if buffer:
                                    await queue.put(buffer)
                            except Exception as read_ex:
                                if terminal_event_received:
                                    logging.info(f"[OpenCode Proxy] Event stream connection closed normally: {read_ex}")
                                else:
                                    logging.error(f"[OpenCode Proxy] Error reading event stream: {read_ex}")
                            finally:
                                logging.info("[OpenCode Proxy] read_events finished")
                                event_stream_closed = True
                                await queue.put(None)
                        
                        # 3. Start the event stream reader task first to establish the socket connection
                        read_task = asyncio.create_task(read_events(event_resp.content, event_queue))
                        session_tasks[str(session_id)] = {
                            "read_task": read_task,
                            "sse_session": sse_session
                        }
                        
                        # Wait a brief moment to allow OpenCode to register the SSE event subscription
                        await asyncio.sleep(0.2)
                        
                        # 4. Start the POST request in the background
                        logging.info(f"[OpenCode Proxy] Starting POST task for session {session_id}")
                        post_task = asyncio.create_task(do_post())
                        active_tasks[str(session_id)] = post_task
                        
                        # We track the primary session and any subagent (child) sessions spawned from it
                        authorized_sids = {str(session_id)}
                        last_event_time = asyncio.get_event_loop().time()
                        last_emitted_states = {}
                        primary_message_ids = set()
                        terminal_event_received = False
                        active_tool_parts = set()
                        
                        while True:
                            now = asyncio.get_event_loop().time()
                            idle_time = now - last_event_time
                            
                            # CRITICAL: We trust the background POST task as the definitive signal for termination.
                            if post_task.done():
                                if post_task.cancelled():
                                    logging.info(f"[OpenCode Proxy] Post task was cancelled for {session_id}")
                                    break
                                if post_task.exception():
                                    logging.error(f"[OpenCode Proxy] Post task exception, closing stream.")
                                    break
                                
                                # Check if the task returned an error dict instead of raising an exception.
                                task_result = post_task.result()
                                logging.debug(f"[OpenCode Proxy] Post task completed with result: {task_result}")
                                if isinstance(task_result, dict) and "error" in task_result:
                                    logging.error(f"[OpenCode Proxy] Post task returned error: {task_result['error']}")
                                    yield f"data: {json.dumps({'error': task_result['error']})}\n\n".encode('utf-8')
                                    break

                                # If the terminal SSE event has already been received, we can close after a brief grace period (e.g. 0.5s silence) to allow trailing events.
                                if terminal_event_received and not active_tool_parts:
                                    if idle_time > 0.5:
                                        logging.info(f"[OpenCode Proxy] Post task completed, terminal event received, no active tools ({len(active_tool_parts)} active) and 0.5s silence. Closing.")
                                        break

                                # Grace period: Wait for trailing SSE events after task completion.
                                # Use a long timeout (120s) so permission dialogs don't time out.
                                if idle_time > 120.0:
                                    logging.info(f"[OpenCode Proxy] Post task completed and stream drained (120s silence). Closing.")
                                    break
                            else:
                                # We no longer time out on silence while generating.
                                pass

                            # CHECK FOR QUEUED QUESTION ANSWER before reading next event.
                            # This must run on EVERY loop iteration — the continue-on-timeout below
                            # would otherwise skip a check placed after the try block.
                            pq_entry = _pending_questions.get(str(session_id))
                            if pq_entry and pq_entry.get("answer") is not None:
                                ans = pq_entry["answer"]
                                pq_entry["answer"] = None
                                rid = pq_entry["requestID"]
                                q_url = f"{core_service.CODE_BASE_URL}/question/{rid}/reply"
                                try:
                                    async with session.post(q_url, json={"answers": [[ans]]}) as q_resp:
                                        if q_resp.status == 200:
                                            logging.info(f"[OpenCode Proxy] Queued answer sent to OpenCode: {rid}")
                                        else:
                                            err_body = await q_resp.text()
                                            logging.error(f"[OpenCode Proxy] Queued answer failed ({q_resp.status}): {err_body}")
                                except Exception as q_err:
                                    logging.error(f"[OpenCode Proxy] Queued answer exception: {q_err}")
                                _pending_questions.pop(str(session_id), None)
                                terminal_event_received = False

                            try:
                                try:
                                    line = await asyncio.wait_for(event_queue.get(), timeout=0.2)
                                except asyncio.TimeoutError:
                                    await asyncio.sleep(0.05)
                                    continue

                                if line is None:
                                    if post_task.done():
                                        logging.info(f"[OpenCode Proxy] SSE Connection closed by backend (EOF) and post_task is done. Closing stream.")
                                        break
                                    else:
                                        logging.info(f"[OpenCode Proxy] SSE Connection closed (EOF) but post_task is still running. Reconnecting to SSE stream...")
                                        if read_task and not read_task.done():
                                            read_task.cancel()
                                            try:
                                                await read_task
                                            except Exception:
                                                pass
                                        try:
                                            event_resp.close()
                                        except Exception:
                                            pass
                                        try:
                                            event_resp = await sse_session.get(event_url, headers={"Accept": "text/event-stream"})
                                            if event_resp.status == 200:
                                                logging.info(f"[OpenCode Proxy] SSE Reconnected successfully.")
                                                read_task = asyncio.create_task(read_events(event_resp.content, event_queue))
                                                session_tasks[str(session_id)] = {
                                                    "read_task": read_task,
                                                    "sse_session": sse_session
                                                }
                                                continue
                                            else:
                                                logging.error(f"[OpenCode Proxy] SSE Reconnect failed with status {event_resp.status}. Waiting before retry.")
                                                await asyncio.sleep(1.0)
                                                await event_queue.put(None)
                                                continue
                                        except Exception as recon_err:
                                            logging.error(f"[OpenCode Proxy] SSE Reconnect exception: {recon_err}. Waiting before retry.")
                                            await asyncio.sleep(1.0)
                                            await event_queue.put(None)
                                            continue
                                    
                                line_str = line.decode('utf-8').strip()
                                
                                if line_str.startswith("data: "):
                                    try:
                                        event_data = json.loads(line_str[6:])
                                        event_type = event_data.get("type", "")
                                        logging.info(f"[OpenCode Proxy] Event read: {event_type}")
                                        props = event_data.get("properties", {})
                                        info = props.get("info") or {} if isinstance(props.get("info"), dict) else {}
                                        event_sid = str(props.get("sessionID") or props.get("sessionId") or event_data.get("sessionID") or event_data.get("sessionId"))
                                        
                                        # QUESTION EVENTS (filtered by authorized_sids if present, otherwise fallback to current session)
                                        if event_type.startswith("question."):
                                            is_authorized = (event_sid in authorized_sids) or (event_sid == "None") or (not event_sid)
                                            if is_authorized:
                                                if event_type == "question.asked":
                                                     logging.info(f"[OpenCode Proxy] question.asked event received. event_sid={event_sid}, props={props}")
                                                     req_id = props.get("id")
                                                     if req_id:
                                                         # Dedup: skip if we already processed this question requestID for this session
                                                         session_processed = _processed_question_ids.setdefault(str(session_id), set())
                                                         if req_id in session_processed:
                                                             logging.info(f"[OpenCode Proxy] Skipping duplicate question.asked: {req_id}")
                                                         else:
                                                             session_processed.add(req_id)
                                                             _pending_questions[str(session_id)] = {
                                                                 "requestID": req_id,
                                                                 "answer": None,
                                                             }
                                                             _pending_questions_metadata[str(req_id)] = {
                                                                 "session_id": str(session_id),
                                                                 "directory": resolved_dir
                                                             }
                                                             logging.info(f"[OpenCode Proxy] Question stored for session {session_id}: id={req_id}, directory={resolved_dir}")
                                                             questions_arr = props.get("questions", [])
                                                             first = questions_arr[0] if questions_arr else {}
                                                             q_text = first.get("question", "")
                                                             q_opts = first.get("options", [])
                                                             yield f"data: {json.dumps({'type': 'question.asked', 'requestID': req_id, 'question': q_text, 'options': q_opts})}\n\n".encode('utf-8')
                                                             terminal_event_received = False
                                                elif event_type == "question.replied":
                                                    req_id = props.get("requestID")
                                                    _pending_questions_metadata.pop(str(req_id), None)
                                                    entry = _pending_questions.get(str(session_id))
                                                    if entry and entry.get("requestID") == req_id:
                                                        _pending_questions.pop(str(session_id), None)
                                                        logging.info(f"[OpenCode Proxy] Question {req_id} replied, cleaned up")
                                                elif event_type == "question.rejected":
                                                    req_id = props.get("requestID")
                                                    _pending_questions_metadata.pop(str(req_id), None)
                                                    entry = _pending_questions.get(str(session_id))
                                                    if entry and entry.get("requestID") == req_id:
                                                        _pending_questions.pop(str(session_id), None)
                                                        logging.info(f"[OpenCode Proxy] Question {req_id} rejected, cleaned up")
                                            continue
                                        
                                        # PERMISSION EVENTS (filtered by authorized_sids if present, otherwise fallback to current session)
                                        if event_type.startswith("permission."):
                                            is_authorized = (event_sid in authorized_sids) or (event_sid == "None") or (not event_sid)
                                            if is_authorized:
                                                if event_type in ("permission.updated", "permission.v2.asked", "permission.asked"):
                                                     logging.info(f"[OpenCode Proxy] {event_type} event received. event_sid={event_sid}, props={props}")
                                                     permission_id = props.get("id")
                                                     if permission_id:
                                                         # Store permission metadata so the frontend reply endpoint can find it
                                                         _pending_permissions_metadata[str(permission_id)] = {
                                                             "session_id": str(session_id),
                                                             "directory": resolved_dir,
                                                         }
                                                         # Yield the permission event to the frontend so it can render the permission UI
                                                         perm_target = (props.get("patterns") or ["unknown"])[0] if props.get("patterns") else "unknown"
                                                         perm_meta = props.get("metadata") or {}
                                                         perm_chunk = {
                                                             "type": "permission.asked",
                                                             "permissionID": permission_id,
                                                             "permission": {
                                                                 "id": permission_id,
                                                                 "title": "Запрос доступа к файловой системе",
                                                                 "patterns": props.get("patterns", []),
                                                                 "metadata": {
                                                                     "Reason": perm_meta.get("command") or perm_meta.get("description") or "Доступ к внешней директории",
                                                                     "Target": perm_target,
                                                                 }
                                                             }
                                                         }
                                                         logging.info(f"[OpenCode Proxy] Forwarding permission {permission_id} to frontend UI")
                                                         yield f"data: {json.dumps(perm_chunk)}\n\n".encode('utf-8')
                                                         terminal_event_received = False
                                                         continue

                                                elif event_type in ("permission.v2.replied", "permission.replied", "permission.v2.rejected", "permission.rejected"):
                                                     req_id = props.get("requestID")
                                                     if req_id:
                                                         _pending_permissions_metadata.pop(str(req_id), None)
                                                         logging.info(f"[OpenCode Proxy] Permission {req_id} resolved, cleaned up")
                                            continue
                                        
                                        # DYNAMIC REGISTRY: If this session claims our target as parent, authorize it
                                        parent_sid = props.get("parentID") or info.get("parentID")
                                        if parent_sid and str(parent_sid) in authorized_sids:
                                            if event_sid not in authorized_sids:
                                                logging.info(f"[OpenCode Proxy] AUTHORIZING SUBAGENT session: {event_sid} (Parent: {parent_sid})")
                                                authorized_sids.add(event_sid)
                                                _child_session_ids.add(event_sid)
                                                # Notify the frontend about the new child session
                                                yield f"data: {json.dumps({'action': 'child_session_created', 'childId': event_sid, 'parentId': str(parent_sid)})}\n\n".encode('utf-8')

                                        # Only process events for authorized sessions (Primary + Subagents)
                                        logging.debug(f"[OpenCode Proxy] Event: {event_type} | event_sid: {event_sid} | authorized: {event_sid in authorized_sids} | authorized_sids: {list(authorized_sids)}")
                                        if event_sid in authorized_sids:
                                            # DEBUG: Trace EVERY event for authorized sessions
                                            logging.debug(f"[OpenCode Proxy] EVENT: {event_type} | SID: {event_sid} | Primary: {event_sid == str(session_id)}")

                                            last_event_time = asyncio.get_event_loop().time()
                                            
                                            if event_type == "session.updated":
                                                title = (info.get("title") or 
                                                         props.get("title") or 
                                                         event_data.get("title") or 
                                                         (props.get("session", {}).get("title") if isinstance(props.get("session"), dict) else None))
                                                if title:
                                                    logging.info(f"[OpenCode Proxy] Stream rename event received: {title}")
                                                    yield f"data: {json.dumps({'action': 'rename', 'title': title})}\n\n".encode('utf-8')
                                            
                                            elif event_type == "session.diff":
                                                yield f"data: {json.dumps({'action': 'refresh_diffs'})}\n\n".encode('utf-8')
                                            
                                            elif event_type == "session.idle":
                                                if event_sid == str(session_id):
                                                    logging.info(f"[OpenCode Proxy] Primary SESSION IDLE event. Generation complete.")
                                                    terminal_event_received = True
                                                else:
                                                    logging.info(f"[OpenCode Proxy] Subagent session idle: {event_sid}")
                                                    # Auto-save explore results to .knowledge/
                                                    eid = str(event_sid)
                                                    if eid not in _saved_knowledge_ids and eid in _child_session_ids:
                                                        _saved_knowledge_ids.add(eid)
                                                        parent_dir = _active_session_directories.get(str(session_id))
                                                        if parent_dir:
                                                            asyncio.create_task(_save_child_knowledge(eid, parent_dir))
                                                            
                                            elif event_type == "session.status":
                                                status_info = props.get("status", {})
                                                if isinstance(status_info, dict):
                                                    status_type = status_info.get("type")
                                                    if status_type == "idle":
                                                        if event_sid == str(session_id):
                                                            logging.info(f"[OpenCode Proxy] Primary SESSION STATUS IDLE event. Generation complete.")
                                                            terminal_event_received = True
                                                        else:
                                                            logging.info(f"[OpenCode Proxy] Subagent status idle: {event_sid}")
                                                            eid = str(event_sid)
                                                            if eid not in _saved_knowledge_ids and eid in _child_session_ids:
                                                                _saved_knowledge_ids.add(eid)
                                                                parent_dir = _active_session_directories.get(str(session_id))
                                                                if parent_dir:
                                                                    asyncio.create_task(_save_child_knowledge(eid, parent_dir))
                                                    elif status_type in ["thinking", "running", "generating"]:
                                                        if event_sid == str(session_id):
                                                            logging.info(f"[OpenCode Proxy] Primary SESSION STATUS ACTIVE ({status_type}). Resetting terminal status.")
                                                            terminal_event_received = False
                                            
                                            elif event_type in ["message.created", "message.updated"]:
                                                msg_role = info.get("role")
                                                msg_id = info.get("id")
                                                
                                                if msg_role == "assistant":
                                                    if msg_id:
                                                        is_new = (msg_id not in primary_message_ids)
                                                        is_first_msg = (len(primary_message_ids) == 0)
                                                        
                                                        if is_new:
                                                            primary_message_ids.add(msg_id)
                                                            if event_sid == str(session_id):
                                                                terminal_event_received = False
                                                            # Start a new bubble if this isn't the very first assistant message of the stream
                                                            if not is_first_msg:
                                                               logging.info(f"[OpenCode Proxy] NEW assistant message: {msg_id}. Switching bubbles.")
                                                               yield f"data: {json.dumps({'action': 'new_message', 'role': 'assistant', 'message_id': msg_id})}\n\n".encode('utf-8')
                                                            else:
                                                               logging.debug(f"[OpenCode Proxy] Tracking primary response message: {msg_id}")
                                                        else:
                                                           logging.debug(f"[OpenCode Proxy] Tracking primary response message: {msg_id}")
                                                        
                                                        # Special check: If this message is UPDATED and has a completion time, it's NOT a terminal event for the stream anymore, 
                                                        # as there might be more messages in the same turn. we only rely on session.status: idle.
                                                        if event_type == "message.updated" and info.get("time", {}).get("completed"):
                                                            if msg_id in primary_message_ids:
                                                                logging.debug(f"[OpenCode Proxy] Message {msg_id} marked as completed in metadata.")
                                                    
                                            elif event_type == "message.part.delta":
                                                if event_sid == str(session_id) and terminal_event_received:
                                                    # Stale delta after idle — stop streaming token-by-token
                                                    pass
                                                else:
                                                    if event_sid == str(session_id):
                                                        terminal_event_received = False
                                                    chunk = {
                                                        "id": props.get("partID") or props.get("id"),
                                                        "delta": props.get("delta"),
                                                        "field": props.get("field", "text"),
                                                        "type": "thought" if props.get("field") == "thought" else "text"
                                                    }
                                                    yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                                                
                                            elif event_type in ["message.part.updated", "part.update", "message.part.created"]:
                                                part = props.get("part") or props or {}
                                                part_id = part.get("id") or props.get("partID")
                                                part_type = part.get("type") or props.get("type")
                                                state_obj = part.get("state")
                                                
                                                if part_type == "tool" and part_id:
                                                    if isinstance(state_obj, dict):
                                                        status = state_obj.get("status")
                                                        if status in ["success", "error", "finished"]:
                                                            if part_id in active_tool_parts:
                                                                logging.info(f"[OpenCode Proxy] Tool part {part_id} finished with status: {status}")
                                                                active_tool_parts.remove(part_id)
                                                        else:
                                                            logging.info(f"[OpenCode Proxy] Tool part {part_id} is active (status: {status})")
                                                            active_tool_parts.add(part_id)
                                                            if event_sid == str(session_id):
                                                                terminal_event_received = False
                                                
                                                if isinstance(state_obj, dict):
                                                    state_copy = dict(state_obj); state_copy.pop("time", None)
                                                    state_str = json.dumps(state_copy, sort_keys=True)
                                                    if last_emitted_states.get(part_id) != state_str:
                                                        last_emitted_states[part_id] = state_str
                                                        chunk = { "id": part_id, "type": part_type or part.get("type"), "state": state_obj, "action": "part_update" }
                                                        yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                                                else:
                                                    chunk = { "id": part_id, "type": part_type or part.get("type"), "state": state_obj, "action": "part_update" }
                                                    yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                                                
                                            elif event_type in ["task.finished", "task.error", "session.completed", "task.closed"]:
                                                if event_sid == str(session_id):
                                                    logging.info(f"[OpenCode Proxy] TERMINAL SSE EVENT: {event_type}. Generation complete.")
                                                    terminal_event_received = True
                                                    
                                    except json.JSONDecodeError:
                                        pass 
                            except Exception as loop_e:
                                logging.error(f"[OpenCode Proxy] Error in event loop: {loop_e}")
                                break
                        
                        # 4. Final cleanup
                        if not post_task.done():
                            await post_task
                        
                        yield b"data: {\"done\": true}\n\n"
                        
            except Exception as e:
                logging.error(f"[OpenCode Proxy] Stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')
            finally:
                if 'event_resp' in locals() and event_resp:
                    try:
                        event_resp.close()
                    except Exception:
                        pass
                if read_task and not read_task.done():
                    read_task.cancel()
                    try:
                        await read_task
                    except asyncio.CancelledError:
                        pass
                if str(session_id) in active_tasks:
                    task = active_tasks.pop(str(session_id), None)
                    if task and not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                # Clean up any pending question entry for this session
                pq = _pending_questions.pop(str(session_id), None)
                if pq and pq.get("requestID"):
                    _pending_questions_metadata.pop(str(pq["requestID"]), None)
                # Clean up any pending permission metadata for this session
                perm_meta_remove = [rid for rid, m in _pending_permissions_metadata.items() if m.get("session_id") == str(session_id)]
                for rid in perm_meta_remove:
                    _pending_permissions_metadata.pop(rid, None)
                session_tasks.pop(str(session_id), None)
                _processed_question_ids.pop(str(session_id), None)
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error in prompt proxy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/code/changes")
async def proxy_opencode_changes(request: Request, session_id: str = Query(...)):
    """
    OnMyDisk expects: { "changes": [ { "id": "msg_id", "title": "...", "timestamp": ... } ] }
    OpenCode returns: [ MessageV2, ... ]
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
    
    # Use proxy_request but we need to intercept the response for transformation
    # Optimization: if we don't want to re-implement proxy_request here, 
    # we can just fetch it manually.
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            changes = []
            if isinstance(data, list):
                for msg in data:
                    # Assistant messages are the ones that "change" things
                    if msg.get("info", {}).get("role") == "assistant":
                        msg_info = msg.get("info", {})
                        changes.append({
                            "id": msg_info.get("id"),
                            "title": msg_info.get("title") or f"Change {msg_info.get('id')[:8]}",
                            "timestamp": msg_info.get("time", {}).get("updated") or msg_info.get("time", {}).get("created")
                        })
            return {"changes": changes}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error fetching changes: {e}")
        return {"changes": []}

def shorten_patch(patch_str, n=1):
    if not patch_str: return ""
    lines = patch_str.splitlines()
    output = []
    hunk_started = False
    hunk_lines = []
    
    def process_hunk(h_lines):
        if not h_lines: return []
        change_indices = []
        for idx, line in enumerate(h_lines):
            if idx == 0 and line.startswith("@@"):
                continue
            if (line.startswith("+") and not line.startswith("+++")) or (line.startswith("-") and not line.startswith("---")):
                change_indices.append(idx)
        if not change_indices:
            return h_lines
            
        keep_indices = set()
        for idx in range(len(h_lines)):
            if idx == 0:
                continue
            if idx in change_indices:
                keep_indices.add(idx)
                continue
            for c_idx in change_indices:
                if abs(idx - c_idx) <= n:
                    keep_indices.add(idx)
                    break
                    
        shortened = []
        shortened.append(h_lines[0])  # Keep hunk header
        last_idx = 0
        for idx in sorted(keep_indices):
            if last_idx > 0 and idx > last_idx + 1:
                shortened.append("...")
            shortened.append(h_lines[idx])
            last_idx = idx
        return shortened

    for line in lines:
        if line.startswith("@@"):
            if hunk_started:
                output.extend(process_hunk(hunk_lines))
                hunk_lines = []
            hunk_started = True
            hunk_lines.append(line)
        elif hunk_started:
            if line.startswith("diff --git") or line.startswith("Index:") or (line.startswith("--- ") and not line.startswith("--- \t") and not hunk_lines[-1].startswith("@@")):
                output.extend(process_hunk(hunk_lines))
                hunk_lines = []
                hunk_started = False
                output.append(line)
            else:
                hunk_lines.append(line)
        else:
            output.append(line)
            
    if hunk_started:
        output.extend(process_hunk(hunk_lines))
    return "\n".join(output)

def get_local_git_diff(directory: str):
    if not directory or not os.path.exists(directory):
        return []
    
    # Resolve directory to absolute path
    directory = os.path.abspath(directory)
    
    # We want to find any Git repositories under this directory (up to depth 2)
    git_repos = []
    if os.path.exists(os.path.join(directory, ".git")):
        git_repos.append(directory)
    else:
        try:
            for item in os.listdir(directory):
                sub = os.path.join(directory, item)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, ".git")):
                    git_repos.append(sub)
        except Exception:
            pass
            
    if not git_repos:
        return []
        
    diffs = []
    import subprocess
    for repo in git_repos:
        try:
            # 1. Run git status --porcelain
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo, capture_output=True, text=True, timeout=2
            )
            if status_res.returncode != 0:
                continue
                
            files_status = {}
            for line in status_res.stdout.splitlines():
                if len(line) > 3:
                    code = line[:2].strip()
                    file_path = line[3:].strip()
                    # Handle quoted git paths
                    if file_path.startswith('"') and file_path.endswith('"'):
                        file_path = file_path[1:-1]
                    status_name = "modified"
                    if "A" in code or "?" in code:
                        status_name = "added"
                    elif "D" in code:
                        status_name = "deleted"
                    files_status[file_path] = status_name
                    
            if not files_status:
                continue
                
            # 2. Run git diff --numstat
            numstat_res = subprocess.run(
                ["git", "diff", "--numstat"],
                cwd=repo, capture_output=True, text=True, timeout=2
            )
            stats = {}
            if numstat_res.returncode == 0:
                for line in numstat_res.stdout.splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        adds, dels, fpath = parts[0], parts[1], parts[2]
                        additions = 0 if adds == "-" else int(adds)
                        deletions = 0 if dels == "-" else int(dels)
                        stats[fpath] = (additions, deletions)
                        
            # 3. For each file, get its diff
            for fpath, status in files_status.items():
                diff_res = subprocess.run(
                    ["git", "diff", "--", fpath],
                    cwd=repo, capture_output=True, text=True, timeout=2
                )
                diff_content = diff_res.stdout if diff_res.returncode == 0 else ""
                
                adds, dels = stats.get(fpath, (0, 0))
                if adds == 0 and dels == 0 and status == "added":
                    try:
                        with open(os.path.join(repo, fpath), "r", encoding="utf-8", errors="ignore") as f:
                            adds = len(f.readlines())
                    except Exception:
                        adds = 0
                
                rel_path = os.path.relpath(os.path.join(repo, fpath), directory)
                
                diffs.append({
                    "file": rel_path,
                    "additions": adds,
                    "deletions": dels,
                    "status": status,
                    "diff": diff_content
                })
        except Exception as e:
            logging.error(f"[OpenCode Proxy] Error running git diff for {repo}: {e}")
            
    return diffs

@app.get("/code/diff")
async def proxy_opencode_diff(request: Request, change_id: str = Query(...), session_id: str = Query(...)):
    """
    OnMyDisk expects: { "diff": "unified diff string" }
    OpenCode returns: [ { file, diff, ... }, ... ]
    """
    # First, get session directory to ensure we pass it to OpenCode server's diff API
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)

    directory = None
    try:
        session_info_url = f"{core_service.CODE_BASE_URL}/session/{session_id}"
        async with session.get(session_info_url, headers=headers) as info_resp:
            if info_resp.status == 200:
                session_info = await info_resp.json()
                directory = session_info.get("directory")
    except Exception as ex:
        logging.error(f"[OpenCode Proxy] Failed to fetch session info for diff: {ex}")

    import urllib.parse
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/diff?messageID={change_id}"
    if directory:
        target_url += f"&directory={urllib.parse.quote(directory)}"
    
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            full_diff = ""
            diffs = data if isinstance(data, list) else []
            
            # Fallback to local git diff if empty
            if not diffs and directory:
                diffs = get_local_git_diff(directory)
                
            for file_diff in diffs:
                full_diff += f"File: {file_diff.get('file')}\n"
                diff_content = file_diff.get('diff') or file_diff.get('patch', '')
                full_diff += shorten_patch(diff_content, n=1) + "\n\n"
            return {"diff": full_diff}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error fetching diff: {e}")
        return {"diff": "Error loading diff"}

@app.post("/code/apply")
async def proxy_opencode_apply(request: Request):
    # OpenCode applies changes immediately. 
    # We can just return success or trigger a compaction/summary if needed.
    return {"status": "success"}

# ----------------------------

@app.get("/code/sessions/{session_id}/diffs")
async def proxy_opencode_session_diffs(request: Request, session_id: str, message_id: str = Query(None)):
    """
    Returns an array of file modifications with added/deleted line stats for the entire session or specific message.
    """
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    import difflib
    
    def inject_diffs(diff_list):
        if not isinstance(diff_list, list): return []
        import copy
        new_list = copy.deepcopy(diff_list)
        for item in new_list:
            if "diff" in item and item["diff"]:
                item["diff"] = shorten_patch(item["diff"], n=1)
                continue
            if "patch" in item and item["patch"]:
                item["diff"] = shorten_patch(item["patch"], n=1)
                continue
            file_name = item.get("file", "unknown")
            before = item.get("before")
            after = item.get("after")
            if before is not None or after is not None:
                before_lines = (before or "").splitlines(keepends=True)
                after_lines = (after or "").splitlines(keepends=True)
                diff_gen = difflib.unified_diff(
                    before_lines, after_lines,
                    fromfile=f"a/{file_name}", tofile=f"b/{file_name}", n=1
                )
                item["diff"] = "".join(diff_gen)
            else:
                item["diff"] = ""
        return new_list

    try:
        # First, fetch the session directory to pass to the diff API for correct workspace matching
        directory = None
        try:
            session_info_url = f"{core_service.CODE_BASE_URL}/session/{session_id}"
            async with session.get(session_info_url, headers=headers) as info_resp:
                if info_resp.status == 200:
                    session_info = await info_resp.json()
                    directory = session_info.get("directory")
        except Exception as ex:
            logging.error(f"[OpenCode Proxy] Failed to fetch session info: {ex}")

        import urllib.parse
        dir_param = f"&directory={urllib.parse.quote(directory)}" if directory else ""

        if message_id:
            # First, fetch all messages to find the parent user message ID
            session_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
            async with session.get(session_url, headers=headers) as resp:
                data = await resp.json()
                messages = data if isinstance(data, list) else []
                
                # Locate the assistant message with this ID
                assistant_msg = next((m for m in messages if m.get("info", {}).get("id") == message_id), None)
                if assistant_msg:
                    parent_id = assistant_msg.get("info", {}).get("parentID")
                    if parent_id:
                        # Try to get diffs directly from parent user message summary first!
                        parent_msg = next((m for m in messages if m.get("info", {}).get("id") == parent_id), None)
                        if parent_msg:
                            summary = parent_msg.get("info", {}).get("summary") or {}
                            diffs = summary.get("diffs")
                            if diffs:
                                return {"diffs": inject_diffs(diffs)}
                        
                        # Fetch the diff using the parent user message ID as messageID
                        target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/diff?messageID={parent_id}{dir_param}"
                        async with session.get(target_url, headers=headers) as diff_resp:
                            diff_data = await diff_resp.json()
                            diffs = diff_data if isinstance(diff_data, list) else []
                            
                            # Fallback to local git diff if empty
                            if not diffs and directory:
                                diffs = get_local_git_diff(directory)
                            return {"diffs": inject_diffs(diffs)}
                return {"diffs": []}
        else:
            # Cumulative session diff
            target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/diff"
            if directory:
                target_url += f"?directory={urllib.parse.quote(directory)}"
            async with session.get(target_url, headers=headers) as resp:
                data = await resp.json()
                diffs = data if isinstance(data, list) else []
                
                # Fallback to local git diff if session diff is empty
                if not diffs and directory:
                    diffs = get_local_git_diff(directory)
                
                return {"diffs": inject_diffs(diffs)}

    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error fetching session diffs: {e}")
        return {"diffs": []}

@app.post("/code/sessions/{session_id}/revert")
@app.post("/code/session/{session_id}/revert")
async def proxy_opencode_revert(request: Request, session_id: str):
    """
    Triggers OpenCode backend to revert file snapshots effectively undoing code generations since the specified messageID.
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/revert"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    headers.pop("accept-encoding", None)
    
    try:
        req_data = await request.json()
        logging.info(f"[OpenCode Proxy] Revert payload: {req_data}")
        # Translate message_id -> messageID for OpenCode backend
        if "message_id" in req_data:
            msg_id = req_data.pop("message_id")
            if msg_id.startswith("mr") and msg_id in _nonce_to_msg_id:
                translated = _nonce_to_msg_id[msg_id]
                logging.info(f"[OpenCode Proxy] Translating revert nonce {msg_id} -> {translated}")
                msg_id = translated
            req_data["messageID"] = msg_id
    except:
        req_data = {}
        
    try:
        async with session.post(target_url, headers=headers, json=req_data) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = await resp.json()
                return JSONResponse(status_code=resp.status, content=data)
            else:
                text = await resp.text()
                return JSONResponse(status_code=resp.status, content={"status": "error" if resp.status >= 400 else "success", "message": text})
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Exception during session revert: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/code/sessions/{session_id}/messages")
@app.get("/code/session/{session_id}/messages")
@app.get("/code/session/{session_id}/message")
async def proxy_opencode_messages(request: Request, session_id: str):
    """
    Returns all messages for a session.
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            return data
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error fetching messages: {e}")
        return {"messages": []}

@app.delete("/code/sessions/{session_id}/message/{message_id}")
@app.delete("/code/sessions/{session_id}/messages/{message_id}")
@app.delete("/code/session/{session_id}/message/{message_id}")
async def proxy_delete_message(request: Request, session_id: str, message_id: str):
    """
    Surgically deletes a message from a session.
    """
    if message_id.startswith("mr") and message_id in _nonce_to_msg_id:
        translated = _nonce_to_msg_id[message_id]
        logging.info(f"[OpenCode Proxy] Translating delete message nonce {message_id} -> {translated}")
        message_id = translated

    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message/{message_id}"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    
    try:
        async with session.delete(target_url, headers=headers) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                data = await resp.json()
                return JSONResponse(status_code=resp.status, content=data)
            else:
                text = await resp.text()
                return JSONResponse(status_code=resp.status, content={"status": "error" if resp.status >= 400 else "success", "message": text})
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error deleting message {message_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/code/sessions/{session_id}")
@app.delete("/code/session/{session_id}")
async def proxy_opencode_delete_session(request: Request, session_id: str):
    """
    Deletes an entire coding session.
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}"
    
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "X-OMD-Key": request.headers.get("X-OMD-Key", "")
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(target_url, headers=headers) as resp:
                return JSONResponse(status_code=resp.status, content={"status": "ok"})
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error deleting session {session_id}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.api_route("/code/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def opencode_proxy(request: Request, path: str):
    """
    Proxies all /code requests.
    Intelligently routes v1/ and api/ to the AI backend, 
    and everything else to the OpenCode frontend at port 4096.
    """

    # Only route specific Ollama paths to the Ollama backend
    is_ollama = (
        path.startswith("v1/chat/completions") or
        path.startswith("v1/models") or
        path.startswith("api/chat") or
        path.startswith("api/generate") or
        path.startswith("api/tags") or
        path.startswith("api/show")
    )
    
    if is_ollama:
        if path.startswith("v1/chat/completions"):
            return await openai_chat_completions(request)
        target_url = f"{core_service.OLLAMA_URL}/{path}"
    else:
        # Static assets and local OpenCode endpoints (including their own /api/ routes)
        target_url = f"{core_service.CODE_BASE_URL}/{path}"
    
    return await proxy_request(target_url, request, method=request.method)


# --- OpenAI Compatible Endpoints ---

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

            try:
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str.startswith("data:"):
                        raw_data = line_str[5:].strip()
                        if raw_data == "[DONE]":
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
                                    
                                reasoning_val = delta.pop("reasoning", None) or delta.pop("thinking", None)
                                if reasoning_val is not None:
                                    delta["reasoning_content"] = reasoning_val
                                    accumulated_reasoning += reasoning_val

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
        ctx = user_context.UserContext(type="omd", user_id="system", settings={}, history={}, omd_key=token)
        
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
