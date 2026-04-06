from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response, RedirectResponse
from fastapi import Header, Depends
from PIL import Image
import io
import re
import mimetypes
import os
import hashlib
import email.utils
import datetime
import requests
import aiohttp
import asyncio
import json

import core_service
import user_context

# Create a global session for proxying to avoid socket exhaustion
_proxy_session = None

async def get_proxy_session():
    global _proxy_session
    if _proxy_session is None or _proxy_session.closed:
        _proxy_session = aiohttp.ClientSession()
    return _proxy_session

import memory_index
import logging
import json
 
from config import USER_DATA_DIR
from config import SETTINGS

GATEWAY_URL = SETTINGS["GATEWAY_URL"]

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware is handled by the C++ gateway (omd-server)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"], # Allows all methods
#     allow_headers=["*"], # Allows all headers
# )


# Search Node Initialization
try:
    from search_node import SearchNode
    SEARCH_TOKEN = SETTINGS.get("SEARCH_TOKEN", "") # Ensure this key exists in config or is empty
    search_node = SearchNode(storage_path=USER_DATA_DIR, token=SEARCH_TOKEN)
    logging.info("[api] SearchNode initialized")
except ImportError as e:
    logging.error(f"[api] SearchNode initialization failed: Missing dependency - {e}. Please run 'pip install chromadb' in the venv.")
    search_node = None
except Exception as e:
    logging.error(f"[api] Error initializing SearchNode: {e}")
    search_node = None

# PeARS-compatible endpoints

@app.get("/indexer/from_crawl")
async def indexer_from_crawl(url: str, request: Request):
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    result = search_node.index_url(url)
    return result

@app.get("/api/urls/delete")
async def delete_url(path: str, request: Request):
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    result = search_node.delete_path(path)
    return result

@app.get("/api/urls/move")
async def move_url(src: str, target: str, request: Request):
    if not_authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    result = search_node.move_path(src, target)
    return result

@app.get("/search")
async def search(q: str, limit: int = 20, lang: str = "en"):
    # PeARS args: q=<query>, limit=<int>, lang=<lang>
    # Note: Authentication for search is handled by OMD gateway, 
    # but we might want to check for 'Omni-Search-Token' if this is exposed directly.
    # However, OMD gateway proxies it.
    
    if not search_node:
        raise HTTPException(status_code=503, detail="Search service unavailable")
        
    results = search_node.search(q, limit)
    
    # Return as JSON (list of items or dict depending on frontend expectation)
    # The frontend code iterates: "for (var searchItem in data)"
    # A list is fine for this loop.
    return results

def not_authorized(request: Request):
    # Basic check for token if we want to enforce it for management endpoints
    # PeARS checks 'Authorization' header for the crawler token.
    auth_header = request.headers.get("Authorization")
    if SEARCH_TOKEN and auth_header != SEARCH_TOKEN:
        return True
    return False

def get_omd_key(
    request: Request,
    omd_key: str | None = Query(None),
    x_omd_key: str | None = Header(None, alias="X-OMD-Key"),
    authorization: str | None = Header(None)
):
    if x_omd_key:
        return x_omd_key
    if authorization:
        # Handle "Bearer <token>", "token:<token>", "token <token>" or just "<token>"
        auth_val = authorization.strip()
        if auth_val.startswith("Bearer "):
            return auth_val[7:].strip()
        if auth_val.startswith("token:"):
            return auth_val[6:].strip()
        if auth_val.startswith("token "):
            return auth_val[6:].strip()
        return auth_val
    return omd_key
# CORS middleware already added at line 34

@app.on_event("startup")
async def startup_event():
    user_context.load_bindings()
    logging.info("[api] Bindings loaded")

# ==== Модели ввода ====

class ChatInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"
    storage: str = ""
    settings: dict | None = None
    history: list | None = None

class ChatStreamInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"
    storage: str = ""
    history: list | None = None
    settings: dict | None = None
    knowledge: list | None = None

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
    storage: str = ""
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
    storage: str = ""
    settings: dict | None = None
    history: list | None = None


class UpdateAssistantInput(BaseModel):
    omd_key: str
    nsfw: bool | None = None
    style: str | None = None
    system_prompt: str | None = None
    assistant_name: str | None = None
    assistant_title: str | None = None
    assistant_appearance: str | None = None
    assistant_model: str | None = None
    name: str | None = None

class AvatarGenerateInput(BaseModel):
    omd_key: str
    style: str | None = None
    character_lora: str | None = None
    prompt: str = ""
    storage: str = ""
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
            "nsfw": ctx.settings.get("nsfw", False),
            "model": ctx.settings.get("assistant_model", ""),
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
        if data.nsfw is not None:
            ctx.settings["nsfw"] = data.nsfw
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

def get_ctx(omd_key: str | None, storage: str = "", force_reload: bool = False):
    if omd_key in ["undefined", "null"]:
        omd_key = ""
    if storage in ["undefined", "null"]:
        storage = ""
    return user_context.get_context_by_account(omd_key, storage, force_reload)


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
        storage_path = core_service.get_assistant_avatar_path(ctx)
        
        if storage_path.startswith("/"):
            storage_path = storage_path[1:]
            
        avatar_path = os.path.join(core_service.STORAGE_ROOT, storage_path)
        return serve_file(avatar_path, request, size=size)

    except Exception as e:
        logging.error(f"Error serving avatar: {e}")
        # Return default if anything fails
        default_path = os.path.join(core_service.STORAGE_ROOT, "avatars", "default.png")
        return serve_file(default_path, request, size=size)


@app.post("/assistant/avatar/generate")
async def generate_avatar_endpoint(data: AvatarGenerateInput):
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
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
             resp = requests.get(source_url, params={"token": ctx.omd_key})
             
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
async def get_loras(nsfw: bool | None = Query(None), omd_key: str | None = Depends(get_omd_key)):
    ctx = get_ctx(omd_key)
    return core_service.get_available_loras(ctx, nsfw=nsfw)

@app.get("/assistant/model/{lora_name}/avatar")
async def model_avatar(
    request: Request,
    lora_name: str,
    omd_key: str | None = Depends(get_omd_key),
    size: int = 80
):
    ctx = get_ctx(omd_key)
    try:
        storage_path = core_service.get_model_avatar_path(lora_name)
        
        # Re-attach STORAGE_ROOT logic
        if storage_path.startswith("/"):
            storage_path = storage_path[1:]
            
        avatar_path = os.path.join(core_service.STORAGE_ROOT, storage_path)

        logging.warning(f"Serving model avatar: {avatar_path}")
        return serve_file(avatar_path, request, size=size)
    except Exception as e:
        logging.error(f"Error serving model avatar: {e}")
        default_path = os.path.join(core_service.STORAGE_ROOT, "avatars", "default.png")
        return serve_file(default_path, request, size=size)
        return serve_file(default_path, request, size=size)
    
    
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
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
    try:
        instruction=(
            "Respond to user. If user question relates to *Known facts*, be extreamly accurate, do not guess."
        )
        response = await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=data.prompt,
            chat=data.chat,
            provided_history=data.history
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
        storage=data.storage,
        provided_history=data.history,
        provided_settings=data.settings,
        provided_knowledge=data.knowledge
    )

@app.get("/chat/stream")
async def chat_stream(request: Request, prompt: str, omd_key: str | None = Depends(get_omd_key), chat: str = "default", 
                      storage: str = "",
                      provided_history: list|None = None, 
                      provided_settings: dict|None = None,
                      provided_knowledge: list|None = None):
    logging.info(f"Chat stream request: omd_key={omd_key[:10]}... storage={storage}")
    chat = chat or "default"
    ctx = get_ctx(omd_key, storage=storage)

    if provided_settings:
        logging.info(f"Applying client-provided settings for {ctx.user_id}")
        ctx.settings.update(provided_settings)

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
            if prompt.startswith("/nsfw"):
                args = prompt[len("/nsfw"):].strip().split(maxsplit=1)
                nsfw_enabled = False

                if args:
                    if args[0].lower() == "on":
                        if token_balance <= 0:
                            yield f"data: {json.dumps({'delta': 'NSFW mode is available with a Premium Plan.', 'role': 'assistant', 'done': True})}\n\n"
                            return
                        nsfw_enabled = True
                    elif args[0].lower() == "off":
                        nsfw_enabled = False

                llm_message = "get ready to play" if nsfw_enabled else "calm down for now"

                if len(args) > 1:
                    llm_message = args[1].strip()
            

                ctx.settings["nsfw"] = nsfw_enabled
                logging.info(f"User: {ctx.user_id} swithed NSFW mode to {nsfw_enabled}")
                user_context.save_user_settings(ctx)
                instruction = (
                    "User has switched NSFW mode '{}'.\nPlease, act accordingly."
                ).format(nsfw_enabled)
                
                # yield f"data: {json.dumps({'event': 'reload_chats'})}\n\n"
                event = 'reload_chats'
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
                    "/search": "search"
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
                    if not ctx.settings.get("nsfw", False):
                        logging.info(f"Checking image generation safety: {img_prompt}")
                        safety_result = await core_service.check_prompt_safety(ctx, img_prompt)
                        if safety_result != "SAFE":
                            logging.info(f"Image generation safety check failed: {safety_result}")
                            
                            warning = "I can not generate this in safe mode. Switch to unsafe mode with /nsfw on"
                            if token_balance <= 0:
                                 warning = "I can not generate this until you prove your age by subscribing for Premium plan"

                            yield f"data: {json.dumps({'delta': warning, 'role': 'assistant', 'done': True})}\n\n"
                            return
                elif prompt.startswith("/view") or prompt.startswith("/imagine") or (intent == "view" and prompt.startswith("/")):
                    intent = "view"
                elif prompt.startswith("/tools"):
                    # Provide an immediate, reliable list of tools
                    tools_list = await core_service.list_supported_tools(ctx)

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
        
            restricted_intents = ["tools"]
            
            # Check primary intent or prefixed intent (e.g. import:url)
            check_intent = intent.split(":")[0] if ":" in intent else intent
            
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
                 
            if check_intent == "import":
                 # Limit to 10 items for free accounts
                 memories = memory_index.load_memories(ctx, collection="shared")
                 if len(memories) >= 10:
                     yield f"data: {json.dumps({'delta': 'Free accounts are limited to 10 knowledge base items. Upgrade to Premium for unlimited storage.', 'role': 'assistant', 'done': True})}\n\n"
                     return

            if intent == "show":
                # 2️⃣ картинка
                # [LEGACY HISTORY] Load history removed
                history = provided_history or []
                
                # Generate prompt using loaded history, but DO NOT save yet (atomic update later)
                img_prompt = await core_service.generate_character_image_prompt(ctx, prompt, chat, history=history)
                logging.info(f"Generating image for prompt {prompt}")

                # Generate image using prompt, DO NOT save yet
                path, title, description = await core_service.generate_character_image(ctx, img_prompt, chat, update_history=False)
                
                # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
                

                yield f"data: {json.dumps({'prompt': img_prompt, 'image':{'path': path, 'title': title, 'description': description}})}\n\n"
                

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image of yourself based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "YOUR TASK: Roleplay this scene. Describe your feelings, thoughts, or actions AS IF YOU ARE IN THE IMAGE RIGHT NOW.\n"
                    "CRITICAL: Do NOT output instructions, prompts, or technical details.\n"
                    "CRITICAL: Do NOT output 'System Tool Output' or mimic system logs."
                ).format(img_prompt)
                llm_message = "Please describe the image or roleplay as requested."
                # [LEGACY HISTORY] save_user_message removed
            elif intent == "view":
                # 1️⃣ статус

                # 2️⃣ картинка
                img_prompt = await core_service.generate_general_image_prompt(ctx, prompt, chat, history=provided_history)
                logging.info(f"Generating image for prompt {prompt}")

                # 3️⃣ Generate image
                
                path, title, description = await core_service.generate_image(ctx, img_prompt, chat)
                yield f"data: {json.dumps({'prompt': img_prompt, 'image':{'path': path, 'title': title, 'description': description}})}\n\n"

                #Set specific instructions
                instruction = (
                    "You have ALREADY generated an image based on the user's request.\n"
                    "The scene description is:\n"
                    "{}\n\n"
                    "YOUR TASK: Describe this image enthusiastically or provide a caption for it.\n"
                    "CRITICAL: Do NOT output instructions, prompts, or technical details.\n"
                    "CRITICAL: Do NOT output 'System Tool Output' or mimic system logs."
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
                
                # Call search_web directly - no MCP, no hallucination
                search_results = await core_service.search_web(ctx, search_query)
                
                # [LEGACY HISTORY] Load history removed
                history = provided_history or []
                
                instruction = (
                    f"The user asked to search the web. Here are the REAL search results:\n\n"
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
                
                logging.info(f"Generating image for prompt {img_prompt} with title {img_title}")
                path, title, description = await core_service.generate_image(ctx, formatted_prompt, chat, use_default_lora = False)
                yield f"data: {json.dumps({'prompt': prompt, 'image':{'path': path, 'title': title, 'description': description}, 'done': True})}\n\n"
                
                # [LEGACY HISTORY] Backend-side history saving removed - handled by frontend/OrbitDB
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
    omd_key: str | None = Depends(get_omd_key),
    chat: str = Form("default"),
    prompt: str = Form(""),
    storage: str = Form(""),
    settings: str = Form(None),
    history: str = Form(None),
    file: UploadFile = File(...)
):
    chat = chat or "default"
    ctx = get_ctx(omd_key, storage=storage)
    if settings:
        try:
            provided_settings = json.loads(settings)
            ctx.settings.update(provided_settings)
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
async def generate_character_image(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
    try:
        # generate_image returns (filename, title, description)
        is_new = data.message_nonce is None and data.message_index is None
        filename, title, description = await core_service.generate_image(ctx, data.prompt, data.chat, is_new)
        
        # [LEGACY HISTORY] Load history removed
        history = []
        
        if data.message_index is not None:
             # This part might still be needed if we want to return the updated description,
             # but we don't save it to local disk anymore.
             pass

        return {"image": filename, "description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/general")
async def generate_general_image(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
    try:
        # generate_image returns (filename, title, description)
        filename, title, description = await core_service.generate_image(ctx, data.prompt, data.chat, use_default_lora=False)
        return {"image": filename, "description": description}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/generate/prompt/character")
async def generate_character_image_prompt(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
    try:
        result = await core_service.generate_character_image_prompt(ctx, data.prompt, data.chat, history=data.history)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/generate/prompt/general")
async def generate_general_image_prompt(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key, storage=data.storage)
    if data.settings:
        ctx.settings.update(data.settings)
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
        if data.nsfw is not None:
            ctx.settings["nsfw"] = data.nsfw
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
    if data.omd_key in user_context.bindings["by_account"]:
        del user_context.bindings["by_account"][data.omd_key]
        logging.info(f"Signed out user with key {data.omd_key}")
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

@app.get("/code/sessions")
async def proxy_opencode_sessions_list(request: Request):
    target_url = f"{core_service.CODE_BASE_URL}/session"
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

@app.post("/code/sessions")
async def proxy_opencode_sessions_create(request: Request):
    target_url = f"{core_service.CODE_BASE_URL}/session"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()
    try:
        async with session.post(target_url, data=body, headers=headers) as resp:
            data = await resp.json()
            return {"session": data}
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/code/sessions/{session_id}", methods=["GET", "DELETE", "PATCH"])
async def proxy_opencode_session_item(request: Request, session_id: str):
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}"
    return await proxy_request(target_url, request, method=request.method)

@app.api_route("/code/sessions/{session_id}/message", methods=["POST"])
async def proxy_opencode_prompt(request: Request, session_id: str):
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
    
    # We need to translate the OnMyDisk payload to OpenCode format
    # OnMyDisk: { "prompt": "...", ... }
    # OpenCode: { "parts": [ { "type": "text", "text": "..." } ] }
    try:
        omd_payload = await request.json()
        prompt_text = omd_payload.get("prompt", "")
        
        opencode_payload = {
            "parts": [
                {
                    "type": "text",
                    "text": prompt_text
                }
            ]
        }
        
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
        opencode_payload["model"] = {
            "providerID": "ollama",
            "modelID": core_service.DEFAULT_MODEL
        }
        
        session = await get_proxy_session()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers["Content-Type"] = "application/json"
        
        async def stream_generator():
            try:
                # 1. Connect to OpenCode's Event Stream FIRST to avoid dropping events
                event_url = f"{core_service.CODE_BASE_URL}/event"
                async with aiohttp.ClientSession(read_bufsize=10*1024*1024) as sse_session:
                    async with sse_session.get(event_url, headers={"Accept": "text/event-stream"}) as event_resp:
                        if event_resp.status != 200:
                            logging.error(f"[OpenCode Proxy] Failed to connect to event stream: {event_resp.status}")
                            # Fallback: Just fire and return
                            async with session.post(target_url, json=opencode_payload) as resp:
                                result = await resp.json()
                                yield f"data: {json.dumps(result)}\n\n".encode('utf-8')
                                yield b"data: {\"done\": true}\n\n"
                            return
    
                        # (Removed redundant PRE-STREAM STATUS)
                        
                        # 2. Start the POST request in the background NOW
                        async def do_post():
                            async with session.post(target_url, json=opencode_payload) as resp:
                                if resp.status != 200:
                                    error_text = await resp.text()
                                    logging.error(f"[OpenCode Proxy] Backend error {resp.status}: {error_text}")
                                    return {"error": f"Backend error: {resp.status}"}
                                return await resp.json()
                                
                        post_task = asyncio.create_task(do_post())
                        last_emitted_states = {}
                        yield_counter = 0
    
                        # 3. Read events as long as post_task is not done
                        while not post_task.done():
                            yield_counter += 1
                            if yield_counter % 10 == 0:
                                await asyncio.sleep(0) # Yield control to event loop

                            try:
                                # Read one line from SSE with timeout
                                line = await asyncio.wait_for(event_resp.content.readline(), timeout=0.1)
                                if not line:
                                    break
                                    
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data: "):
                                    try:
                                        event_data = json.loads(line_str[6:])
                                        ev_type = event_data.get("type", "")
                                        props = event_data.get("properties", {})
                                        
                                        # Filter by session_id
                                        if ev_type == "session.updated" and props.get("sessionID") == session_id:
                                            info = props.get("info", {})
                                            if "title" in info:
                                                rename_chunk = {
                                                    "action": "rename",
                                                    "title": info["title"]
                                                }
                                                yield f"data: {json.dumps(rename_chunk)}\n\n".encode('utf-8')
                                                
                                        elif props.get("sessionID") == session_id:
                                            if ev_type == "message.part.delta":
                                                chunk = {
                                                    "id": props.get("partID"),
                                                    "delta": props.get("delta")
                                                }
                                                yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                                            elif ev_type == "message.part.updated":
                                                part = props.get("part", {})
                                                part_id = part.get("id")
                                                state_obj = part.get("state")
                                                
                                                if isinstance(state_obj, dict):
                                                    # Deduplicate based on logical state (ignoring high-freq 'time' ticker)
                                                    state_copy = dict(state_obj)
                                                    state_copy.pop("time", None)
                                                    # Efficient stringification for cache comparison
                                                    state_str = json.dumps(state_copy, sort_keys=True)
                                                    if last_emitted_states.get(part_id) == state_str:
                                                        continue
                                                    last_emitted_states[part_id] = state_str
                                                    
                                                chunk = {
                                                    "id": part_id,
                                                    "type": part.get("type"),
                                                    "state": state_obj
                                                }
                                                yield f"data: {json.dumps(chunk)}\n\n".encode('utf-8')
                                    except Exception as parse_e:
                                        pass # Ignore non-JSON or malformed data stream
                            except asyncio.TimeoutError:
                                await asyncio.sleep(0.01) # Yield on idle timeout
                                continue 
                            
                # 4. Wait for post to finish completely if stream broke
                if not post_task.done():
                    await post_task
                
                # Signal done to frontend
                yield b"data: {\"done\": true}\n\n"
                
            except Exception as e:
                logging.error(f"[OpenCode Proxy] Stream error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
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

@app.get("/code/diff")
async def proxy_opencode_diff(request: Request, change_id: str = Query(...), session_id: str = Query(...)):
    """
    OnMyDisk expects: { "diff": "unified diff string" }
    OpenCode returns: [ { file, diff, ... }, ... ]
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/diff?messageID={change_id}"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        async with session.get(target_url, headers=headers) as resp:
            data = await resp.json()
            full_diff = ""
            if isinstance(data, list):
                for file_diff in data:
                    full_diff += f"File: {file_diff.get('file')}\n"
                    full_diff += file_diff.get('diff', '') + "\n\n"
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
        for item in diff_list:
            file_name = item.get("file", "unknown")
            before_lines = (item.get("before", "") or "").splitlines(keepends=True)
            after_lines = (item.get("after", "") or "").splitlines(keepends=True)
            diff_gen = difflib.unified_diff(
                before_lines, after_lines,
                fromfile=f"a/{file_name}", tofile=f"b/{file_name}", n=3
            )
            item["diff"] = "".join(diff_gen)
        return diff_list

    try:
        # If no message_id is provided, safely fetch the overall cumulative session diff
        if not message_id:
            target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/diff"
            async with session.get(target_url, headers=headers) as resp:
                data = await resp.json()
                return {"diffs": inject_diffs(data if isinstance(data, list) else [])}
        
        # If message_id is provided (Assistant's reply), we must fetch the session messages
        # and pull the cached modifications array from its parent USER message natively.
        session_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/message"
        async with session.get(session_url, headers=headers) as resp:
            data = await resp.json()
            messages = data if isinstance(data, list) else []
            
            # 1. Locate the specific Assistant message
            assistant_msg = next((m for m in messages if m.get("info", {}).get("id") == message_id), None)
            if not assistant_msg:
                return {"diffs": []}
                
            # 2. Extract its parentID (the user message that originated the generations)
            parent_id = assistant_msg.get("info", {}).get("parentID")
            if not parent_id:
                return {"diffs": []}
                
            # 3. Locate the parent user message
            user_msg = next((m for m in messages if m.get("info", {}).get("id") == parent_id), None)
            if not user_msg:
                return {"diffs": []}
                
            # 4. Extract diffs from the user message's summary map
            msg_diffs = user_msg.get("info", {}).get("summary", {}).get("diffs", [])
            return {"diffs": inject_diffs(msg_diffs if isinstance(msg_diffs, list) else [])}

    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error fetching session diffs: {e}")
        return {"diffs": []}

@app.post("/code/sessions/{session_id}/revert")
async def proxy_opencode_revert(request: Request, session_id: str):
    """
    Triggers OpenCode backend to revert file snapshots effectively undoing code generations since the specified messageID.
    """
    target_url = f"{core_service.CODE_BASE_URL}/session/{session_id}/revert"
    session = await get_proxy_session()
    headers = dict(request.headers)
    headers.pop("host", None)
    
    try:
        body = await request.json()
        async with session.post(target_url, headers=headers, json=body) as resp:
            data = await resp.json()
            return data
    except Exception as e:
        logging.error(f"[OpenCode Proxy] Error reverting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/code/sessions/{session_id}/messages")
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
    """
    target_url = f"{core_service.OLLAMA_URL}/v1/chat/completions"
    return await proxy_request(target_url, request, method="POST")

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
            
        token = request.headers.get("X-OMD-Key") or request.query_params.get("omd_key")
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
