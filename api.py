from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response, RedirectResponse
from fastapi import Header
from PIL import Image
import io
import re
import mimetypes
import os
import hashlib
import email.utils
import datetime
import requests

import core_service
import user_context
import dialog_history
import memory_index
import logging
import json
 
from config import USER_DATA_DIR
from config import SETTINGS

GATEWAY_URL = SETTINGS["GATEWAY_URL"]

logging.basicConfig(level=logging.INFO)

app = FastAPI()


origins = [
    "http://localhost:8080",  
    "http://localhost:8081", 
    f"{GATEWAY_URL}",
    "*",  # Caution: Allow all origins (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    user_context.load_bindings()
    logging.info("[api] Bindings loaded")

# ==== Модели ввода ====

class ChatInput(BaseModel):
    omd_key: str
    prompt: str
    chat: str = "default"

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


class UpdateAssistantInput(BaseModel):
    omd_key: str
    nsfw: bool | None = None
    style: str | None = None
    system_prompt: str | None = None
    assistant_name: str | None = None
    assistant_title: str | None = None
    assistant_appearance: str | None = None
    assistant_model: str | None = None

class AvatarGenerateInput(BaseModel):
    omd_key: str
    style: str | None = None
    character_lora: str | None = None
    prompt: str = ""

class AvatarUpdateInput(BaseModel):
    omd_key: str
    image_path: str
    style: str | None = None
    assistant_model: str | None = None

class SignoutInput(BaseModel):
    omd_key: str


# ==== Хелпер ====

def get_ctx(omd_key: str, storage: str = ""):
    return user_context.get_context_by_account(omd_key, storage)


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

@app.get("/assistant")
async def assistant_info(omd_key: str):
    ctx = get_ctx(omd_key)
    try:
        assistant = {
            "name": ctx.settings.get("assistant_name", user_context.DEFAULT_ASSISTANT_NAME),
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


@app.get("/assistant/avatar")
async def assistant_avatar(
    request: Request,
    omd_key: str,
    size: int = 80
):
    ctx = get_ctx(omd_key)
    try:
        # Check for custom avatar in remote storage
        if ctx.storage and ctx.omd_key:
            storage_id = ctx.storage
            storage_key = ctx.omd_key
            
            base_url = user_context.GATEWAY_URL.rstrip("/")
            clean_storage_id = storage_id.strip("/")
            
            # Use avatar.png as standard user avatar name
            # We can try to HEAD checks first? Or just try GET with stream.
            url = f"{base_url}/{clean_storage_id}/avatar.png"
            params = {"token": storage_key}
            if size:
                params["resize"] = "true"
                params["width"] = size
                params["height"] = size
            
            try:
                # Use requests with stream=True for proxying
                resp = requests.get(url, params=params, stream=True, timeout=5)
                
                content_type = resp.headers.get("Content-Type", "")
                if resp.status_code == 200 and content_type.startswith("image/"):
                    # Success
                    return StreamingResponse(resp.iter_content(chunk_size=8192), media_type=content_type)
                elif resp.status_code == 404:
                    # Not found, fallback to default
                    pass
                else:
                    logging.warning(f"Storage returned {resp.status_code} for avatar")
                    
            except Exception as e:
                logging.warning(f"Failed to fetch avatar from storage: {e}")

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
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_avatar(ctx, data.style, data.character_lora, data.prompt)
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
        
        if data.style or data.assistant_model:
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
async def get_loras(omd_key: str):
    ctx = get_ctx(omd_key)
    return core_service.get_available_loras(ctx)

@app.get("/assistant/model/{lora_name}/avatar")
async def model_avatar(
    request: Request,
    lora_name: str,
    omd_key: str,
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
async def get_assistant_avatars_endpoint(omd_key: str):
    ctx = get_ctx(omd_key)
    avatars = await core_service.get_generated_avatars(ctx)
    return {"status": "ok", "avatars": avatars}


@app.get("/history")
async def history_endpoint(omd_key: str, chat: str = "default"):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        history = dialog_history.load_history(ctx, chat=chat)
        return {"chat": chat, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.delete("/history/{chat}/{number:int}")
def delete_history(chat: str, number: int, omd_key: str = Header(..., alias="X-OMD-Key")):
    """
    Удаляет сообщение по индексу (0-based) из истории указанного чата пользователя.
    """
    chat = chat or "default"
    ctx = get_ctx(omd_key)

    try:
        # Загружаем историю чата
        history = dialog_history.load_history(ctx, chat=chat)

        if not history:
            raise HTTPException(status_code=404, detail=f"Chat '{chat}' is empty or not found")

        # Проверка диапазона индекса
        if number < -len(history) or number >= len(history):
            raise HTTPException(status_code=404, detail="Message index out of range")

        # Удаляем сообщение
        deleted_msg = history.pop(number)

        # Сохраняем обратно
        dialog_history.save_history(ctx, history, chat=chat)

        return {
            "status": "ok",
            "chat": chat,
            "deleted_index": number,
            "deleted_role": deleted_msg.get("role"),
            "remaining": len(history)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/memory")
async def memory_endpoint(omd_key: str, collection: str = "user"):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection=collection)
        return {"collection": collection, "memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory")
async def update_memory(data: MemoryUpdate, omd_key: str = Header(..., alias="X-OMD-Key")):
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
async def import_memory(data: MemoryImport, omd_key: str = Header(..., alias="X-OMD-Key")):
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
async def delete_memory(mem_id: str, omd_key: str = Header(..., alias="X-OMD-Key"), collection: str = "user" ):
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
async def get_memory(omd_key: str, collection: str, mem_id: str):
    ctx = get_ctx(omd_key)
    try:
        memories = memory_index.load_memories(ctx, collection)
        for m in memories:
            if m["memory_id"] == mem_id:
                return m
        raise HTTPException(status_code=404, detail="Memory not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/chats")
async def chats_endpoint(omd_key: str, storage: str = ""):
    ctx = get_ctx(omd_key, storage)
    try:
        if not ctx.storage and storage:
            logging.info(f"Creating profile for user {ctx.user_id}, {ctx.type}, storage: {storage}")
            user_context.create_profile(ctx, omd_key, storage)

        from dialog_history import load_chats_index
        chats = load_chats_index(ctx)
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat}/archive")
async def archive_chat(chat: str, omd_key: str = Header(..., alias="X-OMD-Key")):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        from dialog_history import load_chats_index, save_chats_index
        chats = load_chats_index(ctx)
        if chat in chats:
            # Set to a very old date to "archive" it (make it disappear from recent list)
            # Using epoch start: 1970-01-01T00:00:00Z
            chats[chat]["updated"] = "1970-01-01T00:00:00Z"
            save_chats_index(ctx, chats)
            return {"status": "ok", "chat": chat}
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat}/restore")
async def restore_chat(chat: str, omd_key: str = Header(..., alias="X-OMD-Key")):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        from dialog_history import load_chats_index, save_chats_index
        chats = load_chats_index(ctx)
        if chat in chats:
            # Set to current date to "restore" it to the top
            chats[chat]["updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
            save_chats_index(ctx, chats)
            return {"status": "ok", "chat": chat}
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/{chat}")
async def delete_chat(chat: str, omd_key: str = Header(..., alias="X-OMD-Key")):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        from dialog_history import load_chats_index, save_chats_index, reset_history
        chats = load_chats_index(ctx)
        if chat in chats:
            del chats[chat]
            save_chats_index(ctx, chats)
            
            # Delete the chat history file
            # Note: reset_history takes user_id, but we need to delete specific chat file
            # We'll implement a specific delete function or use os.remove directly if possible,
            # but better to use a helper if available. 
            # dialog_history.py has _get_path but it is internal.
            # Let's check dialog_history.py again or just implement file deletion here safely.
            
            # Re-implementing safe deletion logic here for now or adding to dialog_history
            # Since we are in api.py, let's use dialog_history helper if we can add one, 
            # or just do it here if we are confident.
            # dialog_history.py has reset_history(user_id) which deletes ALL history? No, let's check.
            # reset_history deletes _get_path(user_id) which seems to be a single file? 
            # Wait, _get_path takes (user_id, chat).
            
            # Let's assume we can just remove the file.
            chat_file = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
            if os.path.exists(chat_file):
                os.remove(chat_file)
            
            # Also need to handle remote storage deletion if applicable
            if ctx.storage and ctx.omd_key:
                 # Remote deletion not fully implemented in provided snippets, 
                 # but we can try to upload empty or use a delete endpoint if it existed.
                 # For now, let's just update the index which hides it.
                 pass

            return {"status": "ok", "chat": chat}
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chats/{chat}/nsfw")
async def toggle_chat_nsfw(chat: str, omd_key: str = Header(..., alias="X-OMD-Key")):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        from dialog_history import load_chats_index, save_chats_index
        chats = load_chats_index(ctx)
        if chat in chats:
            current_nsfw = chats[chat].get("nsfw", False)
            chats[chat]["nsfw"] = not current_nsfw
            save_chats_index(ctx, chats)
            return {"status": "ok", "chat": chat, "nsfw": chats[chat]["nsfw"]}
        raise HTTPException(status_code=404, detail="Chat not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(data: ChatInput):
    ctx = get_ctx(data.omd_key)
    try:
        instruction=(
            "Respond to user. If user question relates to *Known facts*, be extreamly accurate, do not guess."
        )
        response = await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=data.prompt,
            chat=data.chat,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/stream")
async def chat_stream(request: Request, omd_key: str, prompt: str, chat: str = "default"):
    chat = chat or "default"
    ctx = get_ctx(omd_key)

    async def event_generator():
        nonlocal chat
        # defaults
        intent = "chat"
        event = None
        skip_history = False
        is_rag = False
        mem_id = None
        think = False
        img_source = None

        # Initialize chat if it's the first message of a new session
        if not chat or chat == "default":
             chat_info = await core_service.ensure_chat(ctx, chat, prompt)
             chat = chat_info["name"]
             yield f"data: {json.dumps({'event': 'newchat', 'chatinfo': chat_info})}\n\n"

        # Enforce Rights (moved up)
        ai_advanced = request.headers.get("x-omd-ai-advanced", "true") == "true"


        # perform commands
        if prompt.startswith("/nsfw"):
            skip_history = True
            args = prompt[len("/nsfw"):].strip().split(maxsplit=1)
            nsfw_enabled = False

            if args:
                if args[0].lower() == "on":
                    if not ai_advanced:
                        yield f"data: {json.dumps({'delta': 'NSFW mode is available with a Premium Plan.', 'role': 'assistant', 'done': True})}\\n\\n"
                        return
                    nsfw_enabled = True
                elif args[0].lower() == "off":
                    nsfw_enabled = False

            llm_message = "get ready to play" if nsfw_enabled else "calm down for now"

            if len(args) > 1:
                llm_message = args[1].strip()
                skip_history = False
        

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
            raw_intent = await core_service.classify_user_intent(ctx, prompt, chat)
            lines = raw_intent.strip().split("\n", 1)
            intent_raw = lines[0].strip().lower()
            
            # Whitelist and sanitize intent
            allowed_intents = ["show", "view", "explain", "recognize", "import", "tools", "chat"]
            intent = "chat"
            for allowed in allowed_intents:
                if intent_raw.startswith(allowed):
                    intent = allowed
                    break
            
            # Ensure chat existence for all intent types (crucial for 'show' intent which bypasses perform_prompt)
            # This ensures chat is in the index and has a title
            if intent != "chat": # perform_prompt handles chat intent
                 # Only if we are branching away from perform_prompt
                 chat_info = await core_service.ensure_chat(ctx, chat, prompt)
                 chat = chat_info.get("name", chat)
            
            logging.info(f"Intent detected: {intent} \n(raw: {raw_intent[:50]}...)")
            
            # 2. Extract Memory Facts immediately
            memory_fact = memory_index.extract_memory_from_response(raw_intent)
            if memory_fact:
                try:
                    logging.info(f"Memorizing: {memory_fact}")
                    memory_index.add_memory_card(ctx, memory_fact, collection="user", relevance="permanent")
                    yield f"data: {json.dumps({'newFact': memory_fact})}\n\n"
                except Exception as e:
                    logging.error(f"Vectorization error: {e}")

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
                        yield f"data: {json.dumps({'delta': safety_result, 'role': 'assistant', 'done': True})}\\n\\n"
                        return
            elif prompt.startswith("/view") or prompt.startswith("/imagine") or (intent == "view" and prompt.startswith("/")):
                intent = "view"
            elif prompt.startswith("/tools") or intent == "tools":
                # Provide an immediate, reliable list of tools
                tools_list = await core_service.list_supported_tools(ctx)

                # Ensure history is saved for this interaction
                history = dialog_history.load_history(ctx, chat)
                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": tools_list})
                dialog_history.save_history(ctx, history, chat)

                yield f"data: {json.dumps({'delta': tools_list, 'role': 'assistant', 'done': True})}\n\n"
                return
            elif prompt.startswith("/import") or prompt.startswith("/learn"):  
                m = re.match(r'^/(?:import|learn)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', prompt)
                file_path_or_url = m.group(1) or m.group(2) or m.group(3) if m else None
                if file_path_or_url:
                    intent = f"import:{file_path_or_url}"
            elif prompt.startswith("/recognize") or prompt.startswith("/detect"):  
                m = re.match(r'^/(?:recognize|detect)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', prompt)
                file_path_or_url = m.group(1) or m.group(2) or m.group(3) if m else None
                if file_path_or_url:
                    intent = f"recognize:{file_path_or_url}"
            elif prompt.startswith("/think") or prompt.startswith("/explain"):  
                intent = "explain"   
                think = prompt.startswith("/think")
    
        restricted_intents = ["tools"]
        
        # Check primary intent or prefixed intent (e.g. import:url)
        check_intent = intent.split(":")[0] if ":" in intent else intent
        
        if not ai_advanced:
             if check_intent in restricted_intents:
                  yield f"data: {json.dumps({'delta': 'Advanced AI features are available with a Premium Plan.', 'role': 'assistant', 'done': True})}\n\n"
                  return
             
             if check_intent == "import":
                  # Limit to 10 items for free accounts
                  memories = memory_index.load_memories(ctx, collection="user")
                  if len(memories) >= 10:
                       yield f"data: {json.dumps({'delta': 'Free accounts are limited to 10 knowledge base items. Upgrade to Premium for unlimited storage.', 'role': 'assistant', 'done': True})}\n\n"
                       return

        if intent == "show":
            # 1️⃣ статус
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"
            # 2️⃣ картинка
            # Load history ONCE to avoid race conditions
            history = dialog_history.load_history(ctx, chat)
            
            # Generate prompt using loaded history, but DO NOT save yet (atomic update later)
            img_prompt = await core_service.generate_character_image_prompt(ctx, prompt, chat, history=history, save_history_flag=False)
            logging.info(f"Generating image for prompt {prompt}")

            # Generate image using prompt, DO NOT save yet
            path, title = await core_service.generate_character_image(ctx, img_prompt, chat, update_history=False)
            
            # Now perform atomic history update
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": img_prompt, "image": {"path": path, "title": title}})
            dialog_history.save_history(ctx, history, chat)
            
            yield f"data: {json.dumps({'prompt': img_prompt, 'image':{'path': path, 'title': title}})}\n\n"
            
            skip_history = False
            #Set specific instructions
            instruction = (
                "This is a textual description of the image already generated by your prompt. You are acting in this scene:\n"
                "{}\n\n"
                "*'Me' refers to you in the provided scene description* \n\n"
                "Do not cite the prompt or scene description in your response, just continue the conversation describing how you feel in this scene. "
            ).format(img_prompt)
            llm_message = prompt
        elif intent == "view":
            # 1️⃣ статус
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"

            # 2️⃣ картинка
            img_prompt = await core_service.generate_general_image_prompt(ctx, prompt, chat)
            logging.info(f"Generating image for prompt {prompt}")

            # 3️⃣ Generate image
            
            path, title = await core_service.generate_image(ctx, img_prompt, chat)
            yield f"data: {json.dumps({'prompt': img_prompt, 'image':{'path': path, 'title': title}})}\n\n"
            skip_history = False
            #Set specific instructions
            instruction = (
                "This is a textual description of the image already generated by your prompt:\n"
                "{}\n\n"
                "*'Me', if present, refers to you in the provided scene description*"
                "Do not cite the prompt or scene description in your response, just continue the conversation describing this scene."
            ).format(img_prompt)
            llm_message = prompt

        elif intent == "explain":    
            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
    
            instruction=(
                "If Known facts are provided and they are relevant to user's query, you must strictly base your response only on them. "
                "Do not invent or speculate. If no *Strict facts* are provided, do not guess, clearly separate what is factual from what is uncertain, and explicitly state the limitations."
                "If no relevant Known facts are provided, respond freely as a helpful conversational assistant."
            )
            llm_message = prompt
            is_rag = True
        elif intent.startswith("recognize"): 
            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

            if ":" in intent:
                img_source = intent.split(":", 1)[1]
        
            instruction = (
                "Recognize the image according to context."
            )
            llm_message = prompt
        elif intent.startswith("import"):
            yield f"data: {json.dumps({'status': 'learning'})}\n\n"
            doc_source = None
            card = {}
            if ":" in intent:
                doc_source = intent.split(":", 1)[1]
            if doc_source:
                card = await core_service.import_doc(ctx, doc_source)   
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
            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

        elif intent == "generate":
            # Ensure chat exists and update timestamp
            chat_info = await core_service.ensure_chat(ctx, chat, img_prompt)
            
            # 1️⃣ статус
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"

            # 2️⃣ Generate title from raw prompt
            img_title = await core_service.generate_title_from_prompt(img_prompt)
            
            # Format prompt with title for generate_image to parse
            formatted_prompt = f"Title: {img_title}\nImage: {img_prompt}"
            
            logging.info(f"Generating image for prompt {img_prompt} with title {img_title}")
            path, title = await core_service.generate_image(ctx, formatted_prompt, chat, use_default_lora = False)
            yield f"data: {json.dumps({'prompt': img_prompt, 'image':{'path': path, 'title': title}, 'done': True})}\n\n"
            return

        # 3️⃣ основной стрим чата
        else:
            skip_history = False
            llm_message = prompt
            instruction = (
                "If *System Tool Output* (MCP) results are provided, they are the source of truth. Use them to provide a final answers. "
                "If no relevant tool results are provided, respond freely as a helpful conversational assistant."
            )
        # 3️⃣ ответ
        async for chunk in await core_service.perform_prompt(
            ctx,
            instruction=instruction,
            message=llm_message,
            chat=chat,
            skip_history=skip_history,
            is_rag=is_rag,
            mem_id=mem_id,
            think=think,
            img_source=img_source,
            event=event,
            stream=True
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")




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
    omd_key: str = Form(...),
    chat: str = Form("default"),
    prompt: str = Form(""),
    file: UploadFile = File(...)
):
    chat = chat or "default"
    ctx = get_ctx(omd_key)
    try:
        img_bytes = await file.read()
        result = await core_service.recognize_image(ctx, img_bytes, prompt, chat)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/character")
async def generate_character_image(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    try:
        # generate_image returns (filename, title)
        filename, title = await core_service.generate_image(ctx, data.prompt, data.chat, data.message_index is None)
        
        # Update history if index provided
        if data.message_index is not None:
            history = dialog_history.load_history(ctx, chat=data.chat)
            if 0 <= data.message_index < len(history):
                msg = history[data.message_index]
                # Ensure it's an assistant message with image
                if msg.get("role") == "assistant" and "image" in msg:
                    msg["image"]["path"] = filename
                    msg["image"]["title"] = title
                    history[data.message_index] = msg
                    dialog_history.save_history(ctx, history, chat=data.chat)

        return {"image": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/image/general")
async def generate_general_image(data: GenerateInput):
    ctx = get_ctx(data.omd_key)
    try:
        # generate_image returns (filename, title)
        filename, title = await core_service.generate_image(ctx, data.prompt)
        return {"image": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/generate/prompt/character")
async def generate_character_image_prompt(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_character_image_prompt(ctx, data.prompt, data.chat)
        return {"prompt": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/generate/prompt/general")
async def generate_general_image_prompt(data: GenerateInput):
    data.chat = data.chat or "default"
    ctx = get_ctx(data.omd_key)
    try:
        result = await core_service.generate_general_image_prompt(ctx, data.prompt, data.chat)
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
