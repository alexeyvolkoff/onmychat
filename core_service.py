import os
import random
import json
import logging

import aiohttp
import time
import subprocess

from typing import AsyncGenerator

from config import SETTINGS
from config import USER_DATA_DIR

from utils import clean_response, upload_to_storage, upload_data_to_storage, get_image_from_source

from dialog_history import load_history, save_history, load_chats_index, save_chats_index
import user_context
from user_context import UserContext
from datetime import datetime, timezone
import re
import uuid

from memory_index import (
    add_memory_card,
    fetch_document_text,
    chunk_and_vectorize_to_file,
    search_memories
)


DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]

# LLM and RAG settings #
OLLAMA_URL = SETTINGS["OLLAMA_URL"]
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
NSFW_MODEL = SETTINGS["NSFW_MODEL"]

# Imaging settings #
COMFY_API_URL = SETTINGS["COMFY_API_URL"]

WORKFLOW_PATH = SETTINGS["WORKFLOW_PATH"]
COMFY_OUTPUT_DIR = SETTINGS["COMFY_OUTPUT_DIR"]
COMFY_INPUT_DIR = SETTINGS["COMFY_INPUT_DIR"]
AVATAR_DIR = SETTINGS["AVATAR_DIR"]
STORAGE_ROOT = SETTINGS["STORAGE_ROOT"]
APP_ROOT_DIR = SETTINGS["APP_ROOT_DIR"]
HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"])
GATEWAY_URL = SETTINGS["GATEWAY_URL"]
REASONONG_SUPPORTED = False


# Prompt loading logic
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def get_prompt(filename):
    try:
        with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load prompt {filename}: {e}")
        return ""

def get_json_prompt(filename):
    try:
        with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load json prompt {filename}: {e}")
        return {}

# Default system prompts
BASE_SYSTEM_PROMPT = get_prompt("base_system.txt")
MEMORIZATION_PROMPT = get_prompt("memorization.txt")
SYSTEM_INSTRUCTION_CHARACTER = get_prompt("instruction_character.txt")
SYSTEM_INSTRUCTION_GENERAL = get_prompt("instruction_general.txt")
IMAGE_PROMPT_NSFW = get_prompt("image_nsfw.txt")
RAG_SYSTEM_PROMPT = get_prompt("rag_system.txt")
IMPROVEMENT_PROMPT = get_prompt("improvement.txt")

STYLE_MODELS = {
    "realistic": SETTINGS["REALISTIC_MODEL"],
    "realistic_nsfw": SETTINGS["REALISTIC_MODEL_NSFW"],
    "realistic2": SETTINGS.get("REALISTIC2_MODEL", SETTINGS["REALISTIC_MODEL"]),
    "realistic2_nsfw": SETTINGS.get("REALISTIC2_MODEL_NSFW", SETTINGS["REALISTIC_MODEL_NSFW"]),
    "perfect": SETTINGS["PERFECT_MODEL"],
    "perfect_nsfw": SETTINGS["PERFECT_MODEL_NSFW"],
    "fantasy": SETTINGS["FANTASY_MODEL"],
    "fantasy_nsfw": SETTINGS["FANTASY_MODEL_NSFW"],
    "tooned": SETTINGS["TOONED_MODEL"],
    "tooned_nsfw": SETTINGS["TOONED_MODEL_NSFW"],
    "pleasure": SETTINGS["PLEASURE_MODEL"],
    "pleasure_nsfw": SETTINGS["PLEASURE_MODEL_NSFW"],
}

NEGATIVE_PROMPTS = get_json_prompt("negative_prompts.json")
INTENT_PROMPT = get_prompt("intent.txt")
SAFETY_CHECK_PROMPT = get_prompt("safety_check.txt")
SUMMARY_PROMPT = get_prompt("summary.txt")
NSFW_PREPHASE = get_prompt("nsfw_prephase.txt")

# === Онбординг успешен - перенос песрональных данных ===
def bind_account(ctx: UserContext, omd_key: str):
    #check if already linked
    #renaming user data folder 
    if omd_key and not ctx.type == "omd" :
        tmp_user_id = ctx.user_id
        old_dir = os.path.join(USER_DATA_DIR, tmp_user_id)
        ctx = user_context.bind(ctx, omd_key)
        new_dir = os.path.join(USER_DATA_DIR, ctx.user_id)

        if os.path.exists(old_dir):
            # если у нового юзера ещё нет директории — просто переименуем
            if not os.path.exists(new_dir):
                os.rename(old_dir, new_dir)
            else:
                # если у нового юзера уже есть данные — можно смержить
                # пока просто оставим старое и не трогаем
                logging.warning(f"[bind_account] WARNING: {new_dir} already exists, skipping rename")

    # === Проверяем storage и переносим данные ===
    storage = ctx.storage
    omd_key = ctx.omd_key
    if ctx.type == "omd" and storage and omd_key:

        # 1. Переносим чаты
        chats_dir = os.path.join(USER_DATA_DIR, ctx.user_id, "chats")
        if os.path.exists(chats_dir):
            for file in os.listdir(chats_dir):
                if not file.endswith(".json"):
                    continue
                local_path = os.path.join(chats_dir, file)
                try:
                    dest = f"{storage}/{ctx.user_id}/chats"
                    upload_to_storage(omd_key, dest, file, local_path)
                    logging.info(f"[bind_account] Chat {file} uploaded to {dest}")
                except Exception as e:
                    logging.error(f"[bind_account] Failed to upload chat {file}: {e}")

        # 2. Персональная память
        mem_path = os.path.join(USER_DATA_DIR, ctx.user_id,  "memory.jsonl")
        if os.path.exists(mem_path):
            try:
                dest = f"{storage}/{ctx.user_id}"
                upload_to_storage(omd_key, dest,"memory.jsonl", mem_path)
                logging.info(f"[bind_account] Memory uploaded to {dest}")
            except Exception as e:
                logging.error(f"[bind_account] Failed to upload memory: {e}")

    return ctx


    

def get_assistant_avatar_path(ctx: UserContext) -> str:
    model = ctx.settings.get("assistant_model", "Domi")
    full_avatar_path = f"{COMFY_INPUT_DIR}/{AVATAR_DIR}/{model}.png"
    avatar_path = full_avatar_path.replace(STORAGE_ROOT, "")
    return avatar_path

def get_model_avatar_path(model_name: str) -> str:
    full_avatar_path = f"{COMFY_INPUT_DIR}/{AVATAR_DIR}/{model_name}.png"
    avatar_path = full_avatar_path.replace(STORAGE_ROOT, "")
    return avatar_path


def get_available_loras(ctx: UserContext = None) -> list:
    lora_file = os.path.join(os.path.dirname(__file__), "analog_character_lora.json")
    if os.path.exists(lora_file):
        try:
            with open(lora_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Find Power Lora Loader node (ID 103 based on file inspection)
            # Or search for class_type "Power Lora Loader (rgthree)"
            loras = []
            for key, node in data.items():
                if node.get("class_type") == "Power Lora Loader (rgthree)":
                    inputs = node.get("inputs", {})
                    for input_key, input_val in inputs.items():
                        if input_key.startswith("lora_") and isinstance(input_val, dict):
                            if "name" in input_val:
                                # Filter based on NSFW setting
                                lora_nsfw = input_val.get("nsfw", False)
                                if ctx:
                                    user_nsfw = ctx.settings.get("nsfw", False)
                                    # Skip NSFW loras if user has NSFW disabled
                                    if lora_nsfw and not user_nsfw:
                                        continue
                                
                                loras.append({
                                    "name": input_val["name"],
                                    "type": input_val.get("type", "character")
                                })
            
            # Sort by name
            loras.sort(key=lambda x: x["name"])
            return loras

        except Exception as e:
            logging.error(f"Error reading LoRA file: {e}")
            return []
    return []


async def get_avatar_version(ctx: UserContext) -> str:
    # 1. Check remote
    if ctx.storage and ctx.omd_key:
        try:
            storage_id = ctx.storage
            storage_key = ctx.omd_key
            
            base_url = GATEWAY_URL.rstrip("/")
            clean_storage_id = storage_id.strip("/")
            # Manual URL construction to ensure token is passed correctly
            # Use GET instead of HEAD as HEAD might be blocked or malformed for this gateway
            timestamp = str(int(time.time()))
            url = f"{base_url}/{clean_storage_id}/avatar.png?token={storage_key}&_t={timestamp}"
            
            #logging.info(f"[get_avatar_version] Checking remote (GET): {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    #logging.info(f"[get_avatar_version] Response status: {resp.status}")
                    if resp.status == 200:
                        # Use ETag or Last-Modified
                        etag = resp.headers.get("ETag")
                        last_modified = resp.headers.get("Last-Modified")
                        #logging.info(f"[get_avatar_version] ETag: {etag}, Last-Modified: {last_modified}")
                        if etag:
                            return etag.strip('"')
                        if last_modified:
                            # Simple hash of last modified string
                            return str(hash(last_modified))
        except Exception as e:
            logging.warning(f"[get_avatar_version] Failed to get remote avatar version: {e}")

    # 2. Fallback to local
    try:
        storage_path = get_assistant_avatar_path(ctx)
        if storage_path.startswith("/"):
            storage_path = storage_path[1:]
        
        full_path = os.path.join(STORAGE_ROOT, storage_path)
        if os.path.exists(full_path):
            mtime = os.path.getmtime(full_path)
            #logging.info(f"[get_avatar_version] Local avatar local version: {mtime}")

            return str(int(mtime))
    except Exception as e:
        logging.warning(f"Failed to get local avatar version: {e}")
        return str(int(time.time()))

async def get_generated_avatars(ctx: UserContext) -> list:
    avatars = []
    # 1. Remote
    if ctx.storage and ctx.omd_key:
        try:
            storage_id = ctx.storage
            storage_key = ctx.omd_key
            
            base_url = GATEWAY_URL.rstrip("/")
            clean_storage_id = storage_id.strip("/")
            
            # List generated folder
            url = f"{base_url}/{clean_storage_id}/generated?list&token={storage_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200 or resp.status == 206:
                        try:
                            data = await resp.json()
                        except Exception as e:
                             text = await resp.text()
                             try:
                                 data = json.loads(text)
                             except json.JSONDecodeError as je:
                                 if "Extra data" in str(je):
                                     data = json.loads(text[:je.pos])
                                 else:
                                     raise je
                        
                        items = []
                        if "list" in data:
                            items = data["list"]
                        elif "result" in data:
                            items = data["result"]
                            
                        for item in items:
                            name = item.get("name", "")
                            item_type = item.get("type", "")
                            
                            if item_type != "dir" and name.startswith("avatar"):
                                    # Construct direct URL
                                    file_url = f"{base_url}/{clean_storage_id}/generated/{name}?token={storage_key}"
                                    avatars.append({
                                        "name": name,
                                        "url": file_url,
                                        "date": item.get("date"),
                                        "size": item.get("size")
                                    })
            return avatars
        except Exception as e:
            logging.warning(f"Failed to list remote avatars: {e}")
            # Fallthrough to local if needed or just return empty?
            # If storage is configured, we assume remote is source of truth.
            return []

    # 2. Local fallback
    try:
        # Assuming generated is adjacent to avatar.png or in user root?
        # Standard: STORAGE_ROOT / user_id / generated
        # Need to verify path. core_service has upload_to_storage logic.
        # It uses 'generated/' as path.
        # We need absolute path.
        # upload_to_storage uses: os.path.join(STORAGE_ROOT, user_id, path)
        
        user_storage_path = os.path.join(STORAGE_ROOT, ctx.user_id)
        if ctx.storage and not ctx.omd_key: # Local storage mount
             # Logic for local mounts is complex, assume standard structure for now
             user_storage_path = os.path.join(STORAGE_ROOT, ctx.storage.strip("/"))
        
        generated_dir = os.path.join(user_storage_path, "generated")
        
        if os.path.exists(generated_dir):
            for filename in os.listdir(generated_dir):
                if filename.startswith("avatar"):
                    full_path = os.path.join(generated_dir, filename)
                    if os.path.isfile(full_path):
                         # Simple local URL? Or we need api to serve it?
                         # For now, return filename. Frontend likely needs full URL.
                         # But if local, we might need an endpoint to serve it.
                         # Since remote is priority, basic implementation:
                         avatars.append({
                             "name": filename,
                             "url": "", # Frontend might fallback
                             "date": os.path.getmtime(full_path)
                         })
        
        # Sort local by date?
        avatars.sort(key=lambda x: x["date"], reverse=True)
        return avatars

    except Exception as e:
        logging.warning(f"Failed to list local avatars: {e}")
        return []


async def generate_image_workflow(workflow) -> bytes:
    client_id = str(uuid.uuid4())
    ws_url = f"{COMFY_API_URL.replace('http', 'ws')}/ws?clientId={client_id}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Send prompt
                payload = {
                    "prompt": workflow,
                    "client_id": client_id
                }
                async with session.post(f"{COMFY_API_URL}/prompt", json=payload) as resp:
                    resp_data = await resp.json()
                    prompt_id = resp_data.get("prompt_id")
                    logging.info(f"Prompt ID: {prompt_id}")

                # Listen for messages
                final_image_data = None
                final_filename = None
                
                # Timeout safety
                loop_start = time.time()

                async for msg in ws:
                    if time.time() - loop_start > 120:
                        logging.error("Timeout waiting for image generation")
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "executing":
                            data_content = data["data"]
                            # Log heartbeat or node executing
                            if data_content["node"] is None and data_content["prompt_id"] == prompt_id:
                                logging.info(f"Execution finished for prompt {prompt_id}")
                                # Determine if we should break or wait for pending fetches
                                # If we have an image, great. If not, maybe we missed it or it is coming?
                                # Usually 'executed' comes before 'executing' (finished).
                                break
                                
                        elif msg_type == "executed":
                            data_content = data["data"]
                            if data_content["prompt_id"] == prompt_id:
                                outputs = data_content.get("output", {})
                                #logging.info(f"Outputs received: {outputs.keys()}")
                                
                                # outputs is directly the dictionary of outputs for the executed node
                                if "images" in outputs:
                                    images = outputs["images"]
                                    #logging.info(f"Images found: {images}")
                                    for image in images:
                                        filename = image.get("filename")
                                        subfolder = image.get("subfolder", "")
                                        img_type = image.get("type", "output")
                                        
                                        #logging.info(f"Fetching image: {filename} [{img_type}]")
                                        
                                        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
                                        async with session.get(f"{COMFY_API_URL}/view", params=params) as img_resp:
                                            if img_resp.status == 200:
                                                final_image_data = await img_resp.read()
                                                final_filename = filename
                                                logging.info(f"Image fetched: {len(final_image_data)} bytes")
                                            else:
                                                logging.error(f"Failed to fetch image: {img_resp.status}")
                        
                        # Log other messages just in case
                        # else:
                        #     logging.debug(f"WS Message: {msg_type}")

                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        # Preview image handling
                        # First 4 bytes: integer type (1 = JPEG preview)
                        if len(msg.data) > 4:
                            event_type = int.from_bytes(msg.data[:4], 'big')
                            if event_type == 1:
                                logging.info("Received binary preview image")
                                # Only use binary preview if we don't have a high-res one yet, or as fallback
                                if not final_image_data:
                                    # Skip first 8 bytes? ComfyUI source:
                                    # const view_metadata = new DataView(event.data.slice(0, 8)); ... 
                                    # Actually python int.from_bytes is safe. 
                                    # Standard preview is JPEG.
                                    final_image_data = msg.data[8:] 
                                    final_filename = f"preview_{uuid.uuid4()}.jpg"
                                    logging.info(f"Captured preview image: {len(final_image_data)} bytes")
                            else:
                                logging.info(f"Received binary event type: {event_type}")

                if not final_image_data:
                     logging.warning("No image data captured during execution.")

                return final_image_data, final_filename

    except Exception as e:
        logging.error(f"Error in generate_image_workflow: {e}")
        return None, None

    except Exception as e:
        logging.error(f"Error in generate_image_workflow: {e}")
        return None, None



# === RAG ===
async def inject_facts(ctx: UserContext, query: str, collection: str = "", mem_id="") -> tuple[list[str], list[str]]:
    facts = []
    document_ids = []

    # Личные воспоминания
    personal = search_memories(ctx, query, collection="user", mem_id=mem_id, top_k=3)
    for m in personal:
        facts.append(f"• {m['text']}")
        doc_id = m.get("document_id")
        if doc_id:
            document_ids.append(doc_id)

    # Общие знания — если есть collection
    if collection:
        shared = search_memories(ctx, query, collection=collection, mem_id=mem_id, top_k=3)
        for m in shared:
            facts.append(f"• {m['text']}")
            doc_id = m.get("document_id")
            if doc_id:
                document_ids.append(doc_id)

    return facts, document_ids

# === Ollama запрос ===
async def llm_request_stream(payload: dict, headers: dict = None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data
                    except Exception as e:
                        logging.error(f"Stream parse error: {e}")
    except Exception as e:
        logging.error(f"LLM error: {e}")


async def llm_request(payload: dict, headers: dict = None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                return await resp.json()
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return None






# === Чат ===
async def _perform_prompt_stream(
    ctx: UserContext,
    instruction: str,
    message: str,
    chat: str,
    skip_history: bool,
    is_rag: bool,
    img_source: str = None,
    mem_id: str = None,
    think: bool = False,
    event: str = None
):
    nsfw_enabled = ctx.settings.get("nsfw", False)
    model = DEFAULT_MODEL
    b64_image = None

    if chat == "default":
        history = []
    else:
        history = load_history(ctx, chat)

    system_prompt = ""
    # === ВСПОМНИМ ФАКТЫ ===
    strict_fact = ""
    facts_text = ""
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    logging.info(f"Loading facts: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(ctx, message, collection, mem_id)
    if facts:
        facts_text = "\n\n*Known facts:*\n" + "\n".join(facts) if facts else ""
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        logging.info(f"RAG request: {collection}")
        prep_prompt = (
            "You are a fact-checking assistant. Based on *Known facts* only, respond to the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt

        if facts_text:
            rag_system_prompt += facts_text

        prep_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": message}
        ]
        prep_payload = {
            "messages": prep_messages,
            "model": SFW_MODEL,
            "stream": False,
            "options": {
               "temperature": 0.1,
            }
        }
        data = await llm_request(prep_payload)
        if not data:
            yield {"error": "⚠️ RAG query failed."}
            return

        rag_resp = data["message"]["content"].strip()
        if rag_resp and not rag_resp.startswith("No information"):
            strict_fact = rag_resp

        # Инжект фактов и источников в system prompt
        if strict_fact:
            facts_text += f"\n\n*Strict facts:*\n{strict_fact}"

    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{BASE_SYSTEM_PROMPT}"
    else:
        system_prompt = f"{BASE_SYSTEM_PROMPT}"


    # Check if the user is new or recurrent one, and prepare chat info
    chat_info = await ensure_chat(ctx, chat, message)

    # Персонализация
    user_prompt = ctx.settings.get("system_prompt", "")

    if ctx.settings.get("newUser", False):
         user_prompt = user_context.DEFAULT_UNONBOARDED_PROMPT
         system_prompt += "\n\n*Attention*:* You are communicating with the new user!\n"
    else:
         username = ctx.settings.get("username", "User")
         system_prompt += f"\n\n*Attention*:* You are communicating with existing user. User name: {username}.\n"

    system_prompt += "\n\n*Personality, appearance and behaviour:*\n" + user_prompt

    # Факты
    if facts_text:
        logging.info(facts_text)
        system_prompt += facts_text

    # Инструкция
    instruction_prompt = "*Instruction:*\n" + instruction

    instruction_prompt += "\nThe conversation is continuous; avoid redundancy and self-repetition, avoid repeating phrasing or actions from previous turns."

    # Reminder
    if think:
        instruction_prompt += "\n\n*Important:\n*For this request, think through the problem step-by-step (internally), then return only a concise final answer."

    if nsfw_enabled:
        instruction_prompt += "\n\n*Hint:*\nYou are allowed and welcome to respond in more relaxed, NSFW mode"


    # === ОСНОВНОЙ ЗАПРОС ===
    system_prompt +=  f"\nCurrent local date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    messages = [{"role": "system", "content": system_prompt}] + history[-HISTORY_LIMIT:]

    # Добавляем новый запрос
    user_message = {
        "role": "user",
        "content": message,
    }

    instruction_message = {
        "role": "system",
        "content": instruction_prompt,
    }

    if img_source:
        b64_image = await get_image_from_source(ctx, img_source)

    if b64_image:
        user_message["images"] = [b64_image]
        model = DEFAULT_MODEL
        if img_source.startswith("/"):
            user_message["image"] = {"path": img_source}

    if mem_id:
       user_message["mem_id"] = mem_id

    logging.info(f"Starting main request:{model} {think} {img_source} {mem_id}")

    # Добавляем инструкцию
    messages.append(instruction_message)
    # Добавляем пользовательский промпт
    messages.append(user_message)

    main_payload = {
        "messages": messages,
        "model": model,
        "stream": True, # Always stream for _perform_prompt_stream
        "options": {
            "temperature": 0.85,          # немного выше для разнообразия
            "top_p": 0.9,                 # ограничивает вероятность, убирая “хвост”
            "frequency_penalty": 0.6,     # штраф за частое повторение слов
            "presence_penalty": 0.5,      # штраф за повторение идей/тем
        }
    }

    if think and REASONONG_SUPPORTED:
        main_payload["think"] = True

    full_response_content = ""
    full_thinking_content = ""

    try:
        async for chunk in llm_request_stream(main_payload):
            if "content" in chunk["message"]:
                content_chunk = chunk["message"]["content"]
                full_response_content += content_chunk
                yield {"content": content_chunk, "event": event}
            if "thinking" in chunk["message"]:
                thinking_chunk = chunk["message"]["thinking"]
                full_thinking_content += thinking_chunk
                yield {"thinking": thinking_chunk, "event": event}
    except Exception as e:
        logging.error(f"Error during LLM streaming: {e}")
        yield {"error": f"Error during LLM streaming: {e}", "event": event}
        return

    # Post-processing and history saving after stream completes
    llm_response = full_response_content
    llm_response = clean_response(llm_response)
    llm_think_response = full_thinking_content

    # Add sources block
    links = []
    if doc_ids:
        seen = set()
        for doc in doc_ids:
            if doc in seen:
                continue
            seen.add(doc)
            links.append(doc)

    # result object for final history save
    response_for_history = {}
    response_for_history["content"] = llm_response.strip()
    if links:
        response_for_history["sources"] = links
    if strict_fact:
        response_for_history["facts"] = strict_fact
    if llm_think_response:
        response_for_history["thinking"] = llm_think_response

    # === Add to history
    if not skip_history:
        msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
        history.append(msg_to_save)
    history_entry = {
        "role": "assistant",
        "content": llm_response
    }
    if strict_fact:
        history_entry["facts"] = strict_fact
    if links:
        history_entry["sources"] = links
    if llm_think_response:
        history_entry["thinking"] = llm_think_response

    history.append(history_entry)

    chat_name = chat_info.get("name", chat)
    save_history(ctx, history, chat_name)

    # Yield final metadata
    final_metadata = {"chatinfo": chat_info, "event": event}
    if links:
        final_metadata["sources"] = links
    if strict_fact:
        final_metadata["facts"] = strict_fact
    if llm_think_response:
        final_metadata["thinking"] = llm_think_response

    yield final_metadata


async def _perform_prompt_sync(
    ctx: UserContext,
    instruction: str,
    message: str,
    chat: str,
    skip_history: bool,
    is_rag: bool,
    img_source: str = None,
    mem_id: str = None,
    think: bool = False
) -> dict:
    nsfw_enabled = ctx.settings.get("nsfw", False)
    model = DEFAULT_MODEL
    b64_image = None

    if chat == "default":
        history = []
    else:
        history = load_history(ctx, chat)

    system_prompt = ""
    # === ВСПОМНИМ ФАКТЫ ===
    strict_fact = ""
    facts_text = ""
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    logging.info(f"Loading facts: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(ctx, message, collection, mem_id)
    if facts:
        facts_text = "\n\n*Known facts:*\n" + "\n".join(facts) if facts else ""
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        logging.info(f"RAG request: {collection}")
        prep_prompt = (
            "You are a fact-checking assistant. Based on *Known facts* only, respond to the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt

        if facts_text:
            rag_system_prompt += facts_text

        prep_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": message}
        ]
        prep_payload = {
            "messages": prep_messages,
            "model": SFW_MODEL,
            "stream": False,
            "options": {
               "temperature": 0.1,
            }
        }
        data = await llm_request(prep_payload)
        if not data:
            return {"error": "⚠️ RAG query failed."}

        rag_resp = data["message"]["content"].strip()
        if rag_resp and not rag_resp.startswith("No information"):
            strict_fact = rag_resp

        # Инжект фактов и источников в system prompt
        if strict_fact:
            facts_text += f"\n\n*Strict facts:*\n{strict_fact}"

    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{BASE_SYSTEM_PROMPT}"
    else:
        system_prompt = f"{BASE_SYSTEM_PROMPT}"


    # Check if the user is new or recurrent one, and prepare chat info
    chat_info = await ensure_chat(ctx, chat, message)

    # Персонализация
    user_prompt = ctx.settings.get("system_prompt", "")

    if ctx.settings.get("newUser", False):
         user_prompt = user_context.DEFAULT_UNONBOARDED_PROMPT
         system_prompt += "\n\n*Attention*:* You are communicating with the new user!\n"
    else:
         username = ctx.settings.get("username", "User")
         system_prompt += f"\n\n*Attention*:* You are communicating with existing user. User name: {username}.\n"

    system_prompt += "\n\n*Personality, appearance and behaviour:*\n" + user_prompt

    # Факты
    if facts_text:
        logging.info(facts_text)
        system_prompt += facts_text

    # Инструкция
    instruction_prompt = "*Instruction:*\n" + instruction

    instruction_prompt += "\nThe conversation is continuous; avoid redundancy and self-repetition, avoid repeating phrasing or actions from previous turns."

    # Reminder
    if think:
        instruction_prompt += "\n\n*Important:\n*For this request, think through the problem step-by-step (internally), then return only a concise final answer."

    if nsfw_enabled:
        instruction_prompt += "\n\n*Hint:*\nYou are allowed and welcome to respond in more relaxed, NSFW mode"


    # === ОСНОВНОЙ ЗАПРОС ===
    system_prompt +=  f"\nCurrent local date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    messages = [{"role": "system", "content": system_prompt}] + history[-HISTORY_LIMIT:]

    # Добавляем новый запрос
    user_message = {
        "role": "user",
        "content": message,
    }

    instruction_message = {
        "role": "system",
        "content": instruction_prompt,
    }

    if img_source:
        b64_image = await get_image_from_source(ctx, img_source)

    if b64_image:
        user_message["images"] = [b64_image]
        model = DEFAULT_MODEL
        if img_source.startswith("/"):
            user_message["image"] = {"path": img_source}

    if mem_id:
       user_message["mem_id"] = mem_id

    logging.info(f"Starting main request:{model} {think} {img_source} {mem_id}")

    # Добавляем инструкцию
    messages.append(instruction_message)
    # Добавляем пользовательский промпт
    messages.append(user_message)

    main_payload = {
        "messages": messages,
        "model": model,
        "stream": False, # Always non-stream for _perform_prompt_sync
        "options": {
            "temperature": 0.85,          # немного выше для разнообразия
            "top_p": 0.9,                 # ограничивает вероятность, убирая “хвост”
            "frequency_penalty": 0.6,     # штраф за частое повторение слов
            "presence_penalty": 0.5,      # штраф за повторение идей/тем
        }
    }

    if think and REASONONG_SUPPORTED:
        main_payload["think"] = True

    data = await llm_request(main_payload)
    if not data:
        return {"error": "LLM request failed."}

    llm_response = data["message"]["content"]
    llm_response = clean_response(llm_response)
    llm_think_response = None
    if data["message"].get("thinking"):
        llm_think_response = data["message"]["thinking"]

    # Добавляем блок с источниками — только в отображаемый ответ
    links = []
    if doc_ids:
        seen = set()
        for doc in doc_ids:
            if doc in seen:
                continue  # пропускаем дубликаты
            seen.add(doc)
            # Формируем ссылку без экранирования, она будет безопасно обработана позже
            links.append(doc)

    # result object
    response = {}
    response["content"] = llm_response.strip()
    if links:  # добавляем блок только если есть ссылки
        response["sources"] = links
    if strict_fact:
        response["facts"] = strict_fact
    if llm_think_response:
        response["thinking"] = llm_think_response
    # === Добавляем в историю
    if not skip_history:
        msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
        history.append(msg_to_save)
    history_entry = {
        "role": "assistant",
        "content": llm_response
    }
    if strict_fact:
        history_entry["facts"] = strict_fact
    if links:
        history_entry["sources"] = links
    if llm_think_response:
        history_entry["thinking"] = llm_think_response

    history.append(history_entry)

    chat_name = chat_info.get("name", chat)
    save_history(ctx, history, chat_name)
    response["chatinfo"] = chat_info
    return response


async def perform_prompt(ctx: UserContext,
                         instruction: str,
                         message: str,
                         is_rag: bool=False,
                         skip_history: bool=False,
                         chat: str = "default",
                         mem_id: str = None,
                         img_source: str = None,
                         stream: bool = False,
                         think: bool = False,
                         event: str = None) -> str | AsyncGenerator:

    nsfw_enabled = ctx.settings.get("nsfw", False)
    #model =  NSFW_MODEL if nsfw_enabled else SFW_MODEL
    model = DEFAULT_MODEL
    b64_image = None

    if chat == "default":
        history = []
    else:
        history = load_history(ctx, chat)
    
    system_prompt = ""
    # === ВСПОМНИМ ФАКТЫ ===
    strict_fact = ""
    facts_text = ""
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    logging.info(f"Loading facts: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(ctx, message, collection, mem_id)
    if facts:
        facts_text = "\n\n*Known facts:*\n" + "\n".join(facts) if facts else ""
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        logging.info(f"RAG request: {collection}")
        prep_prompt = (
            "You are a fact-checking assistant. Based on *Known facts* only, respond to the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt

        if facts_text:
            rag_system_prompt += facts_text

        prep_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": message}
        ]
        prep_payload = {
            "messages": prep_messages,
            "model": SFW_MODEL,
            "stream": False,
            "options": {
               "temperature": 0.1,
            }
        }
        data = await llm_request(prep_payload)
        if not data:
            return "⚠️ RAG query failed."
        
        rag_resp = data["message"]["content"].strip()
        if rag_resp and not rag_resp.startswith("No information"):
            strict_fact = rag_resp

        # Инжект фактов и источников в system prompt
        if strict_fact:
            facts_text += f"\n\n*Strict facts:*\n{strict_fact}"
    
    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{BASE_SYSTEM_PROMPT}"
    else:  
        system_prompt = f"{BASE_SYSTEM_PROMPT}"


    # Check if the user is new or recurrent one, and prepare chat info
    chat_info = await ensure_chat(ctx, chat, message)

    # Персонализация
    user_prompt = ctx.settings.get("system_prompt", "")

    if ctx.settings.get("newUser", False):
         user_prompt = user_context.DEFAULT_USER_PROMPT
         system_prompt += "\n\n*Attention*:* You are communicating with the new user!\n"
    else:        
         username = ctx.settings.get("username", "User")
         system_prompt += f"\n\n*Attention*:* You are communicating with existing user. User name: {username}.\n"

    system_prompt += "\n\n*Personality, appearance and behaviour:*\n" + user_prompt

    # Факты
    if facts_text:
        logging.info(facts_text)
        system_prompt += facts_text

    # Инструкция
    instruction_prompt = "*Instruction:*\n" + instruction

    instruction_prompt += "\nThe conversation is continuous; avoid redundancy and self-repetition, avoid repeating phrasing or actions from previous turns."

    # Reminder
    if think:
        instruction_prompt += "\n\n*Important:\n*For this request, think through the problem step-by-step (internally), then return only a concise final answer."    
   
    if nsfw_enabled:
        instruction_prompt += "\n\n*Hint:*\nYou are allowed and welcome to respond in more relaxed, NSFW mode"


    # === ОСНОВНОЙ ЗАПРОС ===
    system_prompt +=  f"\nCurrent local date and time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    messages = [{"role": "system", "content": system_prompt}] + history[-HISTORY_LIMIT:]

    # Добавляем новый запрос
    user_message = {
        "role": "user",
        "content": message,
    }

    instruction_message = {
        "role": "system",
        "content": instruction_prompt,
    }

    if img_source:
        b64_image = await get_image_from_source(ctx, img_source)    

    if b64_image:
        user_message["images"] = [b64_image]
        model = DEFAULT_MODEL
        if img_source.startswith("/"):
            user_message["image"] = {"path": img_source}

    if mem_id:
       user_message["mem_id"] = mem_id

    logging.info(f"Starting main request:{model} {think} {img_source} {mem_id}")

    # Добавляем инструкцию
    messages.append(instruction_message)
    # Добавляем пользовательский промпт
    messages.append(user_message)

    main_payload = {
        "messages": messages,
        "model": model,
        "stream": stream,
        "options": {
            "temperature": 0.85,          # немного выше для разнообразия
            "top_p": 0.9,                 # ограничивает вероятность, убирая “хвост”
            "frequency_penalty": 0.6,     # штраф за частое повторение слов
            "presence_penalty": 0.5,      # штраф за повторение идей/тем
        }
    }

    if think and REASONONG_SUPPORTED:
        main_payload["think"] = True

    #post-processing of response
    async def process_response(data) -> dict: 
        logging.info(data)
        llm_response = data["message"]["content"]
        llm_response = clean_response(llm_response)
        llm_think_response = None
        if data["message"].get("thinking"):
            llm_think_response = data["message"]["thinking"]

        # Добавляем блок с источниками — только в отображаемый ответ
        links = []
        if doc_ids:
            seen = set()
            for doc in doc_ids:
                if doc in seen:
                    continue  # пропускаем дубликаты
                seen.add(doc)
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(doc)
    
        # result object
        response = {}
        response["content"] = llm_response.strip()
        if links:  # добавляем блок только если есть ссылки
            response["sources"] = links
        if strict_fact:    
            response["facts"] = strict_fact
        if llm_think_response:    
            response["thinking"] = llm_think_response
        # === Добавляем в историю
        if not skip_history:
            msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
            history.append(msg_to_save)
        history_entry = {
            "role": "assistant", 
            "content": llm_response
        }     
        if strict_fact:    
            history_entry["facts"] = strict_fact
        if links:
            history_entry["sources"] = links
        if llm_think_response:    
            history_entry["thinking"] = llm_think_response

        history.append(history_entry)

        #chat_info = await ensure_chat(ctx, chat, message)
        chat_name = chat_info.get("name", chat)
        save_history(ctx, history, chat_name)
        response["chatinfo"] = chat_info
        return response


    if stream:
        async def gen():
            if strict_fact:
                yield {"facts": strict_fact, "done": False, "event": event}
            accumulated_response = ""
            accumulated_thinking = ""
            thinking = False
            logging.info(f"Requesting LLM {model}")
            async for data in llm_request_stream(main_payload):
                if data.get("done"):  
                    # финал: собираем response на основе всего текста
                    full_data = {
                        "message": {
                            "role": "assistant",
                            "content": accumulated_response
                        }
                    }
                    if accumulated_thinking:
                        full_data["message"]["thinking"] = accumulated_thinking
                    response = await process_response(full_data)
                    response["done"] = True
                    response["event"] = event
                    yield response
                elif data.get("message"):   
                    if data["message"].get("thinking"):
                        delta = data["message"]["thinking"]
                        thinking = True
                        accumulated_thinking += delta
                    else:        
                        delta = data["message"]["content"]
                        thinking = False
                        accumulated_response += delta
                    yield {"delta": delta, "done": False, "thinking": thinking, "event": event}
                elif data.get("error"):    
                    logging.warning(f"{data['error']}")
                    yield {"error": data["error"], "done": True, "event": event}
                else:
                    logging.warning("Empty response")
                    yield {"error": "Empty response", "done": True, "event": event}
        return gen()    
    else:
        logging.info(f"Requesting LLM {model}")
        data = await llm_request(main_payload)
        response = await process_response(data)
        return response

# === Генерация картинок ===

# === Chats naming ==== #
async def generate_chat_title(message: str) -> str:
    """
    Спросить у LLM короткое имя для чата.
    """
    prompt = (
        "You are asked to generate a short (2–3 words) title for a chat conversation "
        "based on the following first message. Title should start with a suitable emoji separated by space. "
        "Return ONLY the title, no explanations.\n\n"
        f"Message: {message}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a naming assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": SFW_MODEL, 
        "stream": False,
        "options": {"temperature": 0.3}
    }

    data = await llm_request(payload)
    if not data:
        return "New chat"
    return data["message"]["content"].strip() or "New chat"


async def ensure_chat(ctx: UserContext, chat: str, first_message: str = None) -> dict:
    """
    Убедиться, что чат есть в chats.json и файлы подготовлены.
    Если чат = default → сгенерировать нормальное название на основе первого сообщения.
    """
    chats = load_chats_index(ctx)

    if chat not in chats or chat == "default":
        title = f"Chat {chat}"

        if (chat == "default" or not chat) and first_message:
            try:
                title = await generate_chat_title(first_message)
                chat = title.lower().replace(" ", "_")  # имя файла без пробелов
                chat = re.sub(r'[^\w]+', '', chat).strip()
                chat = chat.lstrip('_')  # Удаляет "_" слева (в начале)
            except Exception as e:
                print(f"[chats] Title generation error: {e}")

        chats[chat] = {
            "title": title,
            "file": f"{chat}.json",
            "name": chat,
            "created": datetime.now(timezone.utc).isoformat(),
            "updated": datetime.now(timezone.utc).isoformat()
        }

        
    else:
        # обновляем дату, если чат уже существует
        chats[chat]["updated"] = datetime.now(timezone.utc).isoformat()

    save_chats_index(ctx, chats)

    return chats[chat]


# === Intent ===
async def classify_user_intent(ctx: UserContext, prompt: str, chat: str = "default") -> str:
    system_prompt = INTENT_PROMPT
    history = load_history(ctx, chat)
    
    messages = [{"role": "system", "content": system_prompt}]
    # Use limited history for context to avoid confusing the classifier with too much old conversation
    messages.extend(history[-2:]) 
    messages.append({"role": "user", "content": prompt})
    
    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1, # Low temperature for classification
        }
    }
    
    data = await llm_request(request_payload)
    
    if data and "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    
    logging.warning(f"Classification failed, response: {data}")
    return "chat\nFallback" 


async def check_prompt_safety(ctx: UserContext, prompt: str) -> str:
    system_prompt = SAFETY_CHECK_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }
    
    data = await llm_request(request_payload)
    
    if data and "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    
    return "SAFE" # Default fallback

async def generate_image_prompt(ctx: UserContext, instruction: str, prompt: str, chat = "default") -> str:
    user_prompt =  "*Personality and behaviour:*\n" + ctx.settings.get("system_prompt", "") + "\n\n*Appearance:*\n" + ctx.settings.get("assistant_appearance", "")
    nsfw_enabled = ctx.settings.get("nsfw", False)
    #model =  NSFW_MODEL if nsfw_enabled else SFW_MODEL

    if nsfw_enabled:
        system_prompt = f"{NSFW_PREPHASE}\n{user_prompt}"
        image_instruction = f"{IMAGE_PROMPT_NSFW}\n{instruction.format(prompt)}"
    else:  
        system_prompt =  user_prompt
        image_instruction = instruction.format(prompt)


    history = load_history(ctx, chat)

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(history[-20:])

        # Добавляем инструкцию
    #messages.append({ "role": "system", "content": image_instruction})

    # Добавляем запрос
    messages.append({ "role": "user", "content": image_instruction})

    #model = NSFW_MODEL if nsfw_enabled else SFW_MODEL


    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    history.append({"role": "user", "content": prompt })
    #history.append({"role": "assistant", "content": response}) 
    save_history(ctx, history, chat)       

    return response.strip()


# Generate character image, returns full path for further sending or conversion
async def generate_character_image_prompt(ctx: UserContext, prompt, chat="default") -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx,  SYSTEM_INSTRUCTION_CHARACTER, prompt, chat)


# Generate general image, returns full path for further sending or conversion
async def generate_general_image_prompt(ctx: UserContext, prompt, chat="default") -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx, SYSTEM_INSTRUCTION_GENERAL, prompt, chat)


def extract_title_and_prompt(response: str) -> tuple[str, str]:
    """Extract title and image prompt from LLM response.
    Expected format: 'Title: title\nImage: prompt'
    Returns (prompt, title). If title not found, generates one from first words.
    """
    lines = response.strip().split('\n')
    title = ""
    img_prompt = ""
    
    for line in lines:
        line = line.strip()
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("Image:"):
            img_prompt = line[6:].strip()
    
    # Fallback: if no Title found, use image prompt or generate from prompt
    if not title and img_prompt:
        # Generate title from first few words
        words = img_prompt.replace('<', '').replace('>', '').split()
        clean_words = [w for w in words if not w.startswith('<')][:4]
        title = ' '.join(clean_words[:4])
    
    # Fallback: if no Image found, use entire response
    if not img_prompt:
        img_prompt = response.strip()
        if not title:
            words = img_prompt.replace('<', '').replace('>', '').split()
            title = ' '.join(words[:4])
    
    return img_prompt, title


async def generate_title_from_prompt(prompt: str) -> str:
    """Generate a descriptive title from a raw user prompt.
    For short prompts (≤4 words), returns cleaned prompt.
    For long prompts, uses LLM to generate 3-4 word title.
    """
    # Clean tags from prompt
    clean_prompt = prompt.replace('<', '').replace('>', '')
    words = [w for w in clean_prompt.split() if not w.startswith('<')]
    
    # For short prompts, use the cleaned prompt itself
    if len(words) <= 4:
        return ' '.join(words)
    
    # For long prompts, generate a descriptive title using LLM
    title_prompt = (
        "Create a short descriptive title (3-4 words maximum) for this image generation prompt. "
        "Return ONLY the title, no quotes, no explanations.\n\n"
        f"Prompt: {prompt}"
    )
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a title generation assistant."},
            {"role": "user", "content": title_prompt}
        ],
        "model": SFW_MODEL,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    
    try:
        data = await llm_request(payload)
        if data and "message" in data and "content" in data["message"]:
            title = data["message"]["content"].strip()
            logging.info(f"Generated title from raw prompt: {title}")
            return title
        else:
            # Fallback to first few words if LLM fails
            return ' '.join(words[:4])
    except Exception as e:
        logging.warning(f"Title generation failed: {e}")
        # Fallback to first few words
        return ' '.join(words[:4])



async def generate_image(ctx: UserContext, prompt, chat: str = 'default', update_history: bool = True, use_default_lora: bool = True) -> tuple[str, str]:
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    nsfw_enabled = ctx.settings.get("nsfw", False)

    negative_prompt = NEGATIVE_PROMPTS["base"]
    if nsfw_enabled:
        negative_prompt = NEGATIVE_PROMPTS["nsfw"] + "," + negative_prompt


    logging.info(f"Generating image for user: {user_id}")
    logging.info(f"Prompt: {prompt}")
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Промпт для генерации
    workflow_json["4"]["inputs"]["text"] = negative_prompt
    workflow_json["85"]["inputs"]["text"] =  prompt + ", " + IMPROVEMENT_PROMPT
    
    # Randomize seed
    seed = random.randint(1, 1125899906842624)
    if "5" in workflow_json and "inputs" in workflow_json["5"]:
        workflow_json["5"]["inputs"]["seed"] = seed

    # Выбираем модель в соответствии с режимом
    style = ctx.settings.get("style", "realistic")
    
    # Check for style tags in prompt
    tags = re.findall(r"<([^>]+)>", prompt)
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in STYLE_MODELS:
            style = tag_lower
            # If nsfw is enabled and we picked a base style, switch to nsfw version if available
            if nsfw_enabled and not style.endswith("_nsfw"):
                 if f"{style}_nsfw" in STYLE_MODELS:
                     style = f"{style}_nsfw"
            # If nsfw is disabled and we picked an nsfw style, switch to base version if available
            elif not nsfw_enabled and style.endswith("_nsfw"):
                 base_style = style[:-5]
                 if base_style in STYLE_MODELS:
                     style = base_style
            break

    # Apply NSFW suffix if not already present and using default setting logic (or if tag didn't handle it fully)
    # Actually, let's simplify: if we didn't find a tag, we use settings.
    # If we found a tag, we already tried to adjust it above.
    # But if we are using settings, we need to apply nsfw logic.
    
    # Re-evaluating logic flow:
    # 1. Default style from settings
    # 2. Override with tag if found
    # 3. Apply NSFW modifier based on nsfw_enabled flag
    
    style_from_tag = None
    for tag in tags:
        tag_lower = tag.lower()
        # Check if it is a valid style key (ignoring nsfw suffix for matching purposes if possible, or just match exact keys)
        # Let's match exact keys first, but also base keys.
        if tag_lower in STYLE_MODELS:
            style_from_tag = tag_lower
            break
            
    if style_from_tag:
        style = style_from_tag
        
    # Now ensure style matches nsfw setting
    if nsfw_enabled:
        if not style.endswith("_nsfw") and f"{style}_nsfw" in STYLE_MODELS:
            style = f"{style}_nsfw"
    else:
        if style.endswith("_nsfw"):
             base_style = style[:-5]
             if base_style in STYLE_MODELS:
                 style = base_style

    model = STYLE_MODELS.get(style, STYLE_MODELS["realistic"]) # Fallback just in case
    workflow_json["127"]["inputs"]["ckpt_name"] = model
    #workflow_json["103"]["inputs"][lora]["on"] = True
    
    # Dynamic LoRA activation
    lora_map = {}
    # Build map from name to key
    if "103" in workflow_json and "inputs" in workflow_json["103"]:
        for key, value in workflow_json["103"]["inputs"].items():
            if isinstance(value, dict) and "name" in value:
                lora_map[value["name"].lower()] = key

    # Find tags in prompt
    active_lora_keys = []
    tags = re.findall(r"<([^>]+)>", prompt)
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in lora_map:
            key = lora_map[tag_lower]
            active_lora_keys.append(key)
            logging.info(f"Found LoRA tag: {tag} ({key})")

    # Fallback to assistant_model if no tags found
    if not active_lora_keys and use_default_lora:
        assistant_model = ctx.settings.get("assistant_model", "").lower()
        if assistant_model in lora_map:
            key = lora_map[assistant_model]
            active_lora_keys.append(key)
            logging.info(f"Using default LoRA: {assistant_model} ({key})")

    # Activate LoRAs and set strength
    #lora_count = len(active_lora_keys)
    target_strength = 1.0
    #if (style.startswith("perfect") or style.startswith("perfection")) and lora_count > 1:
    #    target_strength = 0.6
    
    for key in active_lora_keys:
        workflow_json["103"]["inputs"][key]["on"] = True
        workflow_json["103"]["inputs"][key]["strength"] = target_strength
        logging.info(f"Activated LoRA {key} with strength {target_strength}")

    logging.info(f"Generating with model: {model}")
    img_data, filename = await generate_image_workflow(workflow_json)
    
    if not img_data:
        raise Exception("Image generation failed")

    # Папка для пользователя
    user_folder = os.path.join(APP_ROOT_DIR, USER_DATA_DIR, ctx.user_id, "generated")
    os.makedirs(user_folder, exist_ok=True)
    
    # Extract title from prompt (prompt may contain "Title: ..." from LLM)
    img_prompt, img_title = extract_title_and_prompt(prompt)
    
    # Format for markdown file
    formatted_prompt = f"#{img_title}\n\n{img_prompt}"
    
    # Rename filename if it comes from ComfyUI temp
    # Pattern: ComfyUI_temp_..._00005_.png
    # Extract index
    match = re.search(r"_(\d{5})_\.png$", filename)
    if match:
        index = match.group(1)
        filename = f"IMG_{index}.png"
    else:
        # Fallback if pattern doesn't match but we want clean names
        if filename.startswith("ComfyUI_temp"):
             # timestamp fallback
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             filename = f"IMG_{timestamp}.png"

    if ctx.storage and ctx.omd_key:
        # Копируем файл юзеру на устройство
        dest = f"{ctx.storage}/generated"
        logging.info(f"Uploading to storage: {dest}/{filename}")
        
        # Since we have bytes, we use upload_data_to_storage (or similar, but upload_data_to_storage handles generic data? check implementation)
        # utils.upload_data_to_storage handles str or bytes.
        try:
            upload_data_to_storage(ctx.omd_key, dest, filename, img_data, "image/png")
            
            # Save prompt as description (Readme.md)
            readme_filename = os.path.splitext(filename)[0] + ".Readme.md"
            upload_data_to_storage(ctx.omd_key, dest, readme_filename, formatted_prompt, "text/markdown")
            logging.info("Upload completed successfully.")
        except Exception as e:
            logging.error(f"Upload to storage failed: {e}")
            # Fallback to local? Or just fail? 
            # If upload fails, maybe we should try local save as backup?
            # For now just log.
            raise e

    else:    
        logging.info("No storage/key found, saving locally.")
        # Копируем файл в user_data (local)
        dest_path = os.path.join(user_folder, filename)
        with open(dest_path, "wb") as f:
            f.write(img_data)
        # Save prompt as description (Readme.md)
        readme_filename = os.path.splitext(filename)[0] + ".Readme.md"
        readme_path = os.path.join(user_folder, readme_filename)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    if update_history:
        history = load_history(ctx, chat)
        history.append({"role": "assistant", "content": img_prompt, "image": {"path": filename, "title": img_title}})
        save_history(ctx, history, chat)
    return filename, img_title

async def generate_character_image(ctx: UserContext, prompt, chat: str = 'default') -> tuple[str, str]:
    return await generate_image(ctx, prompt, chat)

# Generate general image, returns full path for further sending or conversion
async def generate_general_image(ctx: UserContext, prompt, chat: str = 'default') -> tuple[str, str]:
    return await generate_image(ctx, prompt, chat)

# img is base64 image #
async def recognize_image(ctx: UserContext, img, prompt="", chat="default"):

    nsfw_enabled = ctx.settings.get("nsfw", False)

    if nsfw_enabled:
        system_prompt += NSFW_PREPHASE + "\n" +  BASE_SYSTEM_PROMPT + "\n" + "Recognize image"
    else:
        system_prompt +=  BASE_SYSTEM_PROMPT + "\n" + "Recognize image"


    history = load_history(ctx, chat)

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]
    # Добавляем историю
    messages.extend(history[-HISTORY_LIMIT:])
    # Добавляем новый запрос с изображением
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [img]
    })

    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.8,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    history.append({"role": "assistant", "content": response})
    save_history(ctx, history, chat)

    return response.lower().strip()

# Суммаризация документа

async def summarize_for_memory(raw_text: str, limit: int = 8000) -> str:
    """
    Создаёт 'карточку памяти' документа для дальнейшего поиска.
    :param raw_text: исходный текст документа
    :param limit: максимальное количество символов для передачи модели (по умолчанию ~8000)
    """
    # Усечём текст, если длиннее лимита
    text_to_process = raw_text[:limit]

    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": text_to_process},
    ]

    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.1,
        }
    }

    data = await llm_request(request_payload)

    # Универсальное извлечение текста
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            response = data["message"]["content"]
        else:
            response = data.get("content") or str(data)
    else:
        response = str(data)

    logging.info(f"Summary: {response}")    

    return response.strip()

# === Импорт и память ===
async def import_doc(ctx: UserContext, url_or_path, collection="user"):
    key = ctx.omd_key or ctx.settings.get("omd_key", "")

    logging.info(f"[import] importing: {url_or_path}")
    raw_text = ""
    # Определяем, это OMD или нет
    is_omd = url_or_path.startswith("/") or GATEWAY_URL in url_or_path
    
    if is_omd:
        if url_or_path.startswith("/"):
            url_or_path = f"{GATEWAY_URL}{url_or_path}"
        if not key:
            raise Exception("⚠️ Provide On My Disk account key to access your files:\n`/bind abcdxxxxx...`")
        
        raw_text = await fetch_document_text(url_or_path, key)
    else:
        # External URL or fallback
        raw_text = await fetch_document_text(url_or_path)

    if raw_text.startswith("Failed to fetch document:") or raw_text.startswith("Unsupported file type:"):
        logging.error(f"[import] failed fetch: {raw_text}")    
        return {
            "id": "error",
            "error": True,
            "text": raw_text
        }

    # If it's an external HTML or we just want to ensure it's plain text via pandoc
    if not is_omd or url_or_path.lower().endswith(".html") or url_or_path.lower().endswith(".htm"):
        # We use pandoc to clean up HTML or other formats if needed
        # Note: fetch_document_text for OMD might already return clean text if ?totext was used,
        # but for external URLs it returns raw HTML.
        
        cmd = [
            "pandoc",
            "-f", "html",
            "-t", "plain",
        ]
        try:
            logging.info(f"[import] converting with pandoc (input length: {len(raw_text)})")
            result = subprocess.run(cmd, input=raw_text, capture_output=True, text=True, check=True)
            raw_text = result.stdout
        except Exception as e:
            logging.error(f"[import] pandoc failed: {e}")    
            return {
                "id": "error",
                "error": True,
                "text": f"Error during conversion: {e}"
            }

    # Векторизация и сохранение чанков
    chunk_and_vectorize_to_file(
        ctx,
        text=raw_text,
        document_id=url_or_path,
        collection=collection
    )

    # Добавление краткой аннотации в память
    card_text = await summarize_for_memory(raw_text)
    mem_id = add_memory_card(
        ctx,
        text=card_text,
        document_id=url_or_path,
        collection=collection
    )

    mem_card = {
        "id": mem_id,
        "text": card_text
    }
    return mem_card

def memorize(ctx, text):
    # Добавление краткой аннотации в память
    return add_memory_card(ctx, text, collection="user", relevance="permanent")


async def generate_avatar(ctx: UserContext, style: str, character_lora: str, prompt: str):
    try:
        # 1. Load Workflow
        if not os.path.exists(WORKFLOW_PATH):
            logging.error("Workflow file not found")
            return None
            
        try:
            with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load workflow: {e}")
            return None
    
        # 2. Determine Style Checkpoint
        style_map = {
            "realistic": "realistic",
            "dream": "fantasy", 
            "perfect": "perfect",
            "tooned": "tooned"
        }
        backend_style = style_map.get(style, "realistic")
        
        nsfw = ctx.settings.get("nsfw", False)
        if nsfw:
            backend_style_nsfw = backend_style + "_nsfw"
            if backend_style_nsfw in STYLE_MODELS:
                backend_style = backend_style_nsfw
                
        # STYLE_MODELS values are filenames (strings), not dicts
        ckpt_filename = STYLE_MODELS.get(backend_style, STYLE_MODELS["realistic"])
    
        # 3. Setup Workflow Parameters
        seed = random.randint(1, 9999999999)
        
        # Helper to find node
        def find_nodes_by_class(class_type):
            return [node for node in workflow.values() if node.get("class_type") == class_type]
    
        # Set Seed
        for node in find_nodes_by_class("KSampler"):
            if "inputs" in node and "seed" in node["inputs"]:
                node["inputs"]["seed"] = seed
                
        # Set Resolution (512x512)
        for node in find_nodes_by_class("EmptyLatentImage"):
            if "inputs" in node:
                node["inputs"]["width"] = 512
                node["inputs"]["height"] = 512
            
        # Set Checkpoint
        for node in find_nodes_by_class("CheckpointLoaderSimple"):
            if "inputs" in node:
                node["inputs"]["ckpt_name"] = ckpt_filename
            
            # Set Prompt
        prompt_set = False
        
        # Add appearance to prompt
        negative_prompt = NEGATIVE_PROMPTS["nsfw"] + ", " + NEGATIVE_PROMPTS["base"]
        appearance = ctx.settings.get("assistant_appearance", "")
        full_prompt = f"{prompt} {appearance}"
        logging.info(f"Generating avatar with style={style}, character={character_lora} prompt={full_prompt}")
        
        workflow["4"]["inputs"]["text"] = negative_prompt
        workflow["85"]["inputs"]["text"] = full_prompt
    
        # 4. Inject Character LoRA (if workflow supports it)
        if character_lora:
            # We look for "Power Lora Loader (rgthree)" as used in available_loras
            lora_nodes = find_nodes_by_class("Power Lora Loader (rgthree)")
            if lora_nodes:
                for node in lora_nodes:
                    inputs = node.get("inputs", {})
                    # The loader might have inputs like lora_1, lora_2 etc which are dicts? 
                    # based on previous analysis of available_loras, input val is dict with "name"
                    for key, val in inputs.items():
                        if isinstance(val, dict) and val.get("name") == character_lora:
                            val["on"] = True
                            logging.info(f"Enabled LoRA: {character_lora}")
            else:
                # If standard LoraLoader?
                lora_nodes = find_nodes_by_class("LoraLoader")
                if lora_nodes:
                    # We need mapping from Name -> Filename.
                    # This is tricky without reading analog_character_lora.json or having a map.
                    # For now, if usage implies Power Lora Loader, we stick to that or skip.
                    logging.warning("Character LoRA requested but no compatible LoRA loader found in workflow.")
                        
        # 5. Generate
        # Reuse existing workflow generator which handles websocket and bytes retrieval
        image_data, _ = await generate_image_workflow(workflow)
        
        if image_data:
            # Upload to 'generated' folder in user storage
            filename = f"avatar_{uuid.uuid4()}.png"
            
            if ctx.storage and ctx.omd_key:
                try:
                    dest_path = f"{ctx.storage}/generated"
                    # upload_data_to_storage(omd_key, dest, filename, data, mime)
                    upload_data_to_storage(ctx.omd_key, dest_path, filename, image_data, "image/png")
                    logging.info(f"Avatar uploaded to {dest_path}/{filename}")
                    
                    # Construct public URL
                    base_url = GATEWAY_URL.rstrip("/")
                    clean_storage = ctx.storage.strip("/")
                    full_url = f"{base_url}/{clean_storage}/generated/{filename}"
                    
                    return {"image": filename, "url": full_url}
                except Exception as e:
                    logging.error(f"Failed to upload avatar to storage: {e}")
                    return None
            else:
                 # Fallback for local users (if any, though context implies OMD usage mostly)
                 # But we want to avoid local fs if possible. 
                 # If no storage, we might have to save locally or fail?
                 # Let's save locally as fallback but log warning.
                 output_dir = os.path.join(os.path.dirname(__file__), "generated")
                 if not os.path.exists(output_dir):
                     os.makedirs(output_dir, exist_ok=True)
                 filepath = os.path.join(output_dir, filename)
                 with open(filepath, "wb") as f:
                     f.write(image_data)
                 logging.warning(f"No storage context, saved locally to {filepath}")
                 return {"image": filename}

        else:
            logging.error("No image data received from workflow")
            return None

    except Exception as e:
        logging.error(f"Avatar generation crashed: {e}", exc_info=True)
        return None


