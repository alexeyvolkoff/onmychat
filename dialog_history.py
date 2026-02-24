# dialog_history.py

import os
import json
import requests
from config import SETTINGS
from config import USER_DATA_DIR
from user_context import UserContext, create_profile
from utils import  upload_data_to_storage
import time


GATEWAY_URL = SETTINGS["GATEWAY_URL"]

HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"]) or 200  # Используется в telegram-bot.py

def _get_path(user_id: str, chat: str) -> str:
    return  f"{USER_DATA_DIR}/{user_id}/chats/{chat}.json"

# Вспомогательная функция: переносит prompt из image в content
def _inject_image_prompts(history: list) -> list:
    """
    Переносит prompt из image в content и удаляет его из image.
    """
    if not history:
        return []

    for msg in history:
        if msg.get("role") == "assistant" and "image" in msg:
            image_data = msg["image"]
            
            # Проверяем, что есть поле prompt
            if isinstance(image_data, dict) and "prompt" in image_data:
                # .pop() забирает значение и удаляет ключ из словаря
                prompt_text = image_data.pop("prompt") 
                
                # Добавляем текст в content
                existing_content = msg.get("content", "")
                if existing_content:
                    msg["content"] = f"{existing_content}\n{prompt_text}"
                else:
                    msg["content"] = prompt_text

            # Ensure image description exists (for legacy chats or where prompt was just moved)
            if isinstance(image_data, dict) and "description" not in image_data:
                # Use content (which now contains the prompt) as description fallback
                # This ensures frontend has something to show
                msg_content = msg.get("content", "")
                if msg_content:
                     image_data["description"] = msg_content

    return history

def load_history(ctx: UserContext, chat: str = "default") -> list:
    try:

        if not chat or chat == "default":
            return []

        if ctx.history:
            return ctx.history


        if ctx.storage and ctx.omd_key:
            url = f"{GATEWAY_URL}/{ctx.storage}/chats/{chat}.json?nocache={int(time.time())}"
            token = ctx.omd_key
            headers = {"Authorization": f"token:{token}"}
            print(f"[loading history]: {ctx.user_id} {url} {token}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                if resp.text.strip():
                    history = json.loads(resp.content.decode("utf-8"))
                    history = _inject_image_prompts(history)
                    # save chat history in context
                    ctx.history = history   
                    return  history
                return []
            elif resp.status_code == 404:
                return []
            else:
                resp.raise_for_status()
        else:
            # локальный fallback
            path = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
                history = _inject_image_prompts(history)
                # save chat history in context
                ctx.history = history   
                return  history
        
    except Exception as e:
        print(f"[history] Empty history: {ctx.user_id} {ctx.storage} {chat} {e}")
        return []
    

def save_history(ctx: UserContext, history: list, chat: str = "default"):
    try:
        if not chat or chat == "default":
             # logging.warning("Attempted to save history for empty or default chat name")
             return

        if ctx.storage and ctx.omd_key:
            dest = f"{ctx.storage}/chats"
            upload_data_to_storage(ctx.omd_key, dest, f"{chat}.json", history, "application/json")
        else:
            # локальный fallback - только для непривязанных телеграм акков
            if ctx.user_id.isdigit():
                path = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[history] Save error: {ctx.user_id} {e}")

def reset_history(user_id: str):
    path = _get_path(user_id)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"[history] Remove errer: {e}")


def _chats_index_path(user_id: str) -> str:
    return f"{USER_DATA_DIR}/{user_id}/chats/chats.json"

def load_chats_index(ctx: UserContext) -> dict:
    """
    Загрузить список чатов пользователя (индекс).
    Если настроено облачное хранилище – грузим оттуда,
    иначе используем локальный fallback.
    """
    try:
        storage = ctx.storage
        omd_key = ctx.omd_key
        print(f"[history] Load index: {ctx.user_id} {storage}")

        if storage and omd_key:
            url = f"{GATEWAY_URL}/{storage}/chats/chats.json?nocache={int(time.time())}"
            headers = {"Authorization": f"token:{omd_key}"}
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200:
                if resp.text.strip():
                    return json.loads(resp.content.decode("utf-8"))
                return {}
            elif resp.status_code == 404:
                return {}
            else:
                resp.raise_for_status()

        # --- локальный fallback ---
        path = _chats_index_path(ctx.user_id)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"[chats] Load index error: {ctx.user_id} {e}")
        raise e


def save_chats_index(ctx: UserContext, chats: dict):
    """
    Сохранить список чатов пользователя (индекс).
    Если настроено облачное хранилище – сохраняем туда,
    иначе локальный fallback.
    """
    try:
        storage = ctx.storage
        omd_key = ctx.omd_key

        if storage and omd_key:
            dest = f"{storage}/chats"
            upload_data_to_storage(
                omd_key,
                dest,
                "chats.json",
                chats,
                "application/json"
            )
        else:
            # --- локальный fallback - только для непривязанных телеграм акков ---
            if ctx.user_id.isdigit():
                path = _chats_index_path(ctx.user_id)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(chats, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[chats] Save error: {ctx.user_id} {e}")

