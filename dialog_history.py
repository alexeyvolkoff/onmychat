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

def load_history(ctx: UserContext, chat: str = "default") -> list:
    try:
        if ctx.settings.get("storage") and ctx.settings.get("omd_key"):
            url = f"{GATEWAY_URL}/{ctx.settings['storage']}/chats/{chat}.json?nocache={int(time.time())}"
            token = ctx.settings["omd_key"]
            headers = {"Authorization": f"token:{token}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                history = json.loads(resp.content.decode("utf-8"))
                return  history
            return []
        else:
            # локальный fallback
            path = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
                return  history
    except Exception as e:
        print(f"[history] Empty history: {ctx.user_id} {ctx.settings.get("storage")} {chat} {e}")
        return []
    

def save_history(ctx: UserContext, history: list, chat: str = "default"):
    try:
        if ctx.settings.get("storage") and ctx.settings.get("omd_key"):
            dest = f"{ctx.settings['storage']}/chats"
            upload_data_to_storage(ctx.settings['omd_key'], dest, f"{chat}.json", history, "application/json")
        else:
            # локальный fallback
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
        storage = ctx.settings.get("storage")
        omd_key = ctx.settings.get("omd_key")
        #print(f"[history] Load index: {ctx.user_id} {storage}")

        if storage and omd_key:
            url = f"{GATEWAY_URL}/{storage}/chats/chats.json?nocache={int(time.time())}"
            headers = {"Authorization": f"token:{omd_key}"}
            resp = requests.get(url, headers=headers, timeout=10)

            if resp.status_code == 200 and resp.text.strip():
                return json.loads(resp.content.decode("utf-8"))
            return {}

        # --- локальный fallback ---
        path = _chats_index_path(ctx.user_id)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"[chats] Empty index: {ctx.user_id} {e}")
        if storage and omd_key:
            create_profile(ctx, omd_key, storage)
        return {}


def save_chats_index(ctx: UserContext, chats: dict):
    """
    Сохранить список чатов пользователя (индекс).
    Если настроено облачное хранилище – сохраняем туда,
    иначе локальный fallback.
    """
    try:
        storage = ctx.settings.get("storage")
        omd_key = ctx.settings.get("omd_key")

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
            # --- локальный fallback ---
            path = _chats_index_path(ctx.user_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(chats, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[chats] Save error: {ctx.user_id} {e}")