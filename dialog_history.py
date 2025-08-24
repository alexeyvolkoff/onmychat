# dialog_history.py

import os
import json
import requests
from config import SETTINGS
from config import USER_DATA_DIR
from user_context import UserContext
from utils import  upload_data_to_storage


HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"]) or 200  # Используется в telegram-bot.py

def _get_path(user_id: str, chat: str) -> str:
    return  f"{USER_DATA_DIR}/{user_id}/chats/{chat}.json"

def load_history(ctx: UserContext, chat: str = "default", limit: int | None = None) -> list:
    try:
        if ctx.type == "omd" and ctx.settings.get("storage") and ctx.settings.get("omd_key"):
            url = f"https://onmydisk.net/{ctx.settings['storage']}/{ctx.user_id}/chats/{chat}.json"
            token = ctx.settings["omd_key"]
            headers = {"Authorization": f"token:{token}"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                history = json.loads(resp.content.decode("utf-8"))
                return history[-limit:] if limit else history
            return []
        else:
            # локальный fallback
            path = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history[-limit:] if limit else history
    except Exception as e:
        print(f"[history] Load error: {ctx.user_id} {e}")
        return []
    

def save_history(ctx: UserContext, history: list, chat: str = "default"):
    try:
        trimmed = history[-HISTORY_LIMIT:]
        if ctx.type == "omd" and ctx.settings.get("storage") and ctx.settings.get("omd_key"):
            dest = f"{ctx.settings['storage']}/{ctx.user_id}/chats"
            upload_data_to_storage(ctx.settings['omd_key'], dest, f"{chat}.json", trimmed, "application/json")
        else:
            # локальный fallback
            path = f"{USER_DATA_DIR}/{ctx.user_id}/chats/{chat}.json"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, ensure_ascii=False, indent=2)
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

def load_chats_index(user_id: str) -> dict:
    path = _chats_index_path(user_id)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[chats] Load error: {e}")
        return {}

def save_chats_index(user_id: str, chats: dict):
    os.makedirs(f"{USER_DATA_DIR}/{user_id}/chats", exist_ok=True)
    path = _chats_index_path(user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)