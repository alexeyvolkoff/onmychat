# dialog_history.py

import os
import json
from config import SETTINGS
from config import USER_DATA_DIR

HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"]) or 200  # Используется в telegram-bot.py

def _get_path(user_id: str, chat: str) -> str:
    return  f"{USER_DATA_DIR}/{user_id}/chats/{chat}.json"

def load_history(user_id: str, chat: str = "default", limit: int | None = None) -> list:
    path = _get_path(user_id, chat)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
            if limit is not None and limit > 0:
                return history[-limit:]
            return history
    except Exception as e:
        print(f"[history] Load error: {user_id} {e}")
        return []

def save_history(user_id: str, history: list,  chat: str = "default"):
    os.makedirs(f"{USER_DATA_DIR}/{user_id}/chats", exist_ok=True)
    path = _get_path(user_id, chat)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history[-HISTORY_LIMIT:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[history] Save error: {e}")

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