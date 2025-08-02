# dialog_history.py

import os
import json

HISTORY_DIR = "history"
HISTORY_LIMIT = 200  # Используется в telegram-bot.py

def _get_path(user_id: int) -> str:
    return os.path.join(HISTORY_DIR, f"{user_id}.json")

def load_history(user_id: int) -> list:
    path = _get_path(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[history] Ошибка загрузки: {e}")
        return []

def save_history(user_id: int, history: list):
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = _get_path(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history[-HISTORY_LIMIT:], f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[history] Ошибка сохранения: {e}")

def reset_history(user_id: int):
    path = _get_path(user_id)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"[history] Ошибка удаления: {e}")

