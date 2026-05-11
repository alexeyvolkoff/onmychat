from dataclasses import dataclass
import os
import json
import logging
from config import SETTINGS

# Дефольные настройки пользователя
DEFAULT_KB_ID = SETTINGS.get("DEFAULT_KB_ID", "omd")
DEFAULT_ASSISTANT_NAME = SETTINGS.get("DEFAULT_ASSISTANT_NAME", "June")
DEFAULT_ASSISTANT_TITLE = SETTINGS.get("DEFAULT_ASSISTANT_TITLE", "Assistant")
GATEWAY_URL = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net")
USER_DATA_DIR = "user_data"

# Файл для сохранения связей (нужен для Telegram бота)
BINDINGS_FILE = f"{USER_DATA_DIR}/bindings.json"

# Структура биндингов
bindings = {
    "by_telegram": {},
    "by_account": {}
}

@dataclass
class UserContext:
    type: str   # "omd" или "temp"
    user_id: str  # стабильный ID для папок (username или хеш ключа)
    settings: dict
    history: list
    omd_key: str = ""
    storage: str = ""

def get_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    try:
        if os.path.exists(os.path.join(prompts_dir, filename)):
            with open(os.path.join(prompts_dir, filename), "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load prompt {filename}: {e}")
    return ""

DEFAULT_UNONBOARDED_PROMPT = get_prompt("default.txt")
DEFAULT_USER_PROMPT = get_prompt("default_user.txt")
DEFAULT_ASSISTANT_APPEARANCE = get_prompt("default_appearance.txt")

def load_user_settings(user_id=None, **kwargs):
    """Возвращает базовые настройки. Клиент перекроет их своими."""
    return {
        "nsfw": False,
        "style": "realistic",
        "system_prompt": DEFAULT_USER_PROMPT,
        "assistant_name": DEFAULT_ASSISTANT_NAME,
        "assistant_title": DEFAULT_ASSISTANT_TITLE,
        "assistant_appearance": DEFAULT_ASSISTANT_APPEARANCE,
        "kb_id": DEFAULT_KB_ID
    }

def save_user_settings(ctx: UserContext):
    """Legacy. В новой архитектуре настройки хранятся на клиенте/OrbitDB."""
    pass

def load_bindings():
    """Загрузить биндинги (для Telegram бота)."""
    global bindings
    if os.path.exists(BINDINGS_FILE):
        try:
            with open(BINDINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "by_telegram" in data:
                    data["by_telegram"] = {int(k): v for k, v in data["by_telegram"].items()}
                bindings["by_telegram"] = data.get("by_telegram", {})
                bindings["by_account"] = data.get("by_account", {})
        except Exception as e:
            logging.error(f"[bindings] Load error: {e}")

def save_bindings():
    """Сохранить биндинги."""
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        data_to_save = {
            "by_telegram": bindings["by_telegram"],
            "by_account": bindings["by_account"]
        }
        with open(BINDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"[bindings] Save error: {e}")

def get_context_by_account(account_id: str, storage: str = "", force_reload: bool = False) -> UserContext:
    """
    Возвращает контекст. 
    Больше не делает синхронных запросов к OMD /userinfo.
    """
    if not account_id:
        user_id = "temp_anon"
        return UserContext(type="temp", user_id=user_id, settings=load_user_settings(), history=[], storage=storage)

    # Если это известный (привязанный в TG) аккаунт, берем его username
    if account_id in bindings["by_account"]:
        username = bindings["by_account"][account_id]["username"]
        return UserContext(type="omd", user_id=username, settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)
    
    # Для новых Web-сессий используем короткий хеш ключа как user_id для папок
    user_id = f"u_{account_id[:10]}"
    return UserContext(type="temp", user_id=user_id, settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)

def get_context(telegram_id: int) -> UserContext:
    """Нужен только для Telegram бота."""
    if telegram_id in bindings["by_telegram"]:
        binding = bindings["by_telegram"][telegram_id]
        account_id = binding.get("account_id")
        return UserContext(type="omd", user_id=binding["username"], settings=load_user_settings(), history=[], omd_key=account_id)
    
    return UserContext(type="temp", user_id=str(telegram_id), settings=load_user_settings(), history=[])

load_bindings()
