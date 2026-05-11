from dataclasses import dataclass
import os
import json
import logging
from config import SETTINGS

# Дефольные настройки
DEFAULT_KB_ID = SETTINGS.get("DEFAULT_KB_ID", "omd")
DEFAULT_ASSISTANT_NAME = SETTINGS.get("DEFAULT_ASSISTANT_NAME", "June")
DEFAULT_ASSISTANT_TITLE = SETTINGS.get("DEFAULT_ASSISTANT_TITLE", "Assistant")
USER_DATA_DIR = "user_data"

# Файл для сохранения связей (для Telegram бота)
BINDINGS_FILE = f"{USER_DATA_DIR}/bindings.json"

bindings = {"by_telegram": {}, "by_account": {}}

@dataclass
class UserContext:
    type: str
    user_id: str
    settings: dict
    history: list
    omd_key: str = ""
    storage: str = ""

def get_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    try:
        path = os.path.join(prompts_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load prompt {filename}: {e}")
    return ""

DEFAULT_USER_PROMPT = get_prompt("default_user.txt")
DEFAULT_ASSISTANT_APPEARANCE = get_prompt("default_appearance.txt")

def load_user_settings(**kwargs):
    """Возвращает базовые настройки. Все важное придет от клиента."""
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
    """No-op. Настройки теперь живут на клиенте."""
    pass

def load_bindings():
    """Для Telegram бота."""
    global bindings
    if os.path.exists(BINDINGS_FILE):
        try:
            with open(BINDINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "by_telegram" in data:
                    data["by_telegram"] = {int(k): v for k, v in data["by_telegram"].items()}
                bindings["by_telegram"] = data.get("by_telegram", {})
                bindings["by_account"] = data.get("by_account", {})
        except Exception:
            pass

def save_bindings():
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        with open(BINDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(bindings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Save bindings error: {e}")

def get_context_by_account(account_id: str, storage: str = "", **kwargs) -> UserContext:
    """Создает контекст на лету. Минимум логики, максимум скорости."""
    if not account_id:
        return UserContext(type="temp", user_id="anon", settings=load_user_settings(), history=[], storage=storage)

    # Проверка биндингов (для тех, кто пришел из TG)
    if account_id in bindings["by_account"]:
        username = bindings["by_account"][account_id]["username"]
        return UserContext(type="omd", user_id=username, settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)
    
    # Для Web - stateless идентификатор
    return UserContext(type="temp", user_id=f"web_{account_id[:8]}", settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)

def get_context(telegram_id: int) -> UserContext:
    """Для Telegram бота."""
    if telegram_id in bindings["by_telegram"]:
        binding = bindings["by_telegram"][telegram_id]
        return UserContext(type="omd", user_id=binding["username"], settings=load_user_settings(), history=[], omd_key=binding.get("account_id"))
    return UserContext(type="temp", user_id=str(telegram_id), settings=load_user_settings(), history=[])

def bind(ctx: UserContext, account_id: str):
    """Связывает TG с OMD аккаунтом. Используется только при онбординге в Telegram."""
    import requests
    url = f"{GATEWAY_URL}/userinfo"
    data = {"action": "getUserInfo", "session_id": account_id}
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        user_info = response.json()
        if user_info.get("valid"):
            username = user_info["user"]
            bindings["by_telegram"][int(ctx.user_id)] = {"account_id": account_id, "username": username}
            bindings["by_account"][account_id] = {"telegram_id": int(ctx.user_id), "username": username}
            save_bindings()
            ctx.type = "omd"
            ctx.user_id = username
            ctx.omd_key = account_id
            return ctx
    except Exception as e:
        logging.error(f"Bind error: {e}")
    return ctx

load_bindings()
