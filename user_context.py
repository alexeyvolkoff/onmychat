from dataclasses import dataclass
import os
import logging
from config import SETTINGS

# Дефольные настройки
DEFAULT_KB_ID = SETTINGS.get("DEFAULT_KB_ID", "omd")
DEFAULT_ASSISTANT_NAME = SETTINGS.get("DEFAULT_ASSISTANT_NAME", "June")
DEFAULT_ASSISTANT_TITLE = SETTINGS.get("DEFAULT_ASSISTANT_TITLE", "Assistant")
USER_DATA_DIR = "user_data"

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

def get_context_by_account(account_id: str, storage: str = "", force_reload: bool = False, **kwargs) -> UserContext:
    """Создает контекст на лету. Минимум логики, максимум скорости."""
    if not account_id:
        return UserContext(type="temp", user_id="anon", settings=load_user_settings(), history=[], storage=storage)

    # Для Web - stateless идентификатор
    # Пытаемся получить имя пользователя через токен, если возможно
    username = get_username_from_token(account_id)
    if username:
        return UserContext(type="omd", user_id=username, settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)

    return UserContext(type="temp", user_id=f"web_{account_id[:8]}", settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)

def get_username_from_token(account_id: str) -> str | None:
    """Fetches the real username from OMD gateway using the session/token."""
    import requests
    from config import SETTINGS
    gateway_url = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net")
    url = f"{gateway_url}/userinfo"
    data = {"action": "getUserInfo", "session_id": account_id}
    try:
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        user_info = response.json()
        if user_info.get("valid"):
            return user_info["user"]
    except Exception as e:
        logging.error(f"Get username error: {e}")
    return None
