from dataclasses import dataclass
import os
import logging
from config import SETTINGS

# Дефольные настройки
DEFAULT_KB_ID = SETTINGS.get("DEFAULT_KB_ID", "omd")
DEFAULT_ASSISTANT_NAME = SETTINGS.get("DEFAULT_ASSISTANT_NAME", "June")
DEFAULT_ASSISTANT_TITLE = SETTINGS.get("DEFAULT_ASSISTANT_TITLE", "Assistant")
USER_DATA_DIR = "user_data"

def node_owner() -> str:
    """Владелец ноды: NODE_OWNER из конфига, иначе юзер ОС под которым работает процесс."""
    import getpass
    return SETTINGS.get("NODE_OWNER") or getpass.getuser()

@dataclass
class UserContext:
    type: str
    user_id: str
    settings: dict
    history: list
    group: str = ""
    groups: list = None
    omd_key: str = ""
    storage: str = ""
    private_mode: bool = False
    is_unlimited: bool = False
    tokens_consumed: float = 0.0

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

DEFAULT_USER_PROMPT = get_prompt("default.txt")
DEFAULT_ASSISTANT_APPEARANCE = get_prompt("default_appearance.txt")

DEFAULT_ASSISTANT_MODEL = SETTINGS.get("DEFAULT_ASSISTANT_MODEL", "Domi")

def load_user_settings(**kwargs):
    """Возвращает базовые настройки. Все важное придет от клиента."""
    return {
        "style": "realistic",
        "system_prompt": DEFAULT_USER_PROMPT,
        "assistant_name": DEFAULT_ASSISTANT_NAME,
        "assistant_title": DEFAULT_ASSISTANT_TITLE,
        "assistant_appearance": DEFAULT_ASSISTANT_APPEARANCE,
        "assistant_model": DEFAULT_ASSISTANT_MODEL,
        "character_lora": DEFAULT_ASSISTANT_MODEL,
        "kb_id": DEFAULT_KB_ID
    }

def save_user_settings(ctx: UserContext):
    """No-op. Настройки теперь живут на клиенте."""
    pass

def get_context_by_account(account_id: str = "", storage: str = "", force_reload: bool = False, **kwargs) -> UserContext:
    """Создает контекст на лету. Все данные поступают от клиента в контексте без запросов к шлюзу."""
    default_owner = node_owner()
    user_id = kwargs.get("user_id") or kwargs.get("name") or kwargs.get("username") or default_owner or (f"user_{account_id[:8]}" if account_id else "user")
    return UserContext(
        type="omd",
        user_id=user_id,
        settings=load_user_settings(),
        history=[],
        omd_key=account_id or "",
        storage=storage or "",
    )

def get_user_info_from_token(account_id: str) -> dict | None:
    """Legacy stub. Все данные пользователя теперь приходят напрямую в контексте запроса."""
    return None
