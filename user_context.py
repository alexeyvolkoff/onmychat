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
    group: str = ""
    groups: list = None
    omd_key: str = ""
    storage: str = ""
    private_mode: bool = False
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

def load_user_settings(**kwargs):
    """Возвращает базовые настройки. Все важное придет от клиента."""
    return {
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

    # Пытаемся получить информацию о пользователе через токен
    user_info = get_user_info_from_token(account_id)
    if user_info and user_info.get("valid"):
        username = user_info.get("user")
        group = user_info.get("group", "")
        # Могут быть и другие группы или список групп в будущем
        groups = [group] if group else []
        return UserContext(
            type="omd", 
            user_id=username, 
            group=group,
            groups=groups,
            settings=load_user_settings(), 
            history=[], 
            omd_key=account_id, 
            storage=storage
        )

    return UserContext(type="temp", user_id=f"web_{account_id[:8]}", settings=load_user_settings(), history=[], omd_key=account_id, storage=storage)

def get_user_info_from_token(account_id: str) -> dict | None:
    """Fetches the real user info from OMD gateway using the session/token."""
    import requests
    from config import SETTINGS
    gateway_url = SETTINGS.get("GATEWAY_URL", "https://onmydisk.net")
    # Correct endpoint and action for user profile as used by frontend
    url = f"{gateway_url}/" 
    data = {"action": "getProfileData", "user": ""}
    headers = {"Authorization": f"token:{account_id}"}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=5)
        response.raise_for_status()
        resp_json = response.json()
        if resp_json and "result" in resp_json and "profile" in resp_json["result"]:
            user_info = resp_json["result"]["profile"]
            # Ensure 'valid' is set if we got a profile
            if "user" in user_info:
                user_info["valid"] = True
            return user_info
    except Exception as e:
        logging.info(f"Get user info: {e}")
    return None
