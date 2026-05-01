from dataclasses import dataclass
import os
import json
import requests
import logging



from utils import upload_data_to_storage, fetch_json_from_storage
from config import USER_DATA_DIR
from config import SETTINGS

logging.basicConfig(level=logging.INFO)

# Дефольные настройки пользователя
DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]
DEFAULT_SYSTEM_PROMPT_FILE = SETTINGS["DEFAULT_SYSTEM_PROMPT_FILE"]
DEFAULT_ASSISTANT_NAME = SETTINGS["DEFAULT_ASSISTANT_NAME"]
DEFAULT_ASSISTANT_TITLE = SETTINGS["DEFAULT_ASSISTANT_TITLE"]
GATEWAY_URL = SETTINGS["GATEWAY_URL"]


# Файл для сохранения связей
BINDINGS_FILE = f"{USER_DATA_DIR}/bindings.json"

# Структура биндингов:
# {
#   "by_telegram": { telegram_id: { "account_id": str, "username": str } },
#   "by_account": { account_id: { "telegram_id": int, "username": str } }
# }
bindings = {
    "by_telegram": {},
    "by_account": {}
}


@dataclass
class UserContext:
    type: str   # "omd" или "temp"
    user_id: str  # либо account_id/username, либо temp_id
    settings: dict
    history: dict
    omd_key: str = ""
    storage: str = ""


def get_prompt(filename):
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    try:
        with open(os.path.join(prompts_dir, filename), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Failed to load prompt {filename}: {e}")
        return ""

DEFAULT_UNONBOARDED_PROMPT = get_prompt("default.txt")
DEFAULT_USER_PROMPT = get_prompt("default_user.txt")
DEFAULT_ASSISTANT_APPEARANCE = get_prompt("default_appearance.txt")

def load_user_settings(user_id, omd_key=None, storage=None, force_reload=False) :
    # No disk or memory caching allowed. Always return fresh default settings.
    # The endpoint is responsible for applying client-provided overrides.
    settings = {
        "nsfw": False,
        "style": "realistic",
        "system_prompt": DEFAULT_USER_PROMPT,
        "assistant_name": DEFAULT_ASSISTANT_NAME,
        "assistant_title": DEFAULT_ASSISTANT_TITLE,
        "assistant_appearance": DEFAULT_ASSISTANT_APPEARANCE,
        "assistant_model": "Domi",
        "kb_id": DEFAULT_KB_ID
    }

    # Ensure defaults for new fields
    if "assistant_appearance" not in settings:
         settings["assistant_appearance"] = DEFAULT_ASSISTANT_APPEARANCE
    
    return settings


def save_user_settings(ctx: UserContext):
    # No newUser flag in settings
    if "newUser" in ctx.settings:
        ctx.settings.pop("newUser")

    # Update bindings if we have omd_key (account_id)
    if ctx.omd_key and ctx.omd_key in bindings["by_account"]:
        save_bindings()

    # Legacy: saving settings.json to local disk removed.
    # Memory caching (bindings["profiles"]) removed.

    # OrbitDB Sync: No longer uploading settings.json to remote storage.


def create_profile(ctx: "UserContext", omd_key: str, storage: str) -> dict:
    """
    Создаёт начальную структуру профиля пользователя в его OMD-хранилище.
    Legacy folders (chats, vecs, generated) are no longer created.
    """
    if not storage or storage in ["undefined", "null"]:
        logging.warning("Skipping profile creation: invalid or empty storage")
        return {}

    # Сохраняем выбранное хранилище в контекст
    ctx.storage = storage

    headers = {
        "authorization": f"token:{omd_key}",
        "Content-Type": "application/json"
    }

    # Только базовая папка хранилища
    folders = [storage]

    results = {}
    for folder in folders:
        payload = {"action": "createFolder", "newPath": folder}
        logging.info(f"Creating profile dir: {folder}")

        try:
            resp = requests.post(f"{GATEWAY_URL}/{folder}", headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            logging.info(f"Dir info: {resp.json()}")
            results[folder] = resp.json()
        except Exception as e:
            logging.warning(f"Error while creating profile dir: {e}")
            results[folder] = {"error": str(e)}
    
    # Update context
    ctx.storage = storage
    ctx.omd_key = omd_key
    ctx.settings["name"] = "User"
    save_user_settings(ctx)
    return results


def load_bindings():
    """Загрузить биндинги из файла."""
    global bindings
    if os.path.exists(BINDINGS_FILE):
        try:
            with open(BINDINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Приводим ключи telegram_id к int
                if "by_telegram" in data:
                    data["by_telegram"] = {int(k): v for k, v in data["by_telegram"].items()}
                bindings["by_telegram"] = data.get("by_telegram", {})
                bindings["by_account"] = data.get("by_account", {})

        except Exception as e:
            print(f"[bindings] Load error: {e}")
            bindings = {"by_telegram": {}, "by_account": {}, "profiles": {}}

def save_bindings():
    """Сохранить биндинги в файл."""
    try:
        # Avoid saving profiles to disk in bindings file (they are separate files/remote)
        data_to_save = {
            "by_telegram": bindings["by_telegram"],
            "by_account": bindings["by_account"]
        }
        with open(BINDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"[bindings] Save error: {e}")




def get_context(telegram_id: int) -> UserContext:
    """Вернуть контекст пользователя по его telegram_id."""
    if telegram_id in bindings["by_telegram"]:
        binding = bindings["by_telegram"][telegram_id]
        account_id = binding.get("account_id")
        
        settings = load_user_settings(binding["username"], omd_key=account_id)
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings, history=[], omd_key=account_id)
    else:
        settings = load_user_settings(str(telegram_id))
        ctx = UserContext(type="temp", user_id=str(telegram_id), settings=settings, history=[])
    return  ctx



def get_context_by_account(account_id: str, storage: str = "", force_reload: bool = False) -> UserContext:
    logging.info(f"get_context_by_account: account_id={account_id[:10] if account_id else 'None'}... storage={storage}")
    """Вернуть контекст пользователя по account_id (omd_key).
       Если нет в bindings, пробуем спросить у OMD.
    """

    if not account_id:
        user_id = "temp_anon"
        settings = load_user_settings(user_id, storage=storage, omd_key=account_id, force_reload=force_reload)
        ctx = UserContext(type="temp", user_id=user_id, settings=settings, history=[], omd_key=account_id, storage=storage)
        if not ctx.storage and ctx.settings.get("defaultStorage"):
            ctx.storage = ctx.settings["defaultStorage"]
        return ctx

    if account_id in bindings["by_account"]:
        binding = bindings["by_account"][account_id]
        
        settings = load_user_settings(binding["username"], omd_key=account_id, storage=storage, force_reload=force_reload)
        #logging.info(f"Loading settings for user: {binding["username"]}, NSFW: {settings.get("nsfw", False)}")
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings, history=[], omd_key=account_id, storage=storage)
        if not ctx.storage and ctx.settings.get("defaultStorage"):
            ctx.storage = ctx.settings["defaultStorage"]
        return ctx 
    
    # если не найден — пробуем запросить у OMD
    try:
        url = f"{GATEWAY_URL}/userinfo"
        data = {"action": "getUserInfo", "session_id": account_id}
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        user_info = response.json()
        if user_info.get("valid", False):
            username = user_info["user"]
            displayname = user_info.get("displayname", username)
            
            # Check for defaultStorage in profile
            profile_storage = user_info.get("defaultStorage")
            if profile_storage:
                storage = profile_storage
            
            bindings["by_account"][account_id] = {"telegram_id": None, "username": username}
            save_bindings()
            
            settings = load_user_settings(username, storage=storage, omd_key=account_id, force_reload=force_reload)
            settings["username"] = displayname
            if "language" in user_info:
                settings["language"] = user_info["language"]
            ctx = UserContext(type="omd", user_id=username, settings=settings, history=[], omd_key=account_id, storage=storage)
            if not ctx.storage and ctx.settings.get("defaultStorage"):
                ctx.storage = ctx.settings["defaultStorage"]
            return ctx
    except Exception as e:
        logging.warning(f"Unbound OMD key: {account_id}")

    # если ничего не получилось — временный контекст
    user_id = f"temp_{account_id}" if account_id else "temp_anon"
    
    if storage and account_id:
         bindings["by_account"][account_id] = {"telegram_id": None, "username": user_id}
         save_bindings()

    settings = load_user_settings(user_id, storage=storage, omd_key=account_id, force_reload=force_reload)
    ctx = UserContext(type="temp", user_id=user_id, settings=settings, history=[], omd_key=account_id, storage=storage)
    if not ctx.storage and ctx.settings.get("defaultStorage"):
        ctx.storage = ctx.settings["defaultStorage"]
    return ctx





def bind(ctx: UserContext, account_id: str) -> UserContext:
    """
    Привязать телеграм-аккаунт к OMD-аккаунту.
    """
    # check user
    url = "https://onmydisk.net/userinfo"
    data = {"action": "getUserInfo", "session_id": account_id}
    response = requests.post(url, json=data)
    response.raise_for_status()
    user_info = response.json()
    if not user_info.get("valid", False):
        raise ValueError("Invalid OMD account")
    username = user_info["user"]
    displayname = user_info.get("displayname", username)
    telegram_id = str(ctx.user_id)

    bindings["by_telegram"][telegram_id] = {"account_id": account_id, "username": username}
    ctx.type="omd" 
    ctx.user_id=username
    ctx.settings["omd_key"] = account_id
    ctx.settings["username"] = displayname
    save_user_settings(ctx)
    save_bindings()
    return ctx

