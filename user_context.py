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
    "by_account": {},
    "profiles": {}
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

def load_user_settings(user_id, omd_key=None, storage=None) :
    # Check cache first
    if user_id in bindings["profiles"]:
        #logging.info(f"[DEBUG] Cache hit for {user_id}: {bindings['profiles'][user_id].get('nsfw')}")
        profile = bindings["profiles"][user_id]
        if "assistant_appearance" not in profile:
             profile["assistant_appearance"] = DEFAULT_ASSISTANT_APPEARANCE
        return profile
    #ogging.info(f"[DEBUG] Cache miss for {user_id}")

    path = f"{USER_DATA_DIR}/{user_id}/settings.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    else:
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

    # Try to load from storage if configured
    if storage and omd_key:
        remote_settings = fetch_json_from_storage(
            omd_key,
            storage,
            "settings.json"
        )
        if remote_settings:
            logging.info(f"Loaded settings from storage for user {user_id}")
            
            # Ensure defaults for remote settings too
            if "assistant_appearance" not in remote_settings:
                remote_settings["assistant_appearance"] = DEFAULT_ASSISTANT_APPEARANCE
            
            # Update local cache only if NOT using remote storage exclusively
            if not (storage and omd_key):
                try:
                    os.makedirs(f"{USER_DATA_DIR}/{user_id}", exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(remote_settings, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logging.warning(f"Failed to update local settings cache: {e}")
            
            # Update memory cache
            bindings["profiles"][user_id] = remote_settings
            return remote_settings
        else:
             # If remote settings are missing or invalid, upload local settings
             logging.info(f"Remote settings not found or invalid, uploading local settings for user {user_id}")
             try:
                upload_data_to_storage(
                    omd_key,
                    storage,
                    "settings.json",
                    settings,
                    "application/json"
                )
             except Exception as e:
                logging.warning(f"Failed to upload local settings: {e}")
                # Update memory cache even if upload failed
                bindings["profiles"][user_id] = settings
    # Update memory cache
    bindings["profiles"][user_id] = settings
    return settings





def save_user_settings(ctx: UserContext):
    # Remove newUser flag before saving (from both memory and storage)
    if "newUser" in ctx.settings:
        ctx.settings.pop("newUser")

    # Update memory cache (with username if present)
    bindings["profiles"][ctx.user_id] = ctx.settings
    logging.info(f"[DEBUG] Updated cache for {ctx.user_id}: {ctx.settings.get('nsfw')}")

    # Prepare settings for saving (exclude username)
    settings_to_save = ctx.settings.copy()
    settings_to_save.pop("username", None)

    # Update bindings if we have omd_key (account_id)
    if ctx.omd_key and ctx.omd_key in bindings["by_account"]:
        if ctx.storage:
            bindings["by_account"][ctx.omd_key]["storage"] = ctx.storage

    # Save to local disk as cache/backup ONLY if no remote storage
    if not (ctx.storage and ctx.omd_key):
        os.makedirs(f"{USER_DATA_DIR}/{ctx.user_id}", exist_ok=True)
        path = f"{USER_DATA_DIR}/{ctx.user_id}/settings.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings_to_save, f, ensure_ascii=False, indent=2)

    # Save to storage if configured
    if ctx.storage and ctx.omd_key:
        try:
            upload_data_to_storage(
                ctx.omd_key,
                ctx.storage,
                "settings.json",
                settings_to_save,
                "application/json"
            )
            logging.info(f"Saved settings to storage for user {ctx.user_id}")
        except Exception as e:
            logging.warning(f"Failed to save settings to storage: {e}")


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
                # profiles are not saved, so we keep them empty or retain existing if any (though load overwrites)
                bindings["profiles"] = {} 

        except Exception as e:
            print(f"[bindings] Load error: {e}")
            bindings = {"by_telegram": {}, "by_account": {}, "profiles": {}}




def get_context(telegram_id: int) -> UserContext:
    """Вернуть контекст пользователя по его telegram_id."""
    if telegram_id in bindings["by_telegram"]:
        binding = bindings["by_telegram"][telegram_id]
        account_id = binding.get("account_id")
        
        # Try to find storage in account binding
        storage = None
        if account_id and account_id in bindings["by_account"]:
             storage = bindings["by_account"][account_id].get("storage")

        settings = load_user_settings(binding["username"], omd_key=account_id, storage=storage)
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings, history=[], omd_key=account_id, storage=storage)
    else:
        settings = load_user_settings(str(telegram_id))
        ctx = UserContext(type="temp", user_id=str(telegram_id), settings=settings, history=[])
    return  ctx



def get_context_by_account(account_id: str, storage: str = "") -> UserContext:
    """Вернуть контекст пользователя по account_id (omd_key).
       Если нет в bindings, пробуем спросить у OMD.
    """

    if account_id in bindings["by_account"]:
        binding = bindings["by_account"][account_id]
        # Prioritize storage from binding if set, otherwise use passed storage
        if not storage:
            storage = binding.get("storage")
        else:
             # Update storage in binding if passed
             binding["storage"] = storage

        settings = load_user_settings(binding["username"], omd_key=account_id, storage=storage)
        #logging.info(f"Loading settings for user: {binding["username"]}, NSFW: {settings.get("nsfw", False)}")
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings, history=[], omd_key=account_id, storage=storage)
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
            
            bindings["by_account"][account_id] = {"telegram_id": None, "username": username, "storage": storage}
            
            settings = load_user_settings(username, storage=storage, omd_key=account_id)
            settings["username"] = displayname
            ctx = UserContext(type="omd", user_id=username, settings=settings, history=[], omd_key=account_id, storage=storage)
            return ctx
    except Exception as e:
        logging.warning(f"Unbound OMD key: {account_id}")

    # если ничего не получилось — временный контекст
    user_id=f"temp_{account_id}"
    
    if storage:
         bindings["by_account"][account_id] = {"telegram_id": None, "username": user_id, "storage": storage}

    settings = load_user_settings(user_id, storage=storage, omd_key=account_id)
    ctx = UserContext(type="temp", user_id=user_id, settings=settings, history=[], omd_key=account_id, storage=storage)
    return ctx



def create_profile(ctx: "UserContext", omd_key: str, storage: str) -> dict:
    """
    Создаёт начальную структуру профиля пользователя в его OMD-хранилище:
    - {storage}/onmychat/vecs
    - {storage}/onmychat/chats
    - {storage}/onmychat/generated

    ctx      — контекст пользователя (мы обновим ctx.settings["storage"])
    omd_key  — авторизационный ключ (token)
    storage  — базовый путь в OMD, например: "storage/user123"
    """
    # Сохраняем выбранное хранилище в контекст
    ctx.storage = storage
    

    headers = {
        "authorization": f"token:{omd_key}",
        "Content-Type": "application/json"
    }

    # Список папок, которые нужно создать
    folders = [f"{storage}/vecs", f"{storage}/chats", f"{storage}/generated"]

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
            # Можно залогировать или выбросить исключение
            logging.warning(f"Error while creating profile dir: {e}")
            results[folder] = {"error": str(e)}
    
    # Update context with new storage and key
    ctx.storage = storage
    ctx.omd_key = omd_key
    ctx.settings["name"] = "User"
    save_user_settings(ctx)
    return results


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
    return ctx

