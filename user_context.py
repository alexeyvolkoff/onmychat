from dataclasses import dataclass
import os
import json
import requests

from config import USER_DATA_DIR
from config import SETTINGS


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

# === Настройки пользователя ===
def get_default_system_prompt() -> str:
    if os.path.exists(DEFAULT_SYSTEM_PROMPT_FILE):
        with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()

    return "You are a helpful assistant."  # fallback, если default.txt не найден

def load_user_settings(user_id) :
    path = f"{USER_DATA_DIR}/{user_id}/settings.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {
            "nsfw": False,
            "style": "realistic",
            "system_prompt": get_default_system_prompt(),
            "omd_key": "",
            "storage": "",
            "assistant_name": DEFAULT_ASSISTANT_NAME,
            "assistant_title": DEFAULT_ASSISTANT_TITLE,
            "kb_id": DEFAULT_KB_ID
        }


def save_user_settings(ctx: UserContext):
    os.makedirs(f"{USER_DATA_DIR}/{ctx.user_id}", exist_ok=True)
    path = f"{USER_DATA_DIR}/{ctx.user_id}/settings.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ctx.settings, f, ensure_ascii=False, indent=2)


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
                bindings = data    

        except Exception as e:
            print(f"[bindings] Load error: {e}")
            bindings = {"by_telegram": {}, "by_account": {}}



def save_bindings():
    """Сохранить биндинги в файл."""
    os.makedirs("user_data", exist_ok=True)
    try:
        # Telegram ID конвертим обратно в строки
        data = {
            "by_telegram": {str(k): v for k, v in bindings["by_telegram"].items()},
            "by_account": bindings["by_account"]
        }
        with open(BINDINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[bindings] Save error: {e}")


def get_context(telegram_id: int) -> UserContext:
    """Вернуть контекст пользователя по его telegram_id."""
    if telegram_id in bindings["by_telegram"]:
        binding = bindings["by_telegram"][telegram_id]
        settings = load_user_settings(binding["username"])
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings)
    else:
        settings = load_user_settings(telegram_id)
        ctx = UserContext(type="temp", user_id=str(telegram_id), settings=settings)
    return  ctx



def get_context_by_account(account_id: str) -> UserContext:
    """Вернуть контекст пользователя по account_id (omd_key).
       Если нет в bindings, пробуем спросить у OMD.
    """
    if account_id in bindings["by_account"]:
        binding = bindings["by_account"][account_id]
        settings = load_user_settings(binding["username"])
        ctx = UserContext(type="omd", user_id=binding["username"], settings=settings)
        return ctx 

    # если не найден — пробуем запросить у OMD
    try:
        url = "https://staging.onmydisk.net/userinfo"
        data = {"action": "getUserInfo", "session_id": account_id}
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        user_info = response.json()
        if user_info.get("valid", False):
            username = user_info["user"]
            # сохраняем в биндинги только в by_account (телеграма нет)
            bindings["by_account"][account_id] = {"telegram_id": None, "username": username}
            save_bindings()
            settings = load_user_settings(username)
            ctx = UserContext(type="omd", user_id=username, settings=settings)
            return ctx
    except Exception as e:
        print(f"[bindings] get_context_by_account fetch error: {e}")

    # если ничего не получилось — временный контекст
    settings = load_user_settings(account_id)
    ctx = UserContext(type="temp", user_id=f"temp_{account_id}", settings=settings)
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
    ctx.settings["storage"] = storage

    headers = {
        "authorization": f"token:{omd_key}",
        "Content-Type": "application/json"
    }

    # Список папок, которые нужно создать
    folders = [f"{storage}/onmychat/vecs", f"{storage}/onmychat/chats", f"{storage}/onmychat/generated"]

    results = {}
    for folder in folders:
        payload = {"action": "createFolder", "path": folder}

        try:
            resp = requests.post(GATEWAY_URL, headers=headers, json=payload, timeout=10)
            resp.raise_for_status()
            results[folder] = resp.json()
        except Exception as e:
            # Можно залогировать или выбросить исключение
            results[folder] = {"error": str(e)}
    # save settings
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
    telegram_id = str(ctx.user_id)

    bindings["by_telegram"][telegram_id] = {"account_id": account_id, "username": username}
    bindings["by_account"][account_id] = {"telegram_id": telegram_id, "username": username}
    save_bindings()
    ctx.type="omd" 
    ctx.user_id=username
    ctx.settings["omd_key"] = account_id
    save_user_settings(ctx)
    return ctx



