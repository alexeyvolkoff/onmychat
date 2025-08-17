from dataclasses import dataclass
import os
import json
import requests

# Файл для сохранения связей
BINDINGS_FILE = "user_data/bindings.json"

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
        return UserContext(type="omd", user_id=binding["username"])
    else:
        return UserContext(type="temp", user_id=str(telegram_id))


def get_context_by_account(account_id: str) -> UserContext:
    """Вернуть контекст пользователя по account_id (omd_key).
       Если нет в bindings, пробуем спросить у OMD.
    """
    if account_id in bindings["by_account"]:
        binding = bindings["by_account"][account_id]
        return UserContext(type="omd", user_id=binding["username"])

    # если не найден — пробуем запросить у OMD
    try:
        url = "https://onmydisk.net/userinfo"
        data = {"action": "getUserInfo", "session_id": account_id}
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
        user_info = response.json()
        if user_info.get("valid", False):
            username = user_info["username"]
            # сохраняем в биндинги только в by_account (телеграма нет)
            bindings["by_account"][account_id] = {"telegram_id": None, "username": username}
            save_bindings()
            return UserContext(type="omd", user_id=username)
    except Exception as e:
        print(f"[bindings] get_context_by_account fetch error: {e}")

    # если ничего не получилось — временный контекст
    return UserContext(type="temp", user_id=f"temp_{account_id}")


def bind(telegram_id: int, account_id: str) -> UserContext:
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

    bindings["by_telegram"][telegram_id] = {"account_id": account_id, "username": username}
    bindings["by_account"][account_id] = {"telegram_id": telegram_id, "username": username}
    save_bindings()

    return UserContext(type="omd", user_id=username)


