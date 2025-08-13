from dataclasses import dataclass
import uuid

bindings = {}  # telegram_id -> omd_account_id

@dataclass
class UserContext:
    type: str  # "omd" или "temp"
    user_id: str  # либо account_id, либо temp_id

def get_context(telegram_id):
    if telegram_id in bindings:
        return UserContext(type="omd", user_id=bindings[telegram_id])
    else:
        return UserContext(type="temp", user_id=telegram_id)

