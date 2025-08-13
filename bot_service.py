from user_context import get_context
import core_service

async def ask(telegram_id, query):
    ctx = get_context(telegram_id)
    return await core_service.perform_prompt(ctx.user_id, core_service.load_user_settings(ctx.user_id),
                                            "Answer with known facts only", query, is_rag=True)

async def chat(telegram_id, message):
    ctx = get_context(telegram_id)
    return await core_service.perform_prompt(ctx.user_id, core_service.load_user_settings(ctx.user_id),
                                            "Respond helpfully", message)

async def generate_character_image(telegram_id, prompt):
    ctx = get_context(telegram_id)
    return await core_service.generate_character_image(ctx, prompt)

async def generate_general_image(telegram_id, prompt):
    ctx = get_context(telegram_id)
    return await core_service.generate_general_image(ctx, prompt)

async def recognize_image(telegram_id, img_source, prompt=""):
    ctx = get_context(telegram_id)
    return await core_service.recognize_image(ctx, img_source, prompt)

async def import_doc(telegram_id, url, collection="user"):
    ctx = get_context(telegram_id)
    return await core_service.import_doc(ctx, url, collection)

def memorize(telegram_id, text):
    ctx = get_context(telegram_id)
    return core_service.memorize(ctx, text)

def reset_history(telegram_id):
    ctx = get_context(telegram_id)
    core_service.chat_histories[ctx.user_id] = []

def set_nsfw(telegram_id, enabled: bool):
    ctx = get_context(telegram_id)
    s = core_service.load_user_settings(ctx.user_id)
    s["nsfw"] = enabled
    core_service.save_user_settings(ctx.user_id, s)

def set_style(telegram_id, style: str):
    ctx = get_context(telegram_id)
    s = core_service.load_user_settings(ctx.user_id)
    s["style"] = style
    core_service.save_user_settings(ctx.user_id, s)

def set_kb(telegram_id, kb_id: str):
    ctx = get_context(telegram_id)
    s = core_service.load_user_settings(ctx.user_id)
    s["kb_id"] = kb_id
    core_service.save_user_settings(ctx.user_id, s)

def set_omd_key(telegram_id, key: str):
    ctx = get_context(telegram_id)
    s = core_service.load_user_settings(ctx.user_id)
    s["omd_key"] = key
    core_service.save_user_settings(ctx.user_id, s)

def get_settings(telegram_id):
    ctx = get_context(telegram_id)
    return core_service.load_user_settings(ctx.user_id)

