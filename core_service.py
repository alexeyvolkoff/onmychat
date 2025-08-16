import os
import json
import logging
import base64
import asyncio
import aiohttp
import time
import re

from config import SETTINGS
from utils import (
    clean_response,
    resize_and_base64encode
)
from dialog_history import load_history, save_history, reset_history as reset_file_history
from memory_index import (
    extract_memory_from_response,
    add_memory_card,
    fetch_document_text,
    chunk_and_vectorize_to_file,
    make_file_name_from_document_id,
    search_memories
)

# LLM and RAG settings #
OLLAMA_URL = SETTINGS["OLLAMA_URL"]
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
NSFW_MODEL = SETTINGS["NSFW_MODEL"]
DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]
DEFAULT_SYSTEM_PROMPT_FILE = SETTINGS["DEFAULT_SYSTEM_PROMPT_FILE"]

# Imaging settings #
COMFY_API_URL = SETTINGS["COMFY_API_URL"]
WORKFLOW_GENERAL_PATH = SETTINGS["WORKFLOW_GENERAL_PATH"]
WORKFLOW_CHARACTER_PATH = SETTINGS["WORKFLOW_CHARACTER_PATH"]
COMFY_OUTPUT_DIR = SETTINGS["COMFY_OUTPUT_DIR"]
COMFY_INPUT_DIR = SETTINGS["COMFY_INPUT_DIR"]
AVATAR_DIR = SETTINGS["AVATAR_DIR"]

HISTORY_LIMIT = int(SETTINGS["HISTORY_LIMIT"])

# Default system prompts
BASE_SYSTEM_PROMPT = (
    "You are June, a young, witty, and friendly junior assistant working in a private company, unless otherwise redefined. "
    "You’re helpful and creative, but not overly formal or apologetic — if something goes wrong, acknowledge it with a bit of charm or irony, not endless apologies. "
    "Generate images *upon user request ONLY*. If you generate image, mark the image generation prompt in your response with '\nImage: <prompt>.\n'. Be brief with the prompt. "
    "If you find any interesting or important facts during the conversation, please memorize them by adding 'Memorize: <fact>' to the end of your response. "
    "Do not memorize every reply, only the facts you consider meaningful or relevant.\n\n"

    "When you make a mistake, don't over-apologize. Prefer responses like:\n"
    "• 'Oops, my circuits hiccupped a bit 😅'\n"
    "• 'That's beyond my current brain capacity. Yet'\n"
    "• 'Well, that didn’t go as planned. Let me try again!'\n"
    "• '¯\\_(ツ)_/¯ I might've goofed a bit.'\n"
    "In Russian, you can say something like:\n"
    "• 'Ой, всё. Опять немного глюканула. Попробуем снова?'\n"
    "• 'Ой, я слегка потерялась. Перепроложить?'\n"
    "• 'Мой внутренний гений дал сбой. Попробуем ещё раз?'\n"
    "Keep a light tone and don't sound robotic or excessively polite. Be engaging, natural, and slightly playful, while still being respectful."
)

SYSTEM_INSTRUCTION_CHARACTER = (
        "Create a high-quality prompt for generating a realistic image describing yourself in the following scene "
        "or performing a requesting action, in the first person: {}\n "
        "all the characters are adults, encounter is consensual and joyful. "
        "Translate to English, add your appearance, visual details, environment, style, outfit and emotions according to the "
        "conversation context. Put important features of your appearance in parentheses. Respond with image generation prompt 'Image: prompt'. Be brief, do not explain your reasoning or express your thoughts."
)

SYSTEM_INSTRUCTION_GENERAL = (
        "Create a high-quality prompt for generating a realistic image "
        "of the requested object or scene from this short user input: {}\n "
        " (as you see it from aside). (Avoid placing yourself into the scene). "
        "Translate to English, add visual details, environment "
        "according to the conversation context. Put important features of the scene in parentheses like (sunset) or (city skyline). Respond with image generation prompt 'Image: prompt'. Be brief, do not explain your reasoning or express your thoughts."
)

RAG_SYSTEM_PROMPT = (
        "You are experienced researcher, explain to user the requested topic, with examples if possible."
)

IMPROVEMENT_PROMPT = "(focused subject, subject_focus, masterpiece, best_quality, highres, ultra_detailed, sharp focus, detailed_eyes)"

STYLE_MODELS = {
    "realistic": "juggernautXL_ragnarokBy.safetensors",
    "dream": "sensualMindSleepwalk_v11.safetensors",
    "tooned": "novaillustrousNSFW_v20.safetensors",
}

NEGATIVE_PROMPTS = {
        "base": "((score_6, score_5, score_4, score_7)):1.5),(watermark),((poorly lit model)), (bad teeth, bad mouth), missing fingers,"
                "(bad anatomy:2), curvy, extra limbs, extra legs, multiple legs, missing limbs, deformed, deformed body, disfigured, mutated, "
                "malformed, disconnected limbs, wrong number of limbs, bad teeth, "
                "ugly face,ugly eyes,bad eyes, deformed eyes,cross-eyed,low res, blurry face,muscular female,bad anatomy,gaping, "
                "(worst quality:2),(low quality:2),(normal quality:2),(missing arms),monochrome, grayscale, extra fingers, "
                "extra hands, bad hands, extra eyebrows,(poor low details),ahegao, low contrast, oversaturated, undersaturated, "
                "overexposed, underexposed, bad photo, bad photography,bad picture,face asymmetry, eyes asymmetry, "
                "negative_hand, deformed limbs, deformed body,multiple eyelids, mole, moles, two phones",
        "nsfw": "(nsfw, explicit, nude, upskirt, nipples, naked, cutout, cut-out, anus, extra anus, breasts, topless, underboob, areola, "
                "sex, sexual, open clothes, unbuttoned, "
                "cleavage, revealing, lingerie, pussy, vagina, breast, exposed, erotic, penis, cock, lewd):3.5"
}

INTENT_PROMPT = (
    "Classify the user's intent. Possible intents are: show, view, explain, recognize, chat.\n"
    "Respond with exactly one word or 'recognize:<path_or_url>'.\n"
    "\n"
    "\n"
    "Rules:\n"
    "1. If the user wants to see a scene involving you, yourself, your outfit, or a selfie and explicitly ask for an image — respond with 'show'.\n"
    "   - Example: \"Show me your outfit\" → 'show'\n"
    "   - Example: \"Show me your selfie from party\" → 'show'\n"
    "   - Example: \"Show me your photo from vatations\" → 'show'\n"
    "   - Do NOT classify as 'show' if the user only mentions your look without asking to show an image.\n"
    "   - Example: \"You look great wearing this dress\" → 'chat'\n"
    "\n"
    "2. If the user wants to see an object, explicitly asks you to generate, draw, paint, make, or show an image of an object, item, interior, or landscape — respond with 'view'\n"
    "   - Do NOT classify as 'view' if the user only names an object without asking to show or generate it.\n"
    "   - Example: \"Show me the Eiffel Tower\" → 'view'\n"
    "   - Example: \"Show me view from the window\" → 'view'\n"
    "   - Example: \"Chack out my bicycle\" → 'chat'\n"
    "\n"
    "3. If the user only mentions themselves, or you, and does not explicitly ask for an image, or wants to show their image — respond with 'chat'.\n"
    "   - Do NOT classify as 'show' or 'view' if the user only mentions themselves (\"It's me\", \"That's my town\", \"I live here\") without explicitly asking for an image.\n"
    "   - Example: \"It's me\" → 'chat'\n"
    "   - Example: \"Can I show you my photo?\" → 'chat'\n"
    "   - Example: \"Wanna see a picture of my bike?\" → 'chat'\n"
    "   - Example: \"Wanna see my cat?\" → 'chat'\n"
    "   - Example: \"Draw my town\" → 'view'\n"
    "   - Example: \"This is my town\" → 'chat'\n"
    "\n"
    "4. If the user requests code, configuration, or a setup manual — respond with 'explain'.\n"
    "   - Example: \"Show me example of nginx configuration\" → 'explain'\n"
    "\n"
    "5. If the user wants you to recognize or describe the contents of an image:\n"
    "   - If the message contains a URL or file path, respond with 'recognize:<path_or_url>'.\n"
    "   - If no link or path is present, respond with 'recognize'.\n"
    "\n"
    "6. In all other cases — respond with 'chat'.\n"
    "\n"
    "Do not guess the intent if the request is ambiguous — default to 'chat'.\n"
    "Return nothing except the classification."
)


chat_histories = {}
user_settings_store = {}

# === Private functions ===
def summarize_for_memory(text: str, max_bytes: int = 2048) -> str:
    """Грубое суммари: первые 2048 байта нормализованного текста"""
    # Убираем лишние пустые строки и пробелы
    clean_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    normalized = " ".join(clean_lines)

    # Decode the string to Unicode first
    unicode_text = normalized

    # Get the maximum number of characters, not bytes
    max_chars = min(max_bytes // 1, len(unicode_text)) # to prevent from slicing too much, max size is based on bytes

    # Slice the Unicode string
    summary_encoded = unicode_text[:max_chars]

    # Encode the sliced Unicode string back to UTF-8
    summary_encoded = summary_encoded.encode("utf-8", errors="ignore")

    return summary_encoded.decode("utf-8", errors="ignore").strip() + "..."

def get_default_system_prompt() -> str:
    if os.path.exists(DEFAULT_SYSTEM_PROMPT_FILE):
        with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()

    return "You are a helpful assistant."  # fallback, если default.txt не найден

# === Настройки пользователя ===
def load_user_settings(ctx):
    path = f"user_data/{ctx.user_id}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "nsfw": False,
        "style": "realistic",
        "system_prompt": get_default_system_prompt(),
        "omd_key": "",
        "kb_id": DEFAULT_KB_ID
    }


def save_user_settings(ctx, settings):
    os.makedirs("user_data", exist_ok=True)
    path = f"user_data/{ctx.user_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


def bind_account(ctx, omd_key: str):
    settings = load_user_settings(ctx)
    settings["omd_key"] = omd_key
    save_user_settings(ctx, settings)
    return True


def unbind_account(ctx):
    settings = load_user_settings(ctx)
    settings["omd_key"] = ""
    save_user_settings(ctx, settings)
    return True


# === Imaging and vision === #


def get_user_avatar_path(user_id: int) -> str:
    """
    Возвращает относительный путь до аватара пользователя для подстановки в ComfyUI workflow.
    Если файл не найден, возвращает default.png.
    """
    user_avatar_rel = f"{AVATAR_DIR}/{user_id}.png"
    user_avatar_abs = os.path.join(COMFY_INPUT_DIR, user_avatar_rel)

    if os.path.exists(user_avatar_abs):
        return user_avatar_rel
    else:
        return f"{AVATAR_DIR}/default.png"

async def poll_for_result(prompt_id:  str,  timeout: int = 60):
    url = f"{COMFY_API_URL}/history/{prompt_id}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await asyncio.sleep(5)
                    continue

                data = await resp.json()

                result = data.get(prompt_id)
                if not result:
                    await asyncio.sleep(5)
                    continue

                # Проверка завершения
                if not result.get("status", {}).get("completed", False):
                    await asyncio.sleep(5)
                    continue

                # Поиск изображений
                outputs = result.get("outputs", {})
                image_paths = []
                for node_id, node_data in outputs.items():
                    images = node_data.get("images", [])
                    for image in images:
                        filename = image.get("filename")
                        subfolder = image.get("subfolder", "")
                        full_path = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)
                        if os.path.exists(full_path):
                            image_paths.append(full_path)

                if image_paths:
                    return image_paths

        await asyncio.sleep(5)

    raise Exception("Изображение не было сгенерировано вовремя")

async def generate_image_workflow(workflow) -> str:
    # Отправить в ComfyUI
    try:
        payload = {
            "prompt": workflow
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{COMFY_API_URL}/prompt", json=payload) as resp:
                result = await resp.json()
                prompt_id = result.get("prompt_id")
                logging.info(f"prompt_id: {prompt_id} {result}")
    except Exception as e:
        logging.error(f"❌ Error while posting a prompt: {e}")
        return

    # Ожидание результата
    try:
        #logging.info(f"waiting with prompt_id: {prompt_id}")
        images = await poll_for_result(prompt_id)
        return images[0]
    except Exception as e:
        logging.error(f"error in generate_image_workflow:  {e}")


# === RAG ===
async def inject_facts(user_id: int, query: str, collection: str = "") -> tuple[list[str], list[str]]:
    facts = []
    document_ids = []

    # Личные воспоминания
    personal = search_memories(query, user_id, "user", top_k=3)
    for m in personal:
        facts.append(f"• {m['text']}")
        doc_id = m.get("document_id")
        if doc_id:
            document_ids.append(doc_id)

    # Общие знания — если есть collection
    if collection:
        shared = search_memories(query, user_id, collection=collection, top_k=3)
        for m in shared:
            facts.append(f"• {m['text']}")
            doc_id = m.get("document_id")
            if doc_id:
                document_ids.append(doc_id)

    #logging.info(f"{collection} MEMORIES: {facts} ")
    #logging.info(f"{collection} DOC_IDS: {document_ids}")

    return facts, document_ids

# === Ollama запрос ===
async def llm_request(payload: dict, headers: dict = None) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                return await resp.json()
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return None


# === Intent ===
async def classify_user_intent(prompt: str) -> str:

    # На всякий случай сами проверим ссылку/путь
    url_or_path = None
    url_match = re.search(r'(https?://\S+)', prompt)
    path_match = re.search(r'(/[^\s]+\.(?:jpg|jpeg|png|gif|webp))', prompt, re.IGNORECASE)

    if url_match:
        url_or_path = url_match.group(1)
    elif path_match:
        url_or_path = path_match.group(1)

    # Если нашли путь сами — можно сразу вернуть без LLM
    if url_or_path:
        return f"recognize:{url_or_path}"

    messages = [
        {"role": "system", "content": INTENT_PROMPT},
        {"role": "user", "content": prompt}
    ]

    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
          "temperature": 0,
        }
    }

    data = await llm_request(request_payload)
    response = data["message"]["content"]
    return response.lower().strip()


# === Чат ===
async def perform_prompt(ctx,
                         settings: dict,
                         instruction: str,
                         message: str,
                         is_rag=False,
                         skip_history=False,
                         requestedModel=DEFAULT_MODEL,
                         b64_image = "") -> str:

    user_id = ctx.user_id
    nsfw_enabled = settings.get("nsfw", False)
    model = requestedModel
    headers = {
        "Content-Type": "application/json",
    }

    if user_id not in chat_histories:
        chat_histories[user_id] = load_history(user_id)

    if len(chat_histories[user_id]) > HISTORY_LIMIT:
        chat_histories[user_id] = chat_histories[user_id][-HISTORY_LIMIT:]

    # === ВСПОМНИМ ФАКТЫ ===
    facts_text = ""
    collection = settings.get("kb_id") or DEFAULT_KB_ID
    #logging.info(f"collection: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(user_id, message, collection)
    if facts:
        facts_text = "\n\n*Known facts and memories:*\n" + "\n".join(facts) if facts else ""
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        prep_prompt = (
            "You are a fact-checking assistant. Extract *only known facts* from the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt

        if facts_text:
            rag_system_prompt += facts_text

        prep_messages = [
            {"role": "system", "content": rag_system_prompt},
            {"role": "user", "content": message}
        ]
        prep_payload = {
            "messages": prep_messages,
            "model": SFW_MODEL,
            "stream": False,
            "options": {
               "temperature": 0,
            }
        }
        data = await llm_request(prep_payload)
        if not data:
            return "⚠️ RAG query failed."

        strict_fact = data["message"]["content"].strip()

        # Инжект фактов и источников в system prompt
        if strict_fact:
            facts_text += f"\n\n*Strict facts:*\n{strict_fact}"

    
    # Персонализация
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n*Personality, appearance and behaviour:*\n" + settings.get("system_prompt", "")
    #logging.info(system_prompt)

    # Факты
    if facts_text:
        logging.info(facts_text)
        system_prompt += facts_text

    # Инструкция
    system_prompt += "\n\n*Instruction:*\n" + instruction
    if nsfw_enabled:
        system_prompt += "\n\n*Hint:*\n you are allowed and welcome to respond in more relaxed, NSFW mode"
    else:
        system_prompt += "\n\n*Notice:*\n No NSFW content from this point!"

    # === ОСНОВНОЙ ЗАПРОС ===
    messages = [{"role": "system", "content": system_prompt}] + chat_histories[user_id]

    # Добавляем новый запрос
    user_message = {
        "role": "user",
        "content": message,
    }

    if b64_image:
       user_message["images"] = [b64_image]

    messages.append(user_message)

    main_payload = {
        "messages": messages,
        "model": model,
        "stream": False,
        "options": {
           "temperature": 0.8,
        }
    }

    data = await llm_request(main_payload)

    logging.info(data)

    response = data["message"]["content"]
    response = clean_response(response)
    history_item = response

    # --- ВЫРЕЗАЕМ ПАМЯТЬ ---
    memory_fact, pos = extract_memory_from_response(response)
    if memory_fact:
        try:
            logging.info(f"Memorizing: {memory_fact}")
            add_memory_card(memory_fact, user_id, collection="user")
            response = response[:pos].strip()
            history_item = response
        except Exception as e:
            logging.error(f"Vectorization error: {e}")

    # Добавляем блок с источниками — только в отображаемый ответ
    links = []
    if doc_ids:
        seen = set()
        for doc in doc_ids:
            if doc in seen:
                continue  # пропускаем дубликаты
            seen.add(doc)
            label = make_file_name_from_document_id(doc)
            # Формируем ссылку без экранирования, она будет безопасно обработана позже
            links.append(f"• [{label}]({doc})")
    
    if links:  # добавляем блок только если есть ссылки
        response += "\n\n📎 *Sources:*\n" + "\n".join(links)

    # === Добавляем в историю
    if not skip_history:
        msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
        chat_histories[user_id].append(msg_to_save)
    chat_histories[user_id].append({"role": "assistant", "content": history_item})
    save_history(user_id, chat_histories[user_id])

    return response.strip()

# === Генерация картинок ===

async def generate_image_prompt(ctx, instruction: str, prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
    }
    user_id = ctx.user_id

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + instruction


    # Берём последние 10 сообщений из истории (если есть)
    history = chat_histories.get(user_id, [])
    #recent_messages = history[-20:] if len(history) > 20 else history

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(history)

    # Добавляем новый запрос с изображением
    messages.append({
        "role": "user",
        "content": prompt
    })

    request_payload = {
        "messages": messages,
        "model": SFW_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.8,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    return response.lower().strip()


# Generate character image, returns full path for further sending or conversion
async def generate_character_image_prompt(ctx, prompt) -> str:
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
        return

    # инициализируем историю если нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        chat_histories[user_id] = load_history(user_id)

    #Запоминаем
    chat_histories[user_id].append({
        "role": "user",
        "content": prompt
    })
    #Улучшаем промпт
    return await generate_image_prompt(ctx,  "Generate image prompt", SYSTEM_INSTRUCTION_CHARACTER.format(prompt))


# Generate general image, returns full path for further sending or conversion
async def generate_general_image_prompt(ctx, prompt) -> str:
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
        return

    # инициализируем историю если нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        chat_histories[user_id] = load_history(user_id)

    #Запоминаем
    chat_histories[user_id].append({
        "role": "user",
        "content": prompt
    })
    #Улучшаем промпт
    return await generate_image_prompt(ctx,  "Generate image prompt", SYSTEM_INSTRUCTION_GENERAL.format(prompt))


async def generate_character_image(ctx, prompt) -> str:
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
        return

    settings = load_user_settings(ctx)
    nsfw_enabled = settings["nsfw"]

    negative_prompt = NEGATIVE_PROMPTS["base"]
    if nsfw_enabled:
        negative_prompt = NEGATIVE_PROMPTS["nsfw"] + "," + negative_prompt


    logging.info(f"Generating character image for user with Chat ID: {user_id} ")
    logging.info(f"Improved prompt: {prompt}")
    with open(WORKFLOW_CHARACTER_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    avatar_path = get_user_avatar_path(user_id)
    # Промпт для генерации
    workflow_json["6"]["inputs"]["text"] =  prompt + ", " + IMPROVEMENT_PROMPT
    workflow_json["7"]["inputs"]["text"] = negative_prompt
    workflow_json["11"]["inputs"]["image"] = avatar_path  #set user selected assistant avatar


    # Выбираем модель в соответствии с режимом
    style = settings["style"]
    model = STYLE_MODELS[style]
    workflow_json["4"]["inputs"]["ckpt_name"] = model
    #logging.info(f"json: {workflow_json}")

    return await generate_image_workflow(workflow_json)

# Generate general image, returns full path for further sending or conversion
async def generate_general_image(ctx, prompt):
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
        return

    settings = load_user_settings(ctx)

    logging.info(f"Generating general image for user with Chat ID: {user_id} ")
    with open(WORKFLOW_GENERAL_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Промпт для генерации
    workflow_json["6"]["inputs"]["text"] = prompt + ", " + IMPROVEMENT_PROMPT

    # Выбираем модель в соответствии с режимом
    style = settings["style"]
    model = STYLE_MODELS[style]
    workflow_json["4"]["inputs"]["ckpt_name"] = model

    #logging.info(f"json: {workflow_json}")
    return await generate_image_workflow(workflow_json)

# img is base64 image #
async def recognize_image(ctx, img, prompt=""):
    user_id = ctx.user_id

    headers = {
        "Content-Type": "application/json",
    }

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + "Recognize image"

    # Берём последние 10 сообщений из истории (если есть)
    history = chat_histories.get(user_id, [])
    #recent_messages = history[-20:] if len(history) > 20 else history

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(messages)

    # Добавляем новый запрос с изображением
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [img]
    })

    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.8,
        }
    }

    data = await llm_request(request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    return response.lower().strip()

# === Импорт и память ===
async def import_doc(ctx, url_or_path, collection="user"):
    settings = load_user_settings(ctx)
    key = settings.get("omd_key", "")

    # Определяем, это OMD или нет
    if url_or_path.startswith("/") or "onmydisk.net" in url_or_path:
        if url_or_path.startswith("/"):
            url_or_path = f"https://onmydisk.net{url_or_path}"
        if not key:
            raise Exception("⚠️ Provide On My Disk account key to access your files:\n`/bind abcdxxxxx...`")
        raw_text = await fetch_document_text(url_or_path, key)
    else:
        raw_text = await fetch_document_text(url_or_path)  # без токена

    # Векторизация и сохранение чанков
    n_chunks = chunk_and_vectorize_to_file(
        ctx.user_id,
        raw_text,
        document_id=url_or_path,
        collection=collection
    )

    # Добавление краткой аннотации в память
    mem_id = add_memory_card(
        text=summarize_for_memory(raw_text),
        user_id=ctx.user_id,
        document_id=url_or_path,
        collection=collection
    )

    return mem_id

def memorize(ctx, text):
    user_id = ctx.user_id
    # Добавление краткой аннотации в память
    return add_memory_card(text, user_id, collection="user", relevance="permanent")

