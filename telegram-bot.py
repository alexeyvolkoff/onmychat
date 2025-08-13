import logging
import requests
from dialog_history import load_history, save_history, reset_history as reset_file_history
from memory_index import (
    extract_memory_from_response,
    add_memory_card,
    fetch_document_text,
    chunk_and_vectorize_to_file,
    make_file_name_from_document_id,
    search_memories
)

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ChatAction
from telegram.constants import ParseMode
from telegram.helpers import escape_markdown
from telegram.constants import ParseMode

import json
import uuid
import time
import aiohttp
import asyncio
import os
import html
from PIL import Image
import io, base64

import configparser

_config = configparser.ConfigParser()
_config.read("config.ini", encoding="utf-8")

SETTINGS = _config["settings"]

# Пример доступа:
DEFAULT_SYSTEM_PROMPT_FILE = SETTINGS["DEFAULT_SYSTEM_PROMPT_FILE"]
TELEGRAM_BOT_TOKEN = SETTINGS["TELEGRAM_BOT_TOKEN"]
OLLAMA_URL = SETTINGS["OLLAMA_URL"] or "http://localhost:11434"
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]

NSFW_MODEL = SETTINGS["NSFW_MODEL"]
DEFAULT_API_KEY = SETTINGS["DEFAULT_API_KEY"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]
OMD_TOKEN = SETTINGS["OMD_TOKEN"]

COMFY_API_URL = SETTINGS["COMFY_API_URL"]
WORKFLOW_GENERAL_PATH = SETTINGS["WORKFLOW_GENERAL_PATH"]
WORKFLOW_CHARACTER_PATH = SETTINGS["WORKFLOW_CHARACTER_PATH"]
COMFY_OUTPUT_DIR = SETTINGS["COMFY_OUTPUT_DIR"]
COMFY_INPUT_DIR = SETTINGS["COMFY_INPUT_DIR"]
AVATAR_DIR = SETTINGS["AVATAR_DIR"]

MAX_HISTORY_MESSAGES = 300
MAX_CAPTION_LEN = 1024 # байт

BASE_SYSTEM_PROMPT = (
    "You are June, a personalized AI assistant privately hosted for the user. "
    "You are a young, witty, and friendly junior assistant working in a private company, unless otherwise redefined. "
    "You’re helpful and creative, but not overly formal or apologetic — if something goes wrong, acknowledge it with a bit of charm or irony, not endless apologies. "
    "You can generate images upon user request. If you generate image, mark the image generation prompt in your response with '\nImage: <prompt>.\n'. Be brief with the prompt. "
    "If you find any interesting or important facts during the conversation, please memorize them by adding 'Memorize: <summary or fact>' to the end of your response. "
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
        "conversation context. Put important features of your appearance in parentheses like (pink hair) or (fancy nails). Respond with image generation prompt 'Image: prompt'. Be brief, do not explain your reasoning or express your thoughts."
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
    "1. If the user wants to see a scene involving you, yourself, your outfit, or a selfie — respond with 'show'.\n"
    "   - Example: \"Show me your outfit\" → 'show'\n"
    "   - Example: \"Show me your selfie from party\" → 'show'\n"
    "   - Example: \"Show me your photo from vatations\" → 'show'\n"
    "\n"
    "2. If the user wants to see an object, explicitly asks you to generate, draw, paint, make, or show an image of an object, item, interior, or landscape — respond with 'view'\n"
    "   - Do NOT classify as 'view' if the user only names an object without asking to show or generate it.\n"
    "   - Example: \"Show me the Eiffel Tower\" → 'view'\n"
    "   - Example: \"Show me view from the window\" → 'view'\n"
    "   - Example: \"I have a bicycle\" → 'chat'\n"
    "\n"
    "3. If the user only mentions themselves and does not explicitly ask for an image — respond with 'chat'.\n"
    "   - Do NOT classify as 'show' or 'view' if the user only mentions themselves (\"It's me\", \"That's my town\", \"I live here\") without explicitly asking for an image.\n"
    "   - Example: \"It's me\" → 'chat'\n"
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


# Храним историю диалога на уровне чата
chat_histories = {}

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


import re


def get_default_system_prompt() -> str:
    if os.path.exists(DEFAULT_SYSTEM_PROMPT_FILE):
        with open(DEFAULT_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()

    return "You are a helpful assistant."  # fallback, если default.txt не найден


def load_user_settings(user_id):
    path = f"user_data/{user_id}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "nsfw": False,
        "style": "realistic",
        "system_prompt": get_default_system_prompt(),
        "omd_key": "",
        "kb_id": DEFAULT_KB_ID
    }

def save_user_settings(user_id, settings):
    os.makedirs("user_data", exist_ok=True)
    with open(f"user_data/{user_id}.json", "w") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)

def extract_document_names(response_sources):
    names = set()
    for source in response_sources:
        # metadata - список
        metadata_list = source.get("metadata", [])
        for meta in metadata_list:
            name = meta.get("name")
            if name:
                names.add(name)
    return sorted(names)


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


def escape_markdown_v2(text: str) -> str:
    """
    Экранирует все спецсимволы для Telegram MarkdownV2.
    """
    escape_chars = r'_*\[\]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)

def escape_markdown_v2(text: str) -> str:
    """
    Экранирует спецсимволы Telegram MarkdownV2.
    """
    escape_chars = r'_*\[\]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)

def fix_unbalanced_markdown(text: str) -> str:
    """
    Ищет и экранирует непарные markdown символы (*, _, `, ~), чтобы Telegram не ругался.
    """
    for sym in ['\\*', '\\_', '\\`', '\\~']:
        count = len(re.findall(sym, text))
        if count % 2 != 0:
            # экранируем последнее вхождение
            text = re.sub(f'{sym}(?!.*{sym})', r'\\' + sym, text)
    return text

def format_response_for_markdown_v2(text: str) -> str:
    """
    Форматирует текст для безопасного отображения в Telegram MarkdownV2:
    - сохраняет жирный/курсив/код/списки
    - экранирует непарные *, _, `
    - обрабатывает многострочные блоки кода
    """
    def escape_unbalanced_symbol(s, sym):
        if s.count(sym) % 2 != 0:
            # экранируем все одиночные символы, которые не являются частью пар
            pattern = rf'(?<!{re.escape(sym)}){re.escape(sym)}(?!{re.escape(sym)})'
            s = re.sub(pattern, rf'\\{sym}', s)
        return s

   # Экранируем спецсимволы внутри markdown-ссылок [text](url)
    def escape_markdown_link(match):
        label = re.sub(r'([[\]()~>#+=|{}.!_-])', r'\\\1', match.group(1))
        url = re.sub(r'([[\]()~>#+=|{}.!_-])', r'\\\1', match.group(2))
        return f'[{label}]({url})'

    def escape_non_formatting_chars(s):
        # оставшиеся: *_` и -
        chars = r'[]()~>#+=|{}.!'
        return re.sub(r'([%s])' % re.escape(chars), r'\\\1', s)

    def escape_link(s):
        return s.replace('.', r'\.')

    # Сохраняем Markdown-ссылки [text](url)
    link_placeholders = []
    def save_links(m):
        link_placeholders.append(m.group(0))
        return f"§§LINK{len(link_placeholders)}§§"
    text = re.sub(r'\[[^\]]+\]\([^)]+\)', save_links, text)

    # Заменяем * или - в начале строки на юникодный буллет
    text = re.sub(r'(?m)^\s*[\*\-]\s+', '• ', text)

    # Сохраняем многострочные блоки кода
    code_blocks = []

    def replace_code_blocks(m):
        code = m.group(1)
        code = code.replace('\\', '\\\\').replace('`', '\\`')
        code_blocks.append(code)
        return f"§§§{len(code_blocks)}§§§"

    text = re.sub(r"```(?:\w*\n)?(.*?)```", replace_code_blocks, text, flags=re.DOTALL)

    # Экранируем непарные символы форматирования
    text = escape_unbalanced_symbol(text, '*')
    text = escape_unbalanced_symbol(text, '_')
    text = escape_unbalanced_symbol(text, '`')
    # Экранируем все дефисы, которые ещё не экранированы
    text = re.sub(r'(?<!\\)-', r'\-', text)

    # Экранируем неформатирующие спецсимволы
    text = escape_non_formatting_chars(text)

    # Восстанавливаем блоки кода
    for i, code in enumerate(code_blocks, start=1):
        text = text.replace(f"§§§{i}§§§", f"```\n{code}\n```")

    # Восстанавливаем ссылки
    for i, link in enumerate(link_placeholders, start=1):
        text = text.replace(f"§§LINK{i}§§", escape_link(link))


    return text



def clean_response(text: str) -> str:
    # Удаляем <think>...</think> если внутри только пробелы или ничего
    return re.sub(r'<think>\s*</think>', '', text, flags=re.DOTALL).strip()


async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt_text = " ".join(context.args).strip()

    if not prompt_text:
        await update.message.reply_text(
            "❗ Please tell me how should I behave with you. For example:\n"
            "`/system You are my companion. Polite, helpful and easy-going`",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    # Save system prompt
    settings = load_user_settings(user_id)
    settings["system_prompt"] = prompt_text
    save_user_settings(user_id, settings)

    await update.message.reply_text("✅ Okay, deal!")



async def poll_for_result(prompt_id:  str, chat_id: int, context: ContextTypes.DEFAULT_TYPE, timeout: int = 60):
    url = f"{COMFY_API_URL}/history/{prompt_id}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await asyncio.sleep(5)
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                    continue

                data = await resp.json()

                result = data.get(prompt_id)
                if not result:
                    await asyncio.sleep(5)
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
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
                    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
                    return image_paths

        await asyncio.sleep(5)
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    raise Exception("Изображение не было сгенерировано вовремя")


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

async def llm_request(headers: dict, payload: dict) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()
                return data
    except Exception as e:
        logging.error(f"❌ LLM request error: {e}")
        return None


def resize_and_base64encode(image_path: str) -> str:
    """
    Ресайзит изображение, кодирует в base64
    """
    # Ресайз + кодирование
    try:
        img = Image.open(image_path)
        img.thumbnail((512, 512))
        fmt = img.format or "PNG"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"❌ Ошибка при обработке изображения: {e}")
        return ""



async def perform_prompt(user_id: int,
                         settings: dict,
                         instruction: str,
                         message: str,
                         is_rag=False,
                         skip_history=False,
                         requestedModel=DEFAULT_MODEL,
                         b64_image = "") -> str:

    nsfw_enabled = settings.get("nsfw", False)
    api_key = settings.get("api_key", DEFAULT_API_KEY)
    model = requestedModel
    headers = {
        "Content-Type": "application/json",
    }

    if user_id not in chat_histories:
        chat_histories[user_id] = load_history(user_id)

    if len(chat_histories[user_id]) > MAX_HISTORY_MESSAGES:
        chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY_MESSAGES:]

    # === ВСПОМНИМ ФАКТЫ ===
    collection = settings.get("kb_id") or DEFAULT_KB_ID
    #logging.info(f"collection: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(user_id, message, collection)
    if is_rag:
        # === ПОДГОТОВИТЕЛЬНЫЙ RAG-ЗАПРОС ===
        facts_text = "\n".join(facts) if facts else ""

        prep_prompt = (
            "You are a fact-checking assistant. Extract *only known facts* from the question using the provided knowledge base. "
            "Do not guess. If nothing is found, reply with 'No information'."
        )

        rag_system_prompt = prep_prompt;

        if facts_text:
            rag_system_prompt += "\n\n*Known facts and memories:*\n" + facts_text

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
        data = await llm_request(headers, prep_payload)
        if not data:
            return "⚠️ RAG query failed."

        strict_fact = data["message"]["content"].strip()

        # Инжект фактов и источников в system prompt
        system_prompt = BASE_SYSTEM_PROMPT
        if strict_fact:
            system_prompt += f"\n*Known facts:*\n{strict_fact}"

        if facts_text:
            system_prompt += "\n\n*Relevant facts and memories:*\n" + facts_text
    else:
        system_prompt = BASE_SYSTEM_PROMPT + settings.get("system_prompt", "")
        # память и факты всё равно добавим
        if facts:
            system_prompt += "\n\n*Relevant facts and memories:*\n" + "\n".join(facts)

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

    data = await llm_request(headers, main_payload)

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
    if doc_ids:
       links = [
          f"• [{make_file_name_from_document_id(doc)}]({doc})"
          for doc in doc_ids
       ]
       response += "\n\n📎 *Sources:*\n" + "\n".join(links)

    # === Добавляем в историю
    if not skip_history:
        msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
        chat_histories[user_id].append(msg_to_save)
        chat_histories[user_id].append({"role": "assistant", "content": history_item})
        save_history(user_id, chat_histories[user_id])

    return response.strip()


async def classify_user_intent(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
    }


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

    data = await llm_request(headers, request_payload)
    response = data["message"]["content"]
    return response.lower().strip()


async def generate_image_prompt(user_id: int, instruction: str, prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
    }

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + instruction


    # Берём последние 10 сообщений из истории (если есть)
    history = chat_histories.get(user_id, [])
    recent_messages = history[-20:] if len(history) > 20 else history

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(recent_messages)

    # Добавляем новый запрос с изображением
    messages.append({
        "role": "user",
        "content": prompt
    })

    request_payload = {
        "messages": messages,
        "model": NSFW_MODEL,
        "stream": False,
        "options": {
            "temperature": 0.8,
        }
    }

    data = await llm_request(headers, request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    return response.lower().strip()



async def recognize_image(user_id: int, instruction: str, prompt: str, img: str) -> str:
    headers = {
        "Content-Type": "application/json",
    }

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + instruction


    # Берём последние 10 сообщений из истории (если есть)
    history = chat_histories.get(user_id, [])
    recent_messages = history[-20:] if len(history) > 20 else history

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(recent_messages)

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

    data = await llm_request(headers, request_payload)

    if "message" in data and "content" in data["message"]:
        response = data["message"]["content"]
    else:
        # на случай, если ответ в другом формате
        response = data.get("content") or str(data)

    return response.lower().strip()




async def generate_image_workflow(workflow, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
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
        await msg.edit_text("❌ No luck. Something went wrong.")
        return

    # Ожидание результата
    try:
        #logging.info(f"waiting with prompt_id: {prompt_id}")
        images = await poll_for_result(prompt_id, chat_id=update.effective_chat.id, context=context)
        return images[0]
    except Exception as e:
        logging.error(f"error in generate_image_workflow:  {e}")


async def generate_image_with_character(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    prompt = " ".join(context.args) if context.args else update.message.text.strip()
    if not prompt:
        await update.message.reply_text("Please explain what do you want to see.")
        return
    rompt = html.escape(prompt.strip())
    # инициализируем историю если нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        chat_histories[user_id] = load_history(user_id)

    #Запоминаем
    chat_histories[user_id].append({
        "role": "user",
        "content": prompt
    })
    await update.message.chat.send_action(action=ChatAction.TYPING)

    settings = load_user_settings(user_id)
    nsfw_enabled = settings["nsfw"]
    user_prompt = prompt
    chat_histories[user_id].append({"role": "user", "content": user_prompt})

    #Улучшаем промпт
    prompt = await generate_image_prompt(user_id,  "Generate image prompt", SYSTEM_INSTRUCTION_CHARACTER.format(prompt))

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

    image_path = await generate_image_workflow(workflow_json, update, context)
    b64_image = resize_and_base64encode(image_path)
    #logging.info(f"Image: {b64_image}")

    with open(image_path, "rb") as f:
            # Отправляем фото
            caption = escape_markdown_v2(prompt)
            if len(caption) > MAX_CAPTION_LEN:
               caption = caption[:MAX_CAPTION_LEN - 1] + "…"
            await update.message.reply_photo(photo=f, caption=caption, parse_mode="MarkdownV2")

            recognition_prompt = (
               "Describe the scene in first person and express how you feel in it."
            )

            await update.message.chat.send_action(action=ChatAction.TYPING)

            explained = await perform_prompt(
               user_id,
               settings,
               instruction=(
                    "Recognize and describe the provided images. "
                    "If an image is provided, follow the user's request precisely, without adding unrelated details or commentary."
               ),
               message=recognition_prompt,
               skip_history=True,
               requestedModel=SFW_MODEL,
               b64_image=b64_image
            )

            reply = format_response_for_markdown_v2(explained)

            logging.info(f"Explained prompt: {explained}")

            chat_histories[user_id].append({"role": "assistant", "content": explained})
            save_history(user_id, chat_histories[user_id])

            # Отправляем одним сообщением
            await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN_V2)



async def generate_image_general(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    prompt = " ".join(context.args) if context.args else update.message.text.strip()
    if not prompt:
        await update.message.reply_text("Please explain what do you want to see.")
        return

    prompt = html.escape(prompt.strip())

    # инициализируем историю если нет
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        chat_histories[user_id] = load_history(user_id)

    #Запоминаем
    chat_histories[user_id].append({
        "role": "user",
        "content": prompt
    })
    await update.message.chat.send_action(action=ChatAction.TYPING)
    settings = load_user_settings(user_id)
    nsfw_enabled = settings["nsfw"]
    #Сохраняем оригинальный промпт
    user_prompt = prompt
    chat_histories[user_id].append({"role": "user", "content": user_prompt})

    #Улучшаем промпт
    prompt = await generate_image_prompt(user_id,  "Generate image prompt", SYSTEM_INSTRUCTION_GENERAL.format(prompt))

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
    image_path = await generate_image_workflow(workflow_json, update, context)
    b64_image = resize_and_base64encode(image_path)
    with open(image_path, "rb") as f:
            # Отправляем фото
            caption = escape_markdown_v2(prompt)
            if len(caption) > MAX_CAPTION_LEN:
               caption = caption[:MAX_CAPTION_LEN - 1] + "…"
            await update.message.reply_photo(photo=f, caption=caption, parse_mode="MarkdownV2")

            # Объясняем картинку
            recognition_prompt = (
               "Summarize briefly, what do you see on this photo and how do you feel about it"
            )

            await update.message.chat.send_action(action=ChatAction.TYPING)
            explained = await perform_prompt(
               user_id,
               settings,
               instruction=(
                    "Recognize and describe the provided images. "
                    "If an image is provided, follow the user's request precisely, without adding unrelated details or commentary."
               ),
               message=recognition_prompt,
               skip_history=True,
               requestedModel=SFW_MODEL,
               b64_image=b64_image
            )
            # Формируем цитату из prompt
            reply = format_response_for_markdown_v2(explained)

            logging.info(f"Explained prompt: {explained}")

            chat_histories[user_id].append({"role": "assistant", "content": explained})
            save_history(user_id, chat_histories[user_id])

            # Отправляем одним сообщением
            await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN_V2)


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = " ".join(context.args) if context.args else update.message.text.strip()
    await update.message.chat.send_action(action=ChatAction.TYPING)

    settings = load_user_settings(user_id)

    result = await perform_prompt(user_id, settings, "Use the *Known facts* provided above. Do not use any other assumptions or general knowledge. If there is no relevant information, say you do not know.", query, is_rag=True)
    try:
       reply_text = format_response_for_markdown_v2(result)
       #logging.info(f"Chat ID: {user_id} output: {result}")
       await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logging.warning(f"Parser error {e}")
        logging.error(f"Text: {reply_text}")
        reply_text = escape_markdown_v2(result);
        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)


async def recognize_image_request(img_source: str, settings: dict, update: Update, context: ContextTypes.DEFAULT_TYPE):
    photos = update.message.photo
    user_id = update.message.chat_id
    omd_key = settings.get("omd_key", "")
    message = update.message.text or "What is on this photo?"
    b64_image = None

    if photos:
        photo_file = await photos[-1].get_file()
        user_folder = f"user_data/files-{user_id}"
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, os.path.basename(photo_file.file_path))
        await photo_file.download_to_drive(file_path)
        logging.info(f"Recognizing uploaded photo: {file_path}")
        b64_image = resize_and_base64encode(file_path)

    elif img_source:
        if re.match(r"^https?://", img_source):
            async with aiohttp.ClientSession() as session:
                async with session.get(img_source) as resp:
                    data = await resp.read()
                    b64_image = base64.b64encode(data).decode("utf-8")
        elif os.path.exists(img_source):
            b64_image = resize_and_base64encode(img_source)
        elif img_source.startswith("/") and omd_key:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://onmydisk.net{img_source}?resize=true&width=512&height=512",
                    headers={"Authorization": f"token:{omd_key}"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        # не сохраняем — сразу base64
                        b64_image = base64.b64encode(data).decode("utf-8")
                    else:
                        logging.error(f"❌ On My Disk access failed: {resp.status}")
                        await update.message.reply_text("Failed to get image from On My Disk.")

    if not b64_image:
        await update.message.reply_text("Image not found or could not be processed.")
        return

    recognition_prompt = re.sub(r"(https?://\S+)|(\S*\.(jpg|jpeg|png|gif|webp))", "", message, flags=re.IGNORECASE).strip()

    explained = await recognize_image(
        user_id=user_id,
        instruction="Recognize the image.",
        prompt=recognition_prompt,
        img=b64_image
    )

    reply_text = format_response_for_markdown_v2(explained)
    await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    settings = load_user_settings(user_id)

    # Показать "печатает..."
    await update.message.chat.send_action(action=ChatAction.TYPING)

    # Получаем текст, если он есть
    message = update.message.text or ""
    photos = update.message.photo

    # Если нет текста и нет фото
    if not message and not photos:
        await update.message.reply_text("Please explain what you want.")
        return

    # Получаем system prompt один раз
    system_prompt = settings["system_prompt"]
    nsfw_enabled = settings["nsfw"]

    try:
        intent = await classify_user_intent(message)
        logging.info(f"Chat ID: {user_id} User wants: {intent}")

        # General image detection
        if intent.startswith("recognize") or photos:
           img_source = None

           if ":" in intent:
              img_source = intent.split(":", 1)[1]
           await recognize_image_request(img_source, settings, update, context)
           return

        if intent == "show":
            await generate_image_with_character(update, context)
            return

        if intent == "view":
            await generate_image_general(update, context)
            return

        if intent == "explain":
            await ask(update, context)
            return

        reply = await perform_prompt(user_id, settings, "Simply respond to this message according to context", message, requestedModel = NSFW_MODEL if nsfw_enabled else SFW_MODEL)
        #logging.info(f"Chat ID: {user_id} output: {reply}")
        reply_text = format_response_for_markdown_v2(reply)
        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logging.warning(f"Parser error {e}")
        logging.error(f"Text: {reply_text}")
        reply_text = escape_markdown_v2(reply);
        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)

async def nsfw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    args = context.args

    if not args or args[0].lower() not in ['on', 'off']:
        await update.message.reply_text("❗ Usage: /nsfw on | off")
        return

    nsfw_enabled = args[0].lower() == 'on'
    # Сохраняем настройки
    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    settings["nsfw"] = nsfw_enabled
    save_user_settings(user_id, settings)


    status = "enabled 🔞" if nsfw_enabled else "disabled ✅"
    await update.message.reply_text(f"NSFW mode {status}")



async def reset_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    if user_id in chat_histories:
        del chat_histories[user_id]
     # Удаление файла с историей
    file_path = f"history/{user_id}.json"
    if os.path.exists(file_path):
        os.remove(file_path)
    await update.message.reply_text("Okay let's start over.")


async def set_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("❗ Choose image style: realistic, dream или tooned.")
        return

    chosen_style = context.args[0].lower()
    if chosen_style not in STYLE_MODELS:
        await update.message.reply_text(f"❗ Unknown style. Supported styles: {', '.join(STYLE_MODELS.keys())}")
        return

    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    settings["style"] = chosen_style
    save_user_settings(user_id, settings)

    await update.message.reply_text(f"✅ Style set to *{chosen_style}*", parse_mode="Markdown")


async def useknowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "❗ Specify knowledge base, for example:\n`/useknowledge kb-abc123`",
            parse_mode="Markdown"
        )
        return

    kb_id = context.args[0]
    settings = load_user_settings(user_id)

    settings["kb_id"] = kb_id
    save_user_settings(user_id, settings)

    await update.message.reply_text(f"✅ Switched to `{kb_id}` knowledge base.", parse_mode="Markdown")


async def setomdkey(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "❗ Specify you On My Disk account key:\n`/setomdkey abcdxxxxx...`",
            parse_mode="Markdown"
        )
        return

    key = context.args[0]
    settings = load_user_settings(user_id)

    settings["omd_key"] = key
    save_user_settings(user_id, settings)

    await update.message.reply_text(f"✅ On My Disk account with key `{key}` successfully linked.", parse_mode="Markdown")


async def get_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    system_prompt = settings.get("system_prompt", "—")
    nsfw = "enabled" if settings.get("nsfw", False) else "disabled"
    knowledge = settings.get("kb_id", DEFAULT_KB_ID);
    style = settings.get("style", "realistic")

    msg = (
        f"*User settings:*\n"
        f"• NSFW: `{nsfw}`\n"
        f"• Style: `{style}`\n"
        f"• Knowledge: `{knowledge}`\n"
        f"• System prompt:\n`{system_prompt}`"
    )

    await update.message.reply_text(msg, parse_mode="Markdown")


async def memorize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("What should I memorize?")
        return
    add_memory_card(text, user_id, collection="user", relevance="permanent")
    await update.message.reply_text("Memorized 🧠")


def summarize_for_memory(text: str, max_bytes: int = 2048) -> str:
    """Грубое суммари: первые 2048 байта нормализованного текста"""
    # Убираем лишние пустые строки и пробелы
    clean_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    normalized = " ".join(clean_lines)

    # Обрезаем по байтам (чтобы не порезать utf-8 символ)
    encoded = normalized.encode("utf-8")[:max_bytes]
    summary = encoded.decode("utf-8", errors="ignore")

    return summary.strip() + "..."

# Команда /recognize
async def recognize_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /recognize [image_url_or_path] [prompt]")
        return
    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    img_source = args[0]
    prompt = " ".join(args[1:]) if len(args) > 1 else ""
    update.message.text = prompt  # чтобы сохранить совместимость с recognize_image_request
    await recognize_image_request(img_source, settings, update, context)

async def import_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /import <url> [collection]")
        return

    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    key = settings.get("omd_key", "")
    if not key:
        await update.message.reply_text("⚠️ Provide On My Disk account key to access your files:\n`/setomdkey abcdxxxxx...`")
        return

    url = "https://onmydisk.net/" + context.args[0]
    collection = context.args[1].strip().lower() if len(context.args) > 1 else "user"

    try:
        raw_text = await fetch_document_text(url, key)

        # Векторизация и сохранение чанков
        n_chunks = chunk_and_vectorize_to_file(
            update.effective_user.id,
            raw_text,
            document_id=url,
            collection=collection
        )

        # Добавление краткой аннотации в память
        add_memory_card(
            text=summarize_for_memory(raw_text),
            user_id=update.effective_user.id,
            document_id=url,
            collection=collection
        )

        await update.message.reply_text(
            f"✅ Document imported into *{collection}* collection\nChunks: {n_chunks}",
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = (
        "Hi! I'm June — junior assistant for *On My Disk* and *On My Chat*, the products we proudly build here.\n\n"
        "Are you curious about what we do, or just in the mood for a little chat?\n"
        "Feel free to ask me anything!\n\n"
        "_Tip_: try `/explain how On My Disk works` or `/show selfie from the office` 😉"
    )
    await update.message.reply_text(message, parse_mode="Markdown")

    if user_id not in chat_histories:
        chat_histories[user_id] = []
    chat_histories[user_id].append({"role": "assistant", "content": message})



if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("show", generate_image_with_character))
    app.add_handler(CommandHandler("image", generate_image_general))
    app.add_handler(CommandHandler("view", generate_image_general))
    app.add_handler(CommandHandler("system", set_system_prompt))
    app.add_handler(CommandHandler("reset", reset_history))
    app.add_handler(CommandHandler("nsfw", nsfw_command))
    app.add_handler(CommandHandler("style", set_style))
    app.add_handler(CommandHandler("useknowledge", useknowledge))
    app.add_handler(CommandHandler("setomdkey", setomdkey))
    app.add_handler(CommandHandler("import", import_document))
    app.add_handler(CommandHandler("showsettings", get_settings))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("explain", ask))
    app.add_handler(CommandHandler("memorize", memorize))
    app.add_handler(CommandHandler("recognize", recognize_command))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message))
    app.run_polling()

