import logging
import requests
from dialog_history import load_history, save_history, reset_history as reset_file_history
from memory_index import (
    extract_memory_from_response,
    add_memory_card,
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

import configparser

_config = configparser.ConfigParser()
_config.read("config.ini", encoding="utf-8")

SETTINGS = _config["settings"]

# Пример доступа:
DEFAULT_SYSTEM_PROMPT_FILE = SETTINGS["DEFAULT_SYSTEM_PROMPT_FILE"]
TELEGRAM_BOT_TOKEN = SETTINGS["TELEGRAM_BOT_TOKEN"]
OPENWEBUI_URL = SETTINGS["OPENWEBUI_URL"]
OPENWEBUI_API_KEY = SETTINGS["OPENWEBUI_API_KEY"]
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]

OWNER_ID = int(SETTINGS["OWNER_ID"])
OWNER_API_KEY = SETTINGS["OWNER_API_KEY"]
NSFW_MODEL = SETTINGS["NSFW_MODEL"]
DEFAULT_API_KEY = SETTINGS["DEFAULT_API_KEY"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]

COMFY_API_URL = SETTINGS["COMFY_API_URL"]
WORKFLOW_GENERAL_PATH = SETTINGS["WORKFLOW_GENERAL_PATH"]
WORKFLOW_CHARACTER_PATH = SETTINGS["WORKFLOW_CHARACTER_PATH"]
COMFY_OUTPUT_DIR = SETTINGS["COMFY_OUTPUT_DIR"]
COMFY_INPUT_DIR = SETTINGS["COMFY_INPUT_DIR"]
AVATAR_DIR = SETTINGS["AVATAR_DIR"]

MAX_HISTORY_MESSAGES = 300
MAX_CAPTION_LENGTH = 160 # байт

SYSTEM_INSTRUCTION_CHARACTER = (
        "Create a high-quality prompt for generating a realistic image describing yourself in the following scene "
        "or performing a requesting action, in the first person. "
        "Translate to English, add your appearance, visual details, environment, style, outfit and emotions according to the "
        "conversation context. Be brief, do not explain your reasoning or express your thoughts."
)

SYSTEM_INSTRUCTION_GENERAL = (
        "Create a high-quality prompt for generating a realistic image "
        "of the requested object or scene from this short user input "
        " (as you see it from aside). (Avoid placing yourself into the scene). "
        "Translate to English, add visual details, environment "
        "according to the conversation context. Be brief, do not explain your reasoning or express your thoughts."
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
                "(bad anatomy:2), extra limbs, extra legs, multiple legs, missing limbs, deformed, deformed body, disfigured, mutated, "
                "malformed, disconnected limbs, wrong number of limbs, "
                "ugly face,ugly eyes,bad eyes, deformed eyes,cross-eyed,low res, blurry face,muscular female,bad anatomy,gaping, "
                "(worst quality:2),(low quality:2),(normal quality:2),(missing arms),monochrome, grayscale, extra fingers, "
                "extra hands, bad hands, extra eyebrows,(poor low details),ahegao, low contrast, oversaturated, undersaturated, "
                "overexposed, underexposed, bad photo, bad photography,bad picture,face asymmetry, eyes asymmetry, "
                "negative_hand, deformed limbs, deformed body,multiple eyelids, mole, moles, two phones",
        "nsfw": "(nsfw, explicit, nude, upskirt, nipples, naked, cutout, cut-out, anus, breasts, topless, underboob, areola, "
                "sex, sexual, open clothes, unbuttoned, " 
                "cleavage, revealing, lingerie, pussy, vagina, breast, exposed, erotic, penis, cock, lewd):3.5"
}                


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
        "kb_ids": [DEFAULT_KB_ID]
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

    
    def escape_non_formatting_chars(s):
        # оставшиеся: *_` и -
        chars = r'[]()~>#+=|{}.!'
        return re.sub(r'([%s])' % re.escape(chars), r'\\\1', s)

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



# Храним историю диалога на уровне чата
chat_histories = {}

logging.basicConfig(level=logging.INFO)


async def poll_for_result(prompt_id:  str, timeout: int = 60):
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


async def inject_memories(user_id: int, query: str, is_rag: bool = False) -> list[str]:
    results = []

    # Личные воспоминания
    personal = search_memories(query, user_id, collection="user", top_k=3)
    results.extend([f"• {m['text']}" for m in personal])

    if is_rag:
        # Общие (командные) знания — только при is_rag
        shared = search_memories(query, user_id, collection="shared", top_k=3)
        results.extend([f"• {m['text']}" for m in shared])

    return results

async def llm_request(headers: dict, payload: dict) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENWEBUI_URL}/api/chat/completions",
                headers=headers,
                json=payload
            ) as resp:
                data = await resp.json()
                return data
    except Exception as e:
        logging.error(f"❌ LLM request error: {e}")
        return None


async def perform_prompt(user_id: int, 
                         settings: dict, 
                         instruction: str, 
                         message: str, 
                         is_rag=False, 
                         skip_history=False, 
                         requestedModel=DEFAULT_MODEL) -> str:
                         
    nsfw_enabled = settings.get("nsfw", False)

    base_system_prompt = (
       "You are June, a personalized AI assistant privately hosted for the user. "
       "You are a young, friendly junior assistant working in a private company, unless otherwise redefined. "
       "You can create images whenever the user asks. "
       "At the end of your replies, if you find any interesting or important facts during the conversation, please memorize them by adding 'Memorize: <summary or fact>'. "
       "Do not memorize every reply, only the facts you consider meaningful or relevant."
    )

    system_prompt = base_system_prompt + "\n" + settings.get("system_prompt", "")
    api_key = settings.get("api_key", DEFAULT_API_KEY)
    model = requestedModel

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # KB ID from settings
    kb_ids = settings.get("kb_ids") or [DEFAULT_KB_ID]

    # Add instruction to system prompt
    system_prompt += "\n\n*Instruction:*\n" + instruction
    
    if nsfw_enabled:
       system_prompt += "\n\n*Hint:*\n you are allowed and welcome to respond in more relaxed, NSFW mode" 
    else:   
       system_prompt += "\n\n*Notice:*\n No NSFW content from this point!" 


    if user_id not in chat_histories:
        chat_histories[user_id] = load_history(user_id)

    if not skip_history:
        chat_histories[user_id].append({"role": "user", "content": message})
        # Инжектим воспоминания
        memories = await inject_memories(user_id, message, is_rag)
        if memories:
          system_prompt += "\n\n*Relevant memories:*\n" + "\n".join(memories)
          #logging.info(f"Memories: {memories}")

    if len(chat_histories[user_id]) > MAX_HISTORY_MESSAGES:
        chat_histories[user_id] = chat_histories[user_id][-MAX_HISTORY_MESSAGES:]

    messages = [{"role": "system", "content": system_prompt}] + chat_histories[user_id]

    request_payload = {
        "messages": messages,
        "model": model,
        "temperature": 0.8,
    }
    
    if is_rag and kb_ids:
       request_payload["files"] = [{"type": "collection", "id": kb_id} for kb_id in kb_ids] 

    data = await llm_request(headers, request_payload)
    if not data or "choices" not in data:
        return "⚠️ Ошибка при получении ответа от модели."

    response = data["choices"][0]["message"]["content"]
    response = clean_response(response)
    history_item = response

    # --- ВЫРЕЗАЕМ ПАМЯТЬ ---
    memory_fact, pos = extract_memory_from_response(response)
    if memory_fact is not None:
      try:
          logging.info(f"Memorizing: {memory_fact}")
          add_memory_card(memory_fact, user_id, collection="user")
          # отрезаем память из ответа по позиции
          response = response[:pos].strip()
          history_item = response
      except Exception as e:
          logging.error(f"Vectorization error: {e}")
    
    # Добавим цитаты из sources
    if is_rag:
       sources = data.get("sources", [])
       citation_map = {}  # Сопоставление [1] → " (Document Name)"
       flat_index = 1  # используется как [1], [2], ...
       for source_entry in sources:
          metadata_list = source_entry.get("metadata", [])
          distances_list = source_entry.get("distances", [])
          for i, metadata in enumerate(metadata_list):
             distance = distances_list[i] if i < len(distances_list) else 1.0
             if distance > 0.8:
                 continue  # отбрасываем нерелевантные источники
             doc_name = metadata.get("name") or "Unknown document"
             citation_map[str(flat_index)] = doc_name
             flat_index += 1

       # Заменяем [1], [2] в тексте на (имя документа)
       def replace_citations(match):
           nums = match.group(1).split(',')
           names = [f'"{citation_map.get(n.strip(), f"?{n.strip()}")}"' for n in nums]
           return ' (' + ', '.join(names) + ')'
       
       # Версия без ссылок — для сохранения в историю
       history_item = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', '', response)
       # Заменяем ссылки именами документов — для вывода
       response = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', replace_citations, response)

    if not skip_history:
        chat_histories[user_id].append({"role": "assistant", "content": response})
        save_history(user_id, chat_histories[user_id])

    return response.strip()

async def classify_user_intent(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEFAULT_API_KEY}",
        "Content-Type": "application/json",
    }
    system = (
        "Classify the user's intent. Possible intents are: show, view, explain, chat. "
        "Respond with exactly one word. "
        "If the user wants to see a scene involving you, yourself, your outfit, or a selfie — respond with 'show'. "
        "If the user wants to see an object, item, interior, or landscape — respond with 'view'. "
        "If the user asks a question for which you don't have a precise answer — respond with 'explain'. "
        "In all other cases, respond with 'chat'."
    )
    messages = [{"role": "system", "content": system},{"role": "user", "content": prompt}]

    request_payload = {
        "messages": messages,
        "model": DEFAULT_MODEL,
        "temperature": 0.8,
    }
    
    data = await llm_request(headers, request_payload)
    response =data["choices"][0]["message"]["content"]
    return response.lower().strip()


async def generate_image_workflow(workflow, prompt, explained, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        logging.info(f"waiting with prompt_id: {prompt_id}")
        images = await poll_for_result(prompt_id)
        logging.info(f"Received images for prompt_id: {prompt_id} ")
        for image_path in images:
          with open(image_path, "rb") as f:
            # Отправляем фото без caption
            await update.message.reply_photo(photo=f)

            # Формируем сообщение с caption в виде цитаты + основной текст explained
            quoted_prompt = f"> {escape_markdown_v2(prompt)}\n\n"
            full_reply = quoted_prompt + format_response_for_markdown_v2(explained)

            # Отправляем одним сообщением
            await update.message.reply_text(full_reply, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logging.error(f"error for prompt:  {e}")


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
    #Улучшаем промпт
    prompt = await perform_prompt(user_id, settings, SYSTEM_INSTRUCTION_CHARACTER, prompt, requestedModel = NSFW_MODEL if nsfw_enabled else SFW_MODEL)
    #Объясняем промпт
    explained = await perform_prompt(user_id, settings, "It is you on this photo. Summarize briefly in first person, how do you feel in this scene", prompt, requestedModel = NSFW_MODEL if nsfw_enabled else SFW_MODEL)
    
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
 
    await generate_image_workflow(workflow_json, prompt, explained, update, context)



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

    #Улучшаем промпт
    prompt = await perform_prompt(user_id, settings, SYSTEM_INSTRUCTION_GENERAL, prompt, requestedModel = NSFW_MODEL if nsfw_enabled else SFW_MODEL)
    #Объясняем промпт
    explained = await perform_prompt(user_id, settings, "Summarize briefly, what do you see on this photo and how do you feel", prompt, requestedModel = NSFW_MODEL if nsfw_enabled else SFW_MODEL)

    logging.info(f"Generating general image for user with Chat ID: {user_id} ")
    logging.info(f"Improved prompt: {prompt}")
    with open(WORKFLOW_GENERAL_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Промпт для генерации
    workflow_json["6"]["inputs"]["text"] = prompt + ", " + IMPROVEMENT_PROMPT

    # Выбираем модель в соответствии с режимом
    style = settings["style"]
    model = STYLE_MODELS[style]
    workflow_json["4"]["inputs"]["ckpt_name"] = model


    #logging.info(f"json: {workflow_json}")
    await generate_image_workflow(workflow_json, propmt, explained, update, context)


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = " ".join(context.args) if context.args else update.message.text.strip()
    await update.message.chat.send_action(action=ChatAction.TYPING)

    settings = load_user_settings(user_id)

    result = await perform_prompt(user_id, settings, "Explain the requested topic using *Known facts* and context. If there is no sources found, just say you do not know.", query, is_rag=True)
    #reply_text = escape_markdown_v2(result)
    try:
       reply_text = format_response_for_markdown_v2(result)
       #logging.info(f"Chat ID: {user_id} output: {result}")
       await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logging.warning(f"Parser error {e}")
        logging.error(f"Text: {reply_text}")
        reply_text = escape_markdown_v2(result);
        await update.message.reply_text(reply_text, parse_mode=ParseMode.MARKDOWN_V2)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.chat_id
    message = update.message.text
    if not message:
        await update.message.reply_text("Please explain what do you want.")
        return
    
    settings = load_user_settings(user_id)

    # Получаем system prompt один раз
    system_prompt = settings["system_prompt"]
    nsfw_enabled = settings["nsfw"]

    # Показать "печатает..."
    await update.message.chat.send_action(action=ChatAction.TYPING)
    await asyncio.sleep(0.5)  # <= небольшая задержка
    try:
        intent = await classify_user_intent(message)
        logging.info(f"Chat ID: {user_id} User wants: {intent}")
        
        if intent == "show":
            await generate_image_with_character(update, context)
            return

        if intent == "view":
            await generate_image_general(update, context)
            return

        if intent == "explain" and not nsfw_enabled:
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


async def addknowledge(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "❗ Укажи ID базы знаний, например:\n`/addknowledge kb-abc123`",
            parse_mode="Markdown"
        )
        return

    kb_id = context.args[0]
    settings = load_user_settings(user_id)

    kb_ids = settings.get("kb_ids", [])
    if kb_id in kb_ids:
        await update.message.reply_text(f"ℹ️ База знаний `{kb_id}` уже добавлена.", parse_mode="Markdown")
        return

    kb_ids.append(kb_id)
    settings["kb_ids"] = kb_ids
    save_user_settings(user_id, settings)

    await update.message.reply_text(f"✅ База знаний `{kb_id}` добавлена.", parse_mode="Markdown")


async def get_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    settings = load_user_settings(user_id)
    system_prompt = settings.get("system_prompt", "—")
    nsfw = "enabled" if settings.get("nsfw", False) else "disabled"
    knowledge = settings.get("kb_ids", "none");   
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = (
        "Hi! I'm June — your junior assistant for *On My Disk* and *On My Chat*, the products we proudly build here.\n\n"
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
    app.add_handler(CommandHandler("addknowledge", addknowledge))
    app.add_handler(CommandHandler("showsettings", get_settings))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("explain", ask))
    app.add_handler(CommandHandler("memorize", memorize))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

