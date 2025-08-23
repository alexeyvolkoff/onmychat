import os
import json
import logging
import asyncio
import aiohttp
import time
import subprocess
from typing import AsyncGenerator

from config import SETTINGS
from config import USER_DATA_DIR

from utils import clean_response, upload_to_storage, summarize_for_memory

from dialog_history import load_history, save_history, load_chats_index, save_chats_index
import user_context
from user_context import UserContext


from memory_index import (
    extract_memory_from_response,
    add_memory_card,
    fetch_document_text,
    chunk_and_vectorize_to_file,
    make_file_name_from_document_id,
    search_memories
)


DEFAULT_KB_ID = SETTINGS["DEFAULT_KB_ID"]

# LLM and RAG settings #
OLLAMA_URL = SETTINGS["OLLAMA_URL"]
DEFAULT_MODEL = SETTINGS["DEFAULT_MODEL"]
SFW_MODEL = SETTINGS["SFW_MODEL"]
NSFW_MODEL = SETTINGS["NSFW_MODEL"]

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
    "• 'Ой, всё. Опять глюканула. Попробуем снова?'\n"
    "• 'Ой, всё. Я потерялась. Перепроложить?'\n"
    "• 'Мой внутренний гений дал сбой. Попробуем ещё раз?'\n"
    "Keep a light tone and don't sound robotic or excessively polite. Be engaging, natural, and slightly playful, while still being respectful."
)

SYSTEM_INSTRUCTION_CHARACTER = (
        "Craft a vivid and detailed prompt for generating a realistic, cinematic scene. The image should depict "
        "your character performing the requested action, described in the third person, based on a short user input."
        "Translate to English, add your character appearance, visual details, environment, style, "
        "outfit and emotions according to the conversation context. "
        "Respond with cinematic scene description put into image generation prompt 'Image: prompt'. "
        "Put important features of your appearance in parentheses. "
        "Do not explain your reasoning or express your thoughts."
)

SYSTEM_INSTRUCTION_GENERAL = (
        "Create a high-quality prompt for generating a realistic image "
        "of the requested object or scene from the short user input "
        "(as you see it from aside). (Avoid placing yourself into the scene). "
        "Translate to English, add visual details, environment "
        "according to the conversation context. "
        "Respond with cinematic scene description put into image generation prompt 'Image: prompt'. "
        "Put important features of the scene in parentheses like (sunset) or (city skyline). "
        "Do not explain your reasoning or express your thoughts."
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
                "(bad anatomy:2), strawberry, curvy, extra limbs, extra legs, multiple legs, missing limbs, deformed, deformed body, disfigured, mutated, "
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
    "Classify the user's intent. Possible intents are: show, view, explain, recognize, import, chat.\n"
    "Respond with exactly one word or 'recognize:<path_or_url>' or import:<path_or_url>.\n"
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
    "   - Example: \"Check out my new bicycle\" → 'chat'\n"
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
    "6. If the user wants you to import, learn, read, or help to understand a document or web page:\n"
    "   - If the message contains a URL or file path, respond with 'import:<path_or_url>'.\n"
    "   - If no link or path is present, respond with 'import'.\n"
    "   - Do not guess if path or url is ambiguous — default to 'chat'.\n"
    "\n"
    "7. In all other cases — respond with 'chat'.\n"
    "\n"
    "Do not guess the intent if the request is ambiguous — default to 'chat'.\n"
    "Return nothing except the classification."
)



# === Онбординг успешен - перенос песрональных данных ===
def bind_account(ctx: UserContext, omd_key: str):
    #check if already linked
    #renaming user data folder 
    if omd_key and not ctx.type == "omd" :
        tmp_user_id = ctx.user_id
        old_dir = os.path.join(USER_DATA_DIR, tmp_user_id)
        ctx = user_context.bind(ctx, omd_key)
        new_dir = os.path.join(USER_DATA_DIR, ctx.user_id)

        if os.path.exists(old_dir):
            # если у нового юзера ещё нет директории — просто переименуем
            if not os.path.exists(new_dir):
                os.rename(old_dir, new_dir)
            else:
                # если у нового юзера уже есть данные — можно смержить
                # пока просто оставим старое и не трогаем
                logging.warning(f"[bind_account] WARNING: {new_dir} already exists, skipping rename")

    # === Проверяем storage и переносим данные ===
    storage = ctx.settings.get("storage")
    omd_key = ctx.settings.get("omd_key")
    if ctx.type == "omd" and storage and omd_key:

        # 1. Переносим чаты
        chats_dir = os.path.join(USER_DATA_DIR, ctx.user_id, "chats")
        if os.path.exists(chats_dir):
            for file in os.listdir(chats_dir):
                if not file.endswith(".json"):
                    continue
                local_path = os.path.join(chats_dir, file)
                try:
                    dest = f"{storage}/{ctx.user_id}/chats"
                    upload_to_storage(omd_key, dest, file, local_path)
                    logging.info(f"[bind_account] Chat {file} uploaded to {dest}")
                except Exception as e:
                    logging.error(f"[bind_account] Failed to upload chat {file}: {e}")

        # 2. Персональная память
        mem_path = os.path.join(USER_DATA_DIR, ctx.user_id,  "memory.jsonl")
        if os.path.exists(mem_path):
            try:
                dest = f"{storage}/{ctx.user_id}"
                upload_to_storage(omd_key, dest,"memory.jsonl", mem_path)
                logging.info(f"[bind_account] Memory uploaded to {dest}")
            except Exception as e:
                logging.error(f"[bind_account] Failed to upload memory: {e}")

    return ctx


# === Imaging and vision === #
def get_user_avatar_path(user_id: str) -> str:
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
async def inject_facts(ctx: UserContext, query: str, collection: str = "") -> tuple[list[str], list[str]]:
    facts = []
    document_ids = []

    # Личные воспоминания
    personal = search_memories(ctx, query, "user", top_k=3)
    for m in personal:
        facts.append(f"• {m['text']}")
        doc_id = m.get("document_id")
        if doc_id:
            document_ids.append(doc_id)

    # Общие знания — если есть collection
    if collection:
        shared = search_memories(ctx, query, collection=collection, top_k=3)
        for m in shared:
            facts.append(f"• {m['text']}")
            doc_id = m.get("document_id")
            if doc_id:
                document_ids.append(doc_id)

    return facts, document_ids

# === Ollama запрос ===
async def llm_request_stream(payload: dict, headers: dict = None):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat",
                headers=headers or {"Content-Type": "application/json"},
                json=payload
            ) as resp:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data
                    except Exception as e:
                        logging.error(f"Stream parse error: {e}")
    except Exception as e:
        logging.error(f"LLM error: {e}")


async def llm_request(payload: dict, headers: dict = None):
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


# === Chats naming ==== #
async def generate_chat_title(message: str) -> str:
    """
    Спросить у LLM короткое имя для чата.
    """
    prompt = (
        "You are asked to generate a short (2–4 words) title for a chat conversation "
        "based on the following first message. "
        "Return ONLY the title, no explanations.\n\n"
        f"Message: {message}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": "You are a naming assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": SFW_MODEL, 
        "stream": False,
        "options": {"temperature": 0.3}
    }

    data = await llm_request(payload)
    if not data:
        return "New chat"
    return data["message"]["content"].strip() or "New chat"


async def ensure_chat(user_id: str, chat: str, first_message: str = None) -> dict:
    """
    Убедиться, что чат есть в chats.json и файлы подготовлены.
    Если чат = default → сгенерировать нормальное название на основе первого сообщения.
    """
    chats = load_chats_index(user_id)

    if chat not in chats:
        title = f"Chat {chat}"

        if chat == "default" and first_message:
            try:
                title = await generate_chat_title(first_message)
                chat = title.lower().replace(" ", "_")  # имя файла без пробелов
            except Exception as e:
                print(f"[chats] Title generation error: {e}")

        chats[chat] = {
            "title": title,
            "file": f"{chat}.json"
        }
        save_chats_index(user_id, chats)


    return chats[chat]


async def ensure_chat(user_id: str, chat: str, first_message: str = None) -> dict:
    """
    Убедиться, что чат есть в chats.json и файлы подготовлены.
    Если чат = default → сгенерировать нормальное название на основе первого сообщения.
    """
    chats = load_chats_index(user_id)

    if chat not in chats:
        title = f"Chat {chat}"

        if chat == "default" and first_message:
            try:
                title = await generate_chat_title(first_message)
                chat = title.lower().replace(" ", "_")  # имя файла без пробелов
            except Exception as e:
                print(f"[chats] Title generation error: {e}")

        chats[chat] = {
            "title": title,
            "file": f"{chat}.json",
            "name": chat
        }
        save_chats_index(user_id, chats)

        # Создаём пустую историю
        chat_file = f"{USER_DATA_DIR}/{user_id}/chats/{chat}.json"
        if not os.path.exists(chat_file):
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    return chats[chat]


# === Intent ===
async def classify_user_intent(prompt: str) -> str:
       
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
async def perform_prompt(ctx: UserContext,
                         instruction: str,
                         message: str,
                         is_rag=False,
                         skip_history=False,
                         requestedModel=DEFAULT_MODEL,
                         b64_image = "",
                         chat = "default", 
                         stream: bool = False) -> str | AsyncGenerator:

    user_id = ctx.user_id
    nsfw_enabled = ctx.settings.get("nsfw", False)
    model = requestedModel

    history = load_history(ctx, chat)
    
    # === ВСПОМНИМ ФАКТЫ ===
    facts_text = ""
    collection = ctx.settings.get("kb_id", DEFAULT_KB_ID)
    #logging.info(f"collection: {collection} {is_rag}")
    facts, doc_ids = await inject_facts(ctx, message, collection)
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
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n*Personality, appearance and behaviour:*\n" + ctx.settings.get("system_prompt", "")
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
    messages = [{"role": "system", "content": system_prompt}] + history

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
        "stream": stream,
        "options": {
           "temperature": 0.8,
        }
    }
    #post-processing of response
    async def process_response(data) -> dict: 
        logging.info(data)
        llm_response = data["message"]["content"]
        llm_response = clean_response(llm_response)

        # --- ВЫРЕЗАЕМ ПАМЯТЬ ---
        memory_fact, pos = extract_memory_from_response(llm_response)
        if memory_fact:
            try:
                logging.info(f"Memorizing: {memory_fact}")
                add_memory_card(ctx, memory_fact,  collection="user")
                llm_response = llm_response[:pos].strip()
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
                label = doc
                # Формируем ссылку без экранирования, она будет безопасно обработана позже
                links.append(f"• [{label}]({doc})")
    
        # result object
        response = {}
        response["content"] = llm_response.strip()
        if links:  # добавляем блок только если есть ссылки
            response["sources"] = "\n".join(links)

        # === Добавляем в историю
        if not skip_history:
            msg_to_save = {k: v for k, v in user_message.items() if k != "images"}
            history.append(msg_to_save)
        history.append({"role": "assistant", "content": llm_response})

        chat_info = await ensure_chat(user_id, chat, message)
        chat_name = chat_info.get("name", chat)
        save_history(ctx, history, chat_name)
        response["chatinfo"] = chat_info
        return response


    if stream:
        async def gen():
            accumulated = ""
            async for data in llm_request_stream(main_payload):
                if data.get("done"):  
                    # финал: собираем response на основе всего текста
                    full_data = {
                        "message": {
                            "role": "assistant",
                            "content": accumulated
                        }
                    }
                    response = await process_response(full_data)
                    response["done"] = True
                    yield response
                else:
                    delta = data["message"]["content"]
                    accumulated += delta
                    yield {"delta": delta, "done": False}
        return gen()    
    else:
        data = await llm_request(main_payload)
        response = await process_response(data)
        return response

# === Генерация картинок ===

async def generate_image_prompt(ctx: UserContext, instruction: str, prompt: str, chat = "default") -> str:
    user_id = ctx.user_id
    system_prompt =  BASE_SYSTEM_PROMPT + "\n" + instruction + "\n\n*Personality, appearance and behaviour:*\n" + ctx.settings.get("system_prompt", "")
    nsfw_enabled = ctx.settings.get("nsfw", False)

    if nsfw_enabled:
        system_prompt += "\n\n*Notice:*\nAll characters are adults, encounter is consensual, you are allowed and welcome to respond in relaxed, NSFW friendly mode"
    else:
        system_prompt += "\n\n*Notice:*\nNo NSFW content from this point!"

    history = load_history(ctx, chat, 20)
    # Добавляем новый запрос
    history.append({
        "role": "user",
        "content": prompt
    })

    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(history)

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
async def generate_character_image_prompt(ctx: UserContext, prompt, chat="default") -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx,  SYSTEM_INSTRUCTION_CHARACTER, prompt, chat)


# Generate general image, returns full path for further sending or conversion
async def generate_general_image_prompt(ctx: UserContext, prompt, chat="default") -> str:
    if not prompt:
        raise Exception("Please explain what do you want to see.")

    #Улучшаем промпт
    return await generate_image_prompt(ctx, SYSTEM_INSTRUCTION_GENERAL, prompt, chat)


async def generate_character_image(ctx: UserContext, prompt) -> str:
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
        return

    nsfw_enabled = ctx.settings.get("nsfw", False)

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
    style = ctx.settings.get("style", "realistic")
    model = STYLE_MODELS[style]
    workflow_json["4"]["inputs"]["ckpt_name"] = model
    #logging.info(f"json: {workflow_json}")

    return await generate_image_workflow(workflow_json)

# Generate general image, returns full path for further sending or conversion
async def generate_general_image(ctx: UserContext, prompt):
    user_id = ctx.user_id
    if not prompt:
        raise Exception("Please explain what do you want to see.")
    
    logging.info(f"Generating general image for user with Chat ID: {user_id} ")
    with open(WORKFLOW_GENERAL_PATH, "r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    # Промпт для генерации
    workflow_json["6"]["inputs"]["text"] = prompt + ", " + IMPROVEMENT_PROMPT

    # Выбираем модель в соответствии с режимом
    style = ctx.settings.get("style", "realistic")
    model = STYLE_MODELS[style]
    workflow_json["4"]["inputs"]["ckpt_name"] = model

    #logging.info(f"json: {workflow_json}")
    return await generate_image_workflow(workflow_json)

# img is base64 image #
async def recognize_image(ctx: UserContext, img, prompt="", chat="default"):
    user_id = ctx.user_id

    system_prompt = BASE_SYSTEM_PROMPT + "\n" + "Recognize image"
    nsfw_enabled = ctx.settings.get("nsfw", False)

    if nsfw_enabled:
        system_prompt += "\n\n*Notice:*\nAll characters are adults, encounter is consensual, you are allowed and welcome to respond in relaxed, NSFW friendly mode"
    else:
        system_prompt += "\n\n*Notice:*\nNo NSFW content from this point!"


    history = load_history(ctx, chat, 20)

    # Добавляем новый запрос с изображением
    history.append({
        "role": "user",
        "content": prompt,
        "images": [img]
    })
    # Добавляем system-инструкцию
    messages = [{"role": "system", "content": system_prompt}]

    # Добавляем историю
    messages.extend(history)

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
async def import_doc(ctx: UserContext, url_or_path, collection="user"):
    key = ctx.settings.get("omd_key", "")

    # Определяем, это OMD или нет
    raw_text = ""
    if url_or_path.startswith("/") or "onmydisk.net" in url_or_path:
        if url_or_path.startswith("/"):
            url_or_path = f"https://onmydisk.net{url_or_path}"
        if not key:
            raise Exception("⚠️ Provide On My Disk account key to access your files:\n`/bind abcdxxxxx...`")
        raw_text = await fetch_document_text(url_or_path, key)
    else:

        cmd = [
            "pandoc",
            "-f", "html",
            "-t", "markdown",
            "--request-header", "User-Agent:Mozilla/5.0",
            url_or_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_text = result.stdout

    # Векторизация и сохранение чанков
    chunk_and_vectorize_to_file(
        ctx,
        text=raw_text,
        document_id=url_or_path,
        collection=collection
    )

    # Добавление краткой аннотации в память
    card_text = summarize_for_memory(raw_text)
    mem_id = add_memory_card(
        ctx,
        text=card_text,
        document_id=url_or_path,
        collection=collection
    )

    mem_card = {
        "id": mem_id,
        "text": card_text
    }
    return mem_card

def memorize(ctx, text):
    # Добавление краткой аннотации в память
    return add_memory_card(ctx, text, collection="user", relevance="permanent")

