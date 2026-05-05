import base64
from PIL import Image
import re
import io
import os
import logging
import requests
import mimetypes
import json
import aiohttp

from config import SETTINGS
GATEWAY_URL = SETTINGS["GATEWAY_URL"]

def escape_markdown_v2(text: str) -> str:
    """
    Экранирует все спецсимволы для Telegram MarkdownV2.
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
    def escape_markdown_link_str(link: str) -> str:
        # Парсим [label](url)
        m = re.match(r'\[([^\]]+)\]\(([^)]+)\)', link)
        if m:
            # Экранируем спецсимволы только в label
            label = re.sub(r'([_\*\[\]\(\)~`>#+\-=|{}.!])', r'\\\1', m.group(1))
            url = re.sub(r'([_\*\[\]\(\)~`>#+\-=|{}.!])', r'\\\1', m.group(2))
            return f'[{label}]({url})'
        else:
            # Если парсинг не удался — экранируем всю строку
            return re.sub(r'([_\*\[\]\(\)~`>#+\-=|{}.!])', r'\\\1', link)
    
    def escape_non_formatting_chars(s):
        chars = r'[]()~>#+=|{}.!'
        return re.sub(r'([%s])' % re.escape(chars), r'\\\1', s)

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

    # Экранируем одинокие обратные слэши
    text = text.replace('\\', '\\\\')
    # Экранируем непарные символы форматирования
    text = escape_unbalanced_symbol(text, '*')
    text = escape_unbalanced_symbol(text, '_')
    text = escape_unbalanced_symbol(text, '`')

    # Экранируем неформатирующие спецсимволы
    text = escape_non_formatting_chars(text)
    # Восстанавливаем блоки кода
    for i, code in enumerate(code_blocks, start=1):
        text = text.replace(f"§§§{i}§§§", f"```\n{code}\n```")

    # Экранируем все дефисы
    text = re.sub(r"-", r"\-", text)

    # Восстанавливаем ссылки
    for i, link in enumerate(link_placeholders, start=1):
        text = text.replace(f"§§LINK{i}§§", escape_markdown_link_str(link))

    return text

def clean_response(text: str) -> str:
    # Удаляем <think>...</think> если внутри только пробелы или ничего
    return re.sub(r'<think>\s*</think>', '', text, flags=re.DOTALL).strip()

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
        logging.error(f"❌ Ошибка при обработке изображения: {e} {image_path}")
        return ""


async def get_image_from_source(ctx, img_source: str):
    b64_image = None
    omd_key = ctx.omd_key
    if img_source:
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
                    f"{GATEWAY_URL}{img_source}?resize=true&width=512&height=512",
                    headers={"Authorization": f"token:{omd_key}"}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        b64_image = base64.b64encode(data).decode("utf-8")
                    else:
                        logging.error(f"OMD access failed: {resp.status}")
                        return None
    return b64_image                


### upload to OMD ###

def upload_to_storage(omd_key: str, dest: str, filename: str, local_path: str):
    """
    Залить файл в OMD storage
    dest — полный путь (например "storage/user123/history/chat1.json")
    """
    headers = {
        "authorization": f"token:{omd_key}",
        "Response": "json",
        "Dest": requests.utils.quote(dest, safe="")  # полный путь, включая имя файла
    }

    mime, _ = mimetypes.guess_type(local_path)
    headers["Content-Type"] = mime or "application/octet-stream"

    clean_dest = dest.lstrip("/")
    url = f"{GATEWAY_URL}/{clean_dest}/{filename}?jsonResponse=true"  

    with open(local_path, "rb") as f:
        resp = requests.put(url, headers=headers, data=f)
        resp.raise_for_status()
    return resp.json()


def upload_data_to_storage(omd_key: str, dest: str, filename: str, data, mime: str=""):
    """
    Залить файл в OMD storage
    dest — полный путь (например "storage/user123/history/chat1.json")
    """
    if mime == "application/json" and not isinstance(data, (bytes, str)):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    elif isinstance(data, str):
        body = data.encode("utf-8")
    else:
        body = data

    headers = {
        "authorization": f"token:{omd_key}",
        "Response": "json",
        "Dest": requests.utils.quote(dest, safe=""),  # полный путь, включая имя файла
        "Content-Type": mime
    }

    headers["Content-Type"] = mime or "application/octet-stream"

    clean_dest = dest.lstrip("/")
    url = f"{GATEWAY_URL}/{clean_dest}/{filename}?jsonResponse=true"  

    resp = requests.put(url, headers=headers, data=body)
    try:
        resp.raise_for_status()
    except Exception as e:
        logging.error(f"Upload failed: {e} | Response: {resp.text}")
        raise e
    return resp.json()


def fetch_json_from_storage(omd_key: str, dest: str, filename: str):
    """
    Скачать JSON файл из OMD storage
    """
    headers = {
        "authorization": f"token:{omd_key}",
    }
    clean_dest = dest.lstrip("/")
    url = f"{GATEWAY_URL}/{clean_dest}/{filename}"
    try:
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise e
    except Exception as e:
        logging.warning(f"Failed to fetch {filename} from storage: {e}")
        raise e
    return None


def upload_vec_to_storage(omd_key: str, dest: str, filename: str, data: list[dict], mime: str=""):
    """
    Залить файл в OMD storage
    dest — полный путь (например "storage/user123/history/chat1.json")
    """
    body = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in data).encode("utf-8")


    headers = {
        "authorization": f"token:{omd_key}",
        "Response": "json",
        "Dest": requests.utils.quote(dest, safe=""),  # полный путь, включая имя файла
        "Content-Type": mime
    }

    headers["Content-Type"] = mime or "application/octet-stream"

    clean_dest = dest.lstrip("/")
    url = f"{GATEWAY_URL}/{clean_dest}/{filename}?jsonResponse=true"  

    resp = requests.put(url, headers=headers, data=body)
    return resp.json()





def strip_html(text: str) -> str:
    """Удаляет все HTML-теги и атрибуты, оставляет только текст."""
    # Убираем <script> и <style> с содержимым
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Убираем все остальные теги <...>
    text = re.sub(r"<[^>]+>", "", text)
    return text


def clean_text(text: str) -> str:
    """
    Убирает HTML, Markdown-разметку, картинки, ссылки и лишний мусор только с помощью regex.
    Сохраняет основной текст.
    """
    # Шаг 1: Удаляем HTML-теги (простой regex, не вложенные идеально, но для базовых случаев ок)
    text = re.sub(r'<[^>]+>', '', text)
    # Шаг 2: Обрабатываем Markdown

    # Удаляем Markdown-ссылки [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Удаляем изображения ![alt](url) → ''
    text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)

    # Удаляем горизонтальные линии (---, ***, ___)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Удаляем символы списков (-, *, +, 1.) в начале строк
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Удаляем блок-цитаты (> )
    text = re.sub(r'^\s*>\s?', '', text, flags=re.MULTILINE)

    # Удаляем форматирование таблиц (| и ---): заменяем | на пробелы, удаляем строки с ---
    text = re.sub(r'^\s*[-:| ]+\s*$', '', text, flags=re.MULTILINE)  # Удаляем строки вроде |---|
    text = re.sub(r'\s*\|\s*', ' ', text)  # Заменяем | на пробелы

    # Удаляем чистые URL
    text = re.sub(r'https?://\S+', '', text)

    # Удаляем оставшиеся символы форматирования (#, *, _, `, >, ~, -, =, |)
    text = re.sub(r'[#*_`>~=\-|]+', ' ', text)

    # Схлопываем множественные пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text).strip()


    return text


def report_usage(omd_key: str, action: str, amount: float):
    """
    Dummy/logging implementation of usage reporting.
    """
    logging.info(f"Report usage: {action} {amount}")






