import base64
from PIL import Image
import re
import io
import logging
import requests
import mimetypes
import json

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
        logging.error(f"❌ Ошибка при обработке изображения: {e}")
        return ""


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

    url = f"https://onmydisk.net/{dest}/{filename}?jsonResponse=true"  

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

    url = f"https://onmydisk.net/{dest}/{filename}?jsonResponse=true"  

    resp = requests.put(url, headers=headers, data=body)
    return resp.json()


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

    url = f"https://onmydisk.net/{dest}/{filename}?jsonResponse=true"  

    resp = requests.put(url, headers=headers, data=body)
    return resp.json()



def generate_table_of_contents(text: str) -> str:
    """
    Generates a table of contents from the given text.
    Includes both chapter headings and Markdown headings.
    """
    chapter_regex = re.compile(r"^\d+(\.\d+)+.*")  # Regex for chapter headings
    markdown_regex = re.compile(r"^#+\s.*")        # Regex for Markdown headings

    toc_entries = []

    for line in text.splitlines():
        if chapter_regex.match(line) or markdown_regex.match(line):
            toc_entries.append(line.strip())

    return "\n".join(toc_entries)


def summarize_for_memory(
    text: str,
    max_bytes: int = 2048,
    max_leading_bytes: int = 512
) -> str:
    """
    Summarizes text: returns full text if it's small,
    otherwise returns a summary consisting of:
      - Leading context (lines before the first heading, up to max_leading_bytes)
      - Table of contents (headings up to 3 levels deep)
    """
    text_bytes = len(text.encode("utf-8"))

    if text_bytes <= max_bytes:
        return text  # return as-is if small enough

    # --- extract leading context before the first heading ---
    lines = text.splitlines()
    chapter_regex = re.compile(r"^\d+(\.\d+)+.*")
    markdown_regex = re.compile(r"^#+\s.*")

    leading_lines = []
    for line in lines:
        if chapter_regex.match(line) or markdown_regex.match(line):
            break
        if line.strip():
            leading_lines.append(line.strip())

    # truncate leading context if too large
    leading_text = "\n".join(leading_lines)
    if len(leading_text.encode("utf-8")) > max_leading_bytes:
        leading_text = leading_text.encode("utf-8")[:max_leading_bytes].decode("utf-8", errors="ignore") + "..."

    # --- build TOC ---
    toc = generate_table_of_contents(text)

    # --- final summary ---
    if leading_text:
        return f"{leading_text}\n\nTable of Contents:\n{toc}"
    else:
        return toc



