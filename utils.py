import base64
import os
from PIL import Image
import re
import io
import logging

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
