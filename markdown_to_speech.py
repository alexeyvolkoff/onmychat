"""
Markdown-to-Speech Normalizer for Conversational TTS.
Converts structured markdown (headers, lists, tables, code blocks, math, links)
into natural, fluent spoken English suitable for Kokoro-82M.
"""

import re
import html
from typing import List, Optional


class MarkdownToSpeechNormalizer:
    def __init__(self):
        # Common English spoken abbreviation expansions
        self.abbreviations = [
            (r'\be\.g\.,?\s*', 'for example, '),
            (r'\bi\.e\.,?\s*', 'that is, '),
            (r'\betc\.\b', 'etcetera'),
            (r'\bvs\.\b', 'versus'),
            (r'\bw/\b', 'with'),
            (r'\bw/o\b', 'without'),
            (r'\bapprox\.\b', 'approximately'),
            (r'\bdept\.\b', 'department'),
            (r'\bmin\.\b', 'minutes'),
            (r'\bsec\.\b', 'seconds'),
            (r'\bhrs?\.\b', 'hours'),
            (r'\bconfig\.ini\b', 'config dot ini'),
            (r'\b\.py\b', ' dot py'),
            (r'\b\.json\b', ' dot json'),
            (r'\b\.js\b', ' dot js'),
            (r'\b\.md\b', ' dot m d'),
        ]

        try:
            from config import SETTINGS
            val_ec = SETTINGS.get("TTS_EXTENDED_COMMAS", "false")
            self.extended_comma_pauses = val_ec.lower() in ("true", "1", "yes") if isinstance(val_ec, str) else bool(val_ec)
            val_ms = SETTINGS.get("TTS_MICRO_SAMPLES", "true")
            self.use_micro_samples = val_ms.lower() in ("true", "1", "yes") if isinstance(val_ms, str) else bool(val_ms)
        except Exception:
            self.extended_comma_pauses = False
            self.use_micro_samples = True

    def normalize(self, text: str) -> str:
        """Main entry point: transforms markdown text into natural spoken speech."""
        if not text or not text.strip():
            return ""

        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 1. Strip think/reasoning blocks (<think>...</think>)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # 2. Handle GitHub-style callouts/alerts (> [!NOTE], etc.)
        def replace_alert(m):
            alert_type = m.group(1).lower()
            labels = {
                'note': 'Note: ',
                'tip': 'Tip: ',
                'important': 'Important: ',
                'warning': 'Warning: ',
                'caution': 'Caution: '
            }
            return labels.get(alert_type, f'{alert_type.capitalize()}: ')

        text = re.sub(r'^\s*>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*', replace_alert, text, flags=re.MULTILINE | re.IGNORECASE)

        # 3. Strip all remaining blockquote symbols ('> ')
        text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)

        # 4. Handle fenced code blocks
        def replace_code_block(m):
            lang = (m.group(1) or '').strip().lower()
            code = m.group(2).strip()
            lines = [line.strip() for line in code.split('\n') if line.strip()]

            # Single-line command (e.g. npm install, pip install, git checkout)
            if len(lines) == 1 and len(lines[0]) < 80:
                cmd = lines[0]
                if any(cmd.startswith(p) for p in ('npm ', 'pip ', 'git ', 'curl ', 'cargo ', 'docker ', 'python ')):
                    return f"Run: {cmd}.\n"
                # Otherwise, if short code:
                return f"{cmd}.\n"

            # Multi-line code block: summarize instead of reading raw code
            lang_names = {
                'py': 'Python', 'python': 'Python',
                'js': 'JavaScript', 'javascript': 'JavaScript',
                'ts': 'TypeScript', 'typescript': 'TypeScript',
                'html': 'HTML', 'css': 'CSS', 'json': 'JSON',
                'bash': 'shell script', 'sh': 'shell script',
                'cpp': 'C plus plus', 'c': 'C', 'rust': 'Rust',
                'sql': 'SQL query', 'yaml': 'YAML', 'yml': 'YAML'
            }
            speech_lang = lang_names.get(lang, lang.capitalize() if lang else 'code')
            return f"Here is a {speech_lang} snippet shown in the chat.\n"

        text = re.sub(r'```([a-zA-Z0-9_\-\+]*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)

        # 5. Handle Markdown Tables
        text = self._convert_tables_to_speech(text)

        # 6. Handle LaTeX math expressions
        # Block math: $$ ... $$
        text = re.sub(r'\$\$(.*?)\$\$', 'Mathematical equation shown in the message. ', text, flags=re.DOTALL)
        # Inline math: $ ... $
        def replace_inline_math(m):
            math_str = m.group(1).strip()
            # simple substitutions
            math_str = math_str.replace(r'\le', ' less than or equal to ')
            math_str = math_str.replace(r'\ge', ' greater than or equal to ')
            math_str = math_str.replace(r'\approx', ' approximately ')
            math_str = math_str.replace(r'\neq', ' not equal to ')
            math_str = math_str.replace(r'\cdot', ' times ')
            math_str = math_str.replace(r'\times', ' times ')
            math_str = math_str.replace(r'\div', ' divided by ')
            math_str = math_str.replace(r'\pm', ' plus or minus ')
            math_str = math_str.replace('=', ' equals ')
            math_str = math_str.replace('<', ' is less than ')
            math_str = math_str.replace('>', ' is greater than ')
            math_str = math_str.replace('+', ' plus ')
            math_str = math_str.replace('-', ' minus ')
            math_str = re.sub(r'\\[a-zA-Z]+', '', math_str) # strip remaining latex commands
            return f" {math_str} "

        text = re.sub(r'(?<!\$)\$([^\$\n]+?)\$(?!\$)', replace_inline_math, text)

        # 7. Images: ![alt](url) -> ignore or announce alt
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)

        # 8. Markdown Links: [label](url) -> keep only label
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # 9. Bare URLs: https://foo.bar/baz -> domain name
        def replace_url(m):
            url = m.group(0)
            domain_match = re.search(r'https?://(?:www\.)?([^/\s]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                return f"at {domain}"
            return "link"

        text = re.sub(r'https?://\S+', replace_url, text)

        # 10. Inline Code: `identifier_name()` -> clean speech
        def replace_inline_code(m):
            code = m.group(1)
            # Remove trailing parentheses for function calls (e.g. `save()` -> `save`)
            code = re.sub(r'\(\)$', '', code)
            # Replace underscores with spaces for snake_case
            code = code.replace('_', ' ')
            # Split camelCase or PascalCase if appropriate
            code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
            return code

        text = re.sub(r'`([^`]+)`', replace_inline_code, text)

        # 11. Headers: # Header -> Header. (ensure pause with a period)
        def replace_header(m):
            header_text = m.group(2).strip()
            if not header_text:
                return ""
            if not header_text.endswith(('.', '!', '?', ':')):
                header_text += '.'
            return f"\n{header_text}\n"

        text = re.sub(r'^\s*(#{1,6})\s+(.+)$', replace_header, text, flags=re.MULTILINE)

        # 12. Unordered lists: - item / * item -> item.
        def replace_bullet(m):
            bullet_text = m.group(2).strip()
            if not bullet_text:
                return ""
            if not bullet_text.endswith(('.', '!', '?', ';', ':')):
                bullet_text += '.'
            return f"{bullet_text} "

        text = re.sub(r'^\s*([*\-+]|\d+\.)\s+(.+)$', replace_bullet, text, flags=re.MULTILINE)

        # 13. Convert specific conversational action tags (*laughs*, *chuckles*, etc.) into natural speech or micro-cues
        if self.use_micro_samples:
            spoken_actions = {
                'laughs': '[[cue:chuckle]] ',
                'laughing': '[[cue:chuckle]] ',
                'chuckles': '[[cue:chuckle]] ',
                'chuckling': '[[cue:chuckle]] ',
                'giggles': '[[cue:giggle]] ',
                'giggling': '[[cue:giggle]] ',
                'sighs': '[[cue:sigh]] ',
                'sigh': '[[cue:sigh]] ',
                'thinks': 'Hmm... ',
                'thinking': 'Hmm... ',
                'ponders': 'Hmm... ',
                'smiles': 'Well, ',
                'smiling': 'Well, ',
                'clears throat': 'Ahem, ',
                'pause': '... ',
            }
        else:
            spoken_actions = {
                'laughs': 'Heh, ',
                'laughing': 'Heh, ',
                'chuckles': 'Heh, ',
                'chuckling': 'Heh, ',
                'giggles': 'Hehe, ',
                'giggling': 'Hehe, ',
                'sighs': 'Ah... ',
                'sigh': 'Ah... ',
                'thinks': 'Hmm... ',
                'thinking': 'Hmm... ',
                'ponders': 'Hmm... ',
                'smiles': 'Well, ',
                'smiling': 'Well, ',
                'clears throat': 'Ahem, ',
                'pause': '... ',
            }
        action_pattern = r'\*(?:' + '|'.join(re.escape(k) for k in spoken_actions.keys()) + r')\*'
        text = re.sub(action_pattern, lambda m: spoken_actions.get(m.group(0).strip('*').lower(), ''), text, flags=re.IGNORECASE)

        # 14. Text formatting: bold + italic (*** / ___), bold (** / __), italic (* / _), strikethrough (~~)
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
        text = re.sub(r'___(.*?)___', r'\1', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_(.*?)_(?!\w)', r'\1', text)
        text = re.sub(r'~~(.*?)~~', r'\1', text)

        # 15. Expand common spoken abbreviations
        for pattern, replacement in self.abbreviations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 16. Clean decorative Unicode / emojis / symbols (preserve em-dash '—' and pauses)
        # Convert double dashes to em-dash with spacing for natural breath pause
        text = re.sub(r'--+', ' — ', text)
        # Remove arrows and box drawing
        text = re.sub(r'[→⇒➜←↑↓↔↕─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬•★☆✓✔✕✖✗]', ' ', text)
        # Remove emojis (common Unicode ranges)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)

        # 17. Normalize HTML entities
        text = html.unescape(text)

        # 18. Clean whitespace and excess punctuation while preserving pauses (... and —)
        # Normalize repeated punctuation like '!!!' or '???'
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\.{4,}', '...', text)
        # Ensure pause formatting
        text = re.sub(r'\s*—\s*', ' — ', text)

        # 19. Comma pause handling
        if self.extended_comma_pauses:
            text = re.sub(r',(?:\s*—)?\s+(?=[a-zA-Z])', ', — ', text)
        else:
            text = re.sub(r',\s*—\s*', ', ', text)

        # Replace multiple spaces/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        text = text.strip()

        return text

    def _convert_tables_to_speech(self, text: str) -> str:
        """Parses Markdown tables and converts them into spoken English."""
        lines = text.split('\n')
        out_lines = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i].strip()
            # Detect table header row (contains pipes and next line has dashes)
            if '|' in line and i + 1 < n and re.match(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$', lines[i+1].strip()):
                # Collect all table lines
                table_lines = [line]
                i += 2  # skip header separator
                while i < n and '|' in lines[i].strip():
                    if lines[i].strip():
                        table_lines.append(lines[i].strip())
                    i += 1

                # Parse table lines
                rows = []
                for tline in table_lines:
                    cols = [c.strip() for c in tline.strip('|').split('|')]
                    rows.append(cols)

                if len(rows) > 1:
                    headers = rows[0]
                    data_rows = rows[1:]
                    if len(data_rows) <= 3:
                        # Convert to conversational sentence for each row
                        speech_rows = []
                        for row in data_rows:
                            parts = []
                            for idx, val in enumerate(row):
                                if val:
                                    h = headers[idx] if idx < len(headers) else f"Column {idx+1}"
                                    parts.append(f"{h}: {val}")
                            if parts:
                                speech_rows.append(", ".join(parts))
                        spoken_table = "Here are the details: " + "; ".join(speech_rows) + "."
                    else:
                        spoken_table = f"I have organized the {len(data_rows)} rows of data in the table below."
                    out_lines.append(spoken_table)
                continue

            out_lines.append(lines[i])
            i += 1

        return '\n'.join(out_lines)

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits normalized text into sentences suitable for streaming TTS generation, preserving mid-sentence pauses."""
        if not text:
            return []
        
        # Split on sentence end (. ! ? followed by space) but NOT when '.' is part of '...'
        raw_sentences = re.split(r'(?:(?<!\.)\.(?!\.)|[!?])\s+|\n+', text)
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                # restore ending period if missing punctuation
                if not s.endswith(('.', '!', '?', '...')):
                    s += '.'
                sentences.append(s)
        return sentences



normalizer = MarkdownToSpeechNormalizer()
