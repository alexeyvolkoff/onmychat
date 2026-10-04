"""
Asynchronous client for AI Horde (https://aihorde.net) text generation.
Provides streaming emulation and queue status updates for OnMyChat.
"""

import aiohttp
import asyncio
import logging
import re
from typing import AsyncGenerator, Optional, List, Dict, Any

HORDE_BASE_URL = "https://aihorde.net/api/v2"
DEFAULT_CLIENT_AGENT = "OnMyChat:1.0:alexey"
ANONYMOUS_API_KEY = "0000000000"

STOP_TOKENS_RE = re.compile(
    r'(<\|im_end\|>|<\|im_start\|>|<\|end_of_text\|>|<\|endoftext\|>|<\|eot_id\|>|<\|eom_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|<\/s>|<s>|<end_of_turn>|<start_of_turn>|\[DONE\])',
    re.IGNORECASE
)

# Default models known for high quality RP / Creative chat
DEFAULT_HORDE_MODELS = [
    "aphrodite/TheDrummer/Behemoth-X-123B-v2.1",
    "koboldcpp/L3-8B-Stheno-v3.2",
    "aphrodite/TheDrummer/Skyfall-31B-v4.2"
]


def format_messages_for_horde(messages: List[Dict[str, str]]) -> str:
    """
    Format a list of chat messages [{'role': ..., 'content': ...}]
    into a ChatML string widely supported by KoboldCpp / Aphrodite.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    lines.append("<|im_start|>assistant\n")
    return "\n".join(lines)


async def get_online_text_models(limit: int = 20) -> List[Dict[str, Any]]:
    """Fetch currently active text models on AI Horde."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{HORDE_BASE_URL}/status/models?type=text",
                headers={"Client-Agent": DEFAULT_CLIENT_AGENT},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    models = await resp.json()
                    models.sort(key=lambda m: (m.get("count", 0), m.get("performance", 0)), reverse=True)
                    return models[:limit]
    except Exception as e:
        logging.warning(f"[Horde] Failed to fetch models: {e}")
    return []


async def cancel_horde_task(task_id: str, api_key: str = ANONYMOUS_API_KEY, client_agent: str = DEFAULT_CLIENT_AGENT):
    """Cancel an active generation task on AI Horde."""
    if not task_id:
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{HORDE_BASE_URL}/generate/text/status/{task_id}",
                headers={
                    "apikey": api_key,
                    "Client-Agent": client_agent
                },
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                logging.info(f"[Horde] Cancelled task {task_id}: status={resp.status}")
    except Exception as e:
        logging.warning(f"[Horde] Error cancelling task {task_id}: {e}")


async def horde_generate_stream(
    payload: Dict[str, Any],
    api_key: str = ANONYMOUS_API_KEY,
    preferred_model: Optional[str] = None,
    client_agent: str = DEFAULT_CLIENT_AGENT,
    check_interval: float = 1.5,
    max_wait_seconds: int = 180
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Submits a text generation request to AI Horde and yields:
    - Status updates: {"status": "..."}
    - Emulated token stream: {"message": {"role": "assistant", "content": delta}}
    - Final completion: {"done": True, "done_reason": "stop", "eval_count": ..., "worker_info": ...}
    """
    messages = payload.get("messages", [])
    if messages:
        prompt = format_messages_for_horde(messages)
    else:
        prompt = payload.get("prompt", "")

    options = payload.get("options", {})
    is_anon = (not api_key) or (api_key == ANONYMOUS_API_KEY)

    # Horde requires Kudos upfront for requests > 512 tokens if anonymous
    requested_length = options.get("num_predict", 500 if is_anon else 1024)
    if is_anon and requested_length > 500:
        requested_length = 500

    requested_ctx = options.get("num_ctx", 2048 if is_anon else 8192)
    if is_anon and requested_ctx > 2048:
        requested_ctx = 2048

    gen_params = {
        "n": 1,
        "max_context_length": requested_ctx,
        "max_length": requested_length,
        "temperature": options.get("temperature", 0.85),
        "top_p": options.get("top_p", 0.9),
        "rep_pen": options.get("repeat_penalty", 1.1),
        "singleline": False,
        "stop_sequence": [
            "<|im_end|>", "<|im_start|>", "<|end_of_text|>", "<|endoftext|>",
            "<|eot_id|>", "<|eom_id|>", "</s>", "<end_of_turn>"
        ]
    }

    # Selected models list
    models = []
    if preferred_model:
        models.append(preferred_model)
    elif payload.get("model"):
        models.append(payload.get("model"))
    else:
        models.extend(DEFAULT_HORDE_MODELS)

    req_body = {
        "prompt": prompt,
        "params": gen_params,
        "models": models,
        "trusted_workers": False
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": api_key or ANONYMOUS_API_KEY,
        "Client-Agent": client_agent
    }

    task_id = None
    try:
        timeout = aiohttp.ClientTimeout(total=max_wait_seconds, connect=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # 1. Submit request
            async with session.post(f"{HORDE_BASE_URL}/generate/text/async", json=req_body, headers=headers) as resp:
                if resp.status not in (200, 202):
                    err_text = await resp.text()
                    logging.error(f"[Horde] Submit failed ({resp.status}): {err_text}")
                    yield {"error": f"AI Horde error ({resp.status}): {err_text[:120]}", "done": True}
                    return

                res_json = await resp.json()
                task_id = res_json.get("id")

            if not task_id:
                yield {"error": "AI Horde did not return a task ID", "done": True}
                return

            logging.info(f"[Horde] Task submitted: {task_id}, requested models: {models}")
            yield {"status": "Орда: поиск свободной ноды..."}

            # 2. Polling loop
            elapsed = 0.0
            last_queue = None
            while elapsed < max_wait_seconds:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                status_url = f"{HORDE_BASE_URL}/generate/text/status/{task_id}"
                async with session.get(status_url, headers={"Client-Agent": client_agent}) as s_resp:
                    if s_resp.status != 200:
                        continue
                    st = await s_resp.json()

                if st.get("faulted"):
                    logging.warning(f"[Horde] Task {task_id} faulted")
                    yield {"error": "Генерация в Орде прервана с ошибкой воркера", "done": True}
                    return

                if st.get("is_possible") is False:
                    logging.warning(f"[Horde] Task {task_id} unsatisfiable: no workers available")
                    yield {"error": "В Орде сейчас нет свободных воркеров для выбранной модели", "done": True}
                    return

                if st.get("done") and st.get("generations"):
                    gen = st["generations"][0]
                    full_text = gen.get("text", "")
                    full_text = STOP_TOKENS_RE.sub("", full_text).strip()
                    worker_name = gen.get("worker_name", "unknown")
                    actual_model = gen.get("model", models[0] if models else "unknown")
                    logging.info(f"[Horde] Generation completed by '{worker_name}' [{actual_model}], length: {len(full_text)}")

                    # Stream text in smooth chunks to emulate typing
                    # Using sentence or small token splits with micro-delay
                    chunk_size = 4  # words per chunk
                    words = full_text.split(" ")
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        if i + chunk_size < len(words):
                            chunk += " "
                        yield {
                            "message": {
                                "role": "assistant",
                                "content": chunk
                            }
                        }
                        await asyncio.sleep(0.02)

                    yield {
                        "done": True,
                        "done_reason": "stop",
                        "eval_count": len(words),
                        "worker_info": f"{worker_name} [{actual_model}]"
                    }
                    return

                # Update queue / wait status
                q_pos = st.get("queue_position", 0)
                wait_time = st.get("wait_time", 0)
                if q_pos != last_queue or wait_time > 0:
                    last_queue = q_pos
                    if q_pos > 0:
                        yield {"status": f"Орда: позиция в очереди {q_pos} (~{wait_time}с)"}
                    else:
                        yield {"status": "Орда: воркер генерирует ответ..."}

            yield {"error": f"Превышено время ожидания ответа от Орды ({max_wait_seconds}с)", "done": True}

    except asyncio.CancelledError:
        logging.info(f"[Horde] Stream cancelled by client, aborting task {task_id}")
        if task_id:
            asyncio.create_task(cancel_horde_task(task_id, api_key, client_agent))
        raise
    except Exception as e:
        logging.error(f"[Horde] Unexpected stream exception: {e}")
        yield {"error": f"AI Horde connection error: {str(e)}", "done": True}
    finally:
        pass
