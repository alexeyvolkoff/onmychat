"""
AI Horde Dreamer Worker & Image Client for OnMyChat (`horde_dreamer.py`).

Features:
1. Built-in AI Horde Image Worker ("Dreamer"):
   - Unlimited 24/7 Local SDXL generation via local ComfyUI (`flow.json` DMD2 pipeline)
     when local ComfyUI queue is idle.
   - Optional daily-capped Modal Krea 2 (`Krea2-Turbo_fp8`) generation (default max 10/day,
     auto-reset at UTC midnight; once reached, `Krea2-Turbo_fp8` is automatically
     removed from `/v2/generate/pop` models for the rest of the day).
2. AI Horde Image Generation Client (`horde_generate_image`) & `ai_mode` normalization hook.
3. Owner-only management console at `/horde` and JSON endpoints under `/horde/api/*`
   and `/assistant/horde_worker`.
"""

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from typing import Any, Optional

import aiohttp
from PIL import Image
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

import horde_client
import user_context
from config import SETTINGS

logger = logging.getLogger("horde_dreamer")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "horde_dreamer_data.json")


def is_modal_enabled() -> bool:
    return bool(str(SETTINGS.get("MODAL_COMFY_URL", "")).strip())

HORDE_API_BASE = "https://aihorde.net/api/v2"
BRIDGE_AGENT = "AI Horde Worker reGen:18.0.0:https://github.com/alexeyvolkoff/onmychat"
CLIENT_AGENT = "OnMyChat:1.0:https://github.com/alexeyvolkoff/onmychat"
PROMPT_SEPARATOR = "###"

# Mapping from official AI Horde model names to local ComfyUI SDXL checkpoints
LOCAL_SDXL_MODEL_CATALOG: dict[str, dict[str, str]] = {
    "CyberRealistic Pony": {
        "ckpt": "cyberrealisticPony_v180Coreshift.safetensors",
        "description": "Top-4 AI Horde model (Pony/SDXL Photorealism) — stays warm in VRAM",
    },
    "Pony Realism": {
        "ckpt": "cyberrealisticPony_v180Coreshift.safetensors",
        "description": "Uses the same cyberrealisticPony checkpoint (zero VRAM swap)",
    },
    "AlbedoBase XL (SDXL)": {
        "ckpt": "CyberRealisticXLPlay_V10.0_FP16.safetensors",
        "description": "High-demand general SDXL photorealism checkpoint",
    },
    "SDXL 1.0": {
        "ckpt": "CyberRealisticXLPlay_V10.0_FP16.safetensors",
        "description": "Standard SDXL 1.0 requests routed to CyberRealisticXLPlay",
    },
    "WAI-NSFW-illustrious-SDXL": {
        "ckpt": "CyberIllustrious_V12_FP16.safetensors",
        "description": "Illustrious anime/illustration SDXL checkpoint",
    },
}

KREA_HORDE_MODEL_NAME = "Krea2-Turbo_fp8"

_data_lock = threading.Lock()
_worker_lock = asyncio.Lock()
_worker_task: Optional[asyncio.Task] = None
_wake_event = asyncio.Event()

# Live in-memory telemetry
_runtime_status: dict[str, Any] = {
    "state": "stopped",  # stopped | idle_polling | waiting_local_busy | rendering | error | paused
    "current_job_id": None,
    "current_model": None,
    "current_backend": None,
    "last_pop_ts": 0.0,
    "last_submit_ts": 0.0,
    "last_error": "",
    "last_skipped": {},
    "active_models_offered": [],
    "user_info": None,
    "user_info_ts": 0.0,
}


def _utc_today_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _default_data() -> dict[str, Any]:
    raw_enabled = str(SETTINGS.get("HORDE_WORKER_ENABLED", "false")).strip().lower() in ("1", "true", "yes", "on")
    raw_nsfw = str(SETTINGS.get("HORDE_WORKER_NSFW", "true")).strip().lower() in ("1", "true", "yes", "on")
    raw_max_pixels = 1024 * 1024
    try:
        raw_max_pixels = max(262144, int(SETTINGS.get("HORDE_WORKER_MAX_PIXELS", 1048576)))
    except Exception:
        raw_max_pixels = 1024 * 1024
    raw_limit = 10
    try:
        raw_limit = int(SETTINGS.get("HORDE_KREA_DAILY_LIMIT", 10))
    except Exception:
        raw_limit = 10
    raw_key = str(
        SETTINGS.get("HORDE_WORKER_API_KEY")
        or (SETTINGS.get("HORDE_API_KEY") if SETTINGS.get("HORDE_API_KEY") != "0000000000" else "")
        or ""
    ).strip()
    raw_name = str(SETTINGS.get("HORDE_WORKER_NAME") or "OnMyChat-Dreamer").strip() or "OnMyChat-Dreamer"
    raw_models_str = str(SETTINGS.get("HORDE_WORKER_MODELS") or "").strip()
    if raw_models_str:
        default_models = [
            m.strip() for m in raw_models_str.split(",")
            if m.strip() in LOCAL_SDXL_MODEL_CATALOG
        ] or ["CyberRealistic Pony"]
    else:
        default_models = ["CyberRealistic Pony"]

    return {
        "enabled": raw_enabled,
        "worker_name": raw_name,
        "api_key": raw_key,
        "nsfw": raw_nsfw,
        "max_pixels": raw_max_pixels,
        "poll_interval_idle": 10.0,
        "poll_interval_busy": 2.0,
        "sdxl_enabled": True,
        "sdxl_models": default_models,
        "krea_enabled": True,
        "krea_daily_limit": raw_limit,
        "krea_daily_date": _utc_today_str(),
        "krea_daily_count": 0,
        "total_jobs_completed": 0,
        "sdxl_jobs_completed": 0,
        "krea_jobs_completed": 0,
        "total_jobs_faulted": 0,
        "total_kudos_earned": 0.0,
        "total_megapixelsteps": 0.0,
        "recent_jobs": [],
    }


def load_dreamer_data() -> dict[str, Any]:
    with _data_lock:
        defaults = _default_data()
        if not os.path.isfile(DATA_PATH):
            return defaults
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return defaults
            merged = dict(defaults)
            merged.update(loaded)
            if not merged.get("api_key") and defaults.get("api_key"):
                merged["api_key"] = defaults["api_key"]
            # Roll over daily counter if UTC date changed
            today = _utc_today_str()
            if merged.get("krea_daily_date") != today:
                merged["krea_daily_date"] = today
                merged["krea_daily_count"] = 0
            return merged
        except Exception as e:
            logger.error(f"[HordeDreamer] Failed reading {DATA_PATH}: {e}")
            return defaults


def save_dreamer_data(data: dict[str, Any]) -> None:
    with _data_lock:
        today = _utc_today_str()
        if data.get("krea_daily_date") != today:
            data["krea_daily_date"] = today
            data["krea_daily_count"] = 0
        if isinstance(data.get("recent_jobs"), list) and len(data["recent_jobs"]) > 40:
            data["recent_jobs"] = data["recent_jobs"][:40]
        tmp = DATA_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        os.replace(tmp, DATA_PATH)


def get_effective_api_key(data: Optional[dict[str, Any]] = None) -> str:
    if data is None:
        data = load_dreamer_data()
    k = str(data.get("api_key") or "").strip()
    if k and k != "0000000000":
        return k
    cfg_k = str(SETTINGS.get("HORDE_WORKER_API_KEY") or "").strip()
    if cfg_k and cfg_k != "0000000000":
        return cfg_k
    fallback_k = str(SETTINGS.get("HORDE_API_KEY") or "").strip()
    if fallback_k and fallback_k != "0000000000":
        return fallback_k
    return ""


def get_active_models_for_pop(data: Optional[dict[str, Any]] = None) -> list[str]:
    """
    Returns the list of AI Horde models this worker currently offers for `/v2/generate/pop`:
    - Unlimited local SDXL models when `sdxl_enabled` is True.
    - `Krea2-Turbo_fp8` ONLY when `krea_enabled` is True, Modal is configured,
      and today's `krea_daily_count < krea_daily_limit` (or `krea_daily_limit < 0` for unlimited).
    """
    if data is None:
        data = load_dreamer_data()
    today = _utc_today_str()
    if data.get("krea_daily_date") != today:
        data["krea_daily_date"] = today
        data["krea_daily_count"] = 0
        save_dreamer_data(data)

    models: list[str] = []
    if data.get("sdxl_enabled", True):
        for m in data.get("sdxl_models") or ["CyberRealistic Pony"]:
            m_str = str(m).strip()
            if m_str and m_str in LOCAL_SDXL_MODEL_CATALOG and m_str not in models:
                models.append(m_str)

    if data.get("krea_enabled", True) and is_modal_enabled():
        limit = int(data.get("krea_daily_limit", 10))
        used = int(data.get("krea_daily_count", 0))
        if limit < 0 or used < limit:
            if KREA_HORDE_MODEL_NAME not in models:
                models.append(KREA_HORDE_MODEL_NAME)

    return models


async def is_local_comfy_idle() -> bool:
    """
    Returns True if local ComfyUI is reachable AND its running + pending queues are completely empty,
    and Autopilot is not currently generating a post.
    """
    try:
        from extensions import autopilot
        if getattr(autopilot, "_gen_lock", None) is not None and autopilot._gen_lock.locked():
            return False
    except Exception:
        pass

    comfy_url = str(SETTINGS.get("COMFY_API_URL", "http://localhost:8188")).rstrip("/")
    if not comfy_url:
        return False

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2.5)) as session:
            async with session.get(f"{comfy_url}/queue") as resp:
                if resp.status != 200:
                    return False
                qdata = await resp.json()
                running = qdata.get("queue_running") or []
                pending = qdata.get("queue_pending") or []
                return len(running) == 0 and len(pending) == 0
    except Exception:
        return False


def _split_horde_prompt(raw_prompt: str) -> tuple[str, str]:
    """Splits AI Horde prompt string on '###' into (positive_prompt, negative_prompt)."""
    if not raw_prompt:
        return "", ""
    if PROMPT_SEPARATOR in raw_prompt:
        pos, _, neg = raw_prompt.partition(PROMPT_SEPARATOR)
        return pos.strip(), neg.strip()
    return raw_prompt.strip(), ""


def _build_sdxl_horde_workflow(
    positive_prompt: str,
    negative_prompt: str,
    ckpt_name: str,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    """Builds a clean, fast 7-step DMD2 SDXL ComfyUI workflow from `flow.json`."""
    workflow_path = str(SETTINGS.get("WORKFLOW_PATH", "flow.json"))
    if not os.path.isabs(workflow_path):
        workflow_path = os.path.join(ROOT_DIR, workflow_path)
    with open(workflow_path, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Set checkpoint
    if "127" in wf and "inputs" in wf["127"]:
        wf["127"]["inputs"]["ckpt_name"] = ckpt_name

    # Disable all character/style LoRAs in Power Lora Loader (node "103")
    if "103" in wf and "inputs" in wf["103"]:
        for k, v in wf["103"]["inputs"].items():
            if isinstance(v, dict) and ("lora" in v or "name" in v):
                v["on"] = False

    base_neg = (
        "worst quality, low quality, blurry, deformed, bad anatomy, bad hands, extra digits, "
        "missing fingers, watermark, signature"
    )
    eff_neg = f"{negative_prompt}, {base_neg}".strip(", ") if negative_prompt else base_neg
    eff_pos = positive_prompt.strip() or "masterpiece, best quality, ultra detailed"

    if "4" in wf and "inputs" in wf["4"]:
        wf["4"]["inputs"]["text"] = eff_neg
        wf["4"]["inputs"]["clip"] = ["103", 1]
    if "85" in wf and "inputs" in wf["85"]:
        wf["85"]["inputs"]["text"] = eff_pos
        wf["85"]["inputs"]["clip"] = ["103", 1]

    w_clamped = max(512, min(1536, (int(width or 1024) // 8) * 8))
    h_clamped = max(512, min(1536, (int(height or 1024) // 8) * 8))
    if "6" in wf and "inputs" in wf["6"]:
        wf["6"]["inputs"]["width"] = w_clamped
        wf["6"]["inputs"]["height"] = h_clamped

    if "5" in wf and "inputs" in wf["5"]:
        wf["5"]["inputs"]["seed"] = int(seed)
        wf["5"]["inputs"]["steps"] = 7
        wf["5"]["inputs"]["cfg"] = 1.45
        wf["5"]["inputs"]["sampler_name"] = "lcm"
        wf["5"]["inputs"]["scheduler"] = "exponential"

    return wf


def _build_krea_horde_workflow(
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 8,
    cfg: float = 1.5,
) -> dict[str, Any]:
    """Builds the Krea 2 UNET workflow for Modal GPU execution."""
    unet_model = SETTINGS.get("KREA_MODEL", "CyberRealistic_Krea2_V4.0_BF16.safetensors")
    clip_model = SETTINGS.get("KREA_CLIP", "qwen3-vl-4b-heretic.safetensors")
    vae_model = SETTINGS.get("KREA_VAE", "qwen_image_vae.safetensors")

    w_clamped = max(512, min(1536, (int(width or 1024) // 8) * 8))
    h_clamped = max(512, min(1536, (int(height or 1024) // 8) * 8))
    eff_steps = max(4, min(12, int(steps or 8)))
    eff_cfg = max(1.0, min(3.5, float(cfg or 1.5)))
    base_neg = "worst quality, low quality, blurry, bad anatomy, bad hands, deformed, watermark, signature"
    eff_neg = f"{negative_prompt}, {base_neg}".strip(", ") if negative_prompt else base_neg
    eff_pos = positive_prompt.strip() or "masterpiece, photorealistic, high detail"

    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_model, "type": "krea2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_model},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": eff_pos, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": eff_neg, "clip": ["2", 0]},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": w_clamped, "height": h_clamped, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": eff_steps,
                "cfg": eff_cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "generated/horde_krea", "images": ["8", 0]},
        },
    }


async def _run_local_comfy_silent(workflow: dict[str, Any], timeout: int = 180) -> Optional[bytes]:
    """
    Executes a workflow on local ComfyUI (`COMFY_API_URL`) without emitting WebSocket progress
    to chat users, and cleans up temporary output files from disk.
    """
    comfy_url = str(SETTINGS.get("COMFY_API_URL", "http://localhost:8188")).rstrip("/")
    comfy_output_dir = str(SETTINGS.get("COMFY_OUTPUT_DIR", ""))
    client_id = str(uuid.uuid4())
    ws_url = f"{comfy_url.replace('http', 'ws')}/ws?clientId={client_id}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=15.0) as ws:
                payload = {"prompt": workflow, "client_id": client_id}
                async with session.post(f"{comfy_url}/prompt", json=payload, timeout=aiohttp.ClientTimeout(total=15.0)) as resp:
                    if resp.status != 200:
                        err_txt = await resp.text()
                        logger.error(f"[HordeDreamer] ComfyUI /prompt HTTP {resp.status}: {err_txt[:200]}")
                        return None
                    resp_data = await resp.json()
                    prompt_id = resp_data.get("prompt_id")
                    if not prompt_id:
                        logger.error(f"[HordeDreamer] ComfyUI /prompt missing prompt_id: {resp_data}")
                        return None

                final_image_data: Optional[bytes] = None
                loop_start = time.time()

                async for msg in ws:
                    if time.time() - loop_start > timeout:
                        logger.error(f"[HordeDreamer] Timeout ({timeout}s) waiting for local ComfyUI prompt {prompt_id}")
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        data_content = data.get("data") or {}

                        if msg_type == "executing":
                            if data_content.get("node") is None and data_content.get("prompt_id") == prompt_id:
                                break
                        elif msg_type == "execution_error" and data_content.get("prompt_id") == prompt_id:
                            logger.error(f"[HordeDreamer] Local ComfyUI execution error: {data_content}")
                            break
                        elif msg_type == "executed" and data_content.get("prompt_id") == prompt_id:
                            outputs = data_content.get("output") or {}
                            if isinstance(outputs, dict) and "images" in outputs:
                                for image in outputs.get("images") or []:
                                    filename = image.get("filename")
                                    subfolder = image.get("subfolder", "")
                                    img_type = image.get("type", "output")
                                    params = {"filename": filename, "subfolder": subfolder, "type": img_type}
                                    async with session.get(f"{comfy_url}/view", params=params) as img_resp:
                                        if img_resp.status == 200:
                                            final_image_data = await img_resp.read()
                                    # Keep the latest 30 temp preview files for ComfyUI Queue sidebar,
                                    # and prune older ones so ComfyUI/temp/ never fills the disk.
                                    if img_type == "temp":
                                        _prune_comfy_temp(keep=30)
                return final_image_data
    except Exception as e:
        logger.error(f"[HordeDreamer] Error in _run_local_comfy_silent: {e}")
        return None


def _prune_comfy_temp(keep: int = 30) -> None:
    """Keep only the newest `keep` preview PNGs in ComfyUI/temp/ so the Queue UI works without filling disk."""
    try:
        temp_dir = os.path.expanduser("~/projects/ComfyUI/temp")
        if not os.path.isdir(temp_dir):
            return
        entries = []
        for name in os.listdir(temp_dir):
            if name.startswith("ComfyUI_temp_") and name.endswith(".png"):
                full = os.path.join(temp_dir, name)
                try:
                    entries.append((os.path.getmtime(full), full))
                except OSError:
                    pass
        entries.sort(reverse=True)
        for _, old_path in entries[keep:]:
            try:
                os.remove(old_path)
            except OSError:
                pass
    except Exception:
        pass


async def _run_modal_krea_silent(workflow: dict[str, Any], timeout: int = 180) -> Optional[bytes]:
    """
    Dispatches a Krea 2 workflow to remote Modal GPU without emitting progress events
    to the node's interactive chat stream.
    """
    url = str(SETTINGS.get("MODAL_COMFY_URL", "")).strip()
    if not url:
        return None

    headers = {"Content-Type": "application/json"}
    token = str(SETTINGS.get("MODAL_AUTH_TOKEN", "")).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    stream_endpoint = url.rstrip("/") + "/generate_stream"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                stream_endpoint,
                json={"workflow": workflow},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    err_txt = await resp.text()
                    logger.error(f"[HordeDreamer] Modal HTTP {resp.status}: {err_txt[:200]}")
                    return None
                buf = bytearray()
                image_bytes: Optional[bytes] = None
                async for chunk in resp.content.iter_any():
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    while True:
                        nl = buf.find(b"\n")
                        if nl < 0:
                            break
                        line = bytes(buf[:nl]).strip()
                        del buf[: nl + 1]
                        if not line:
                            continue
                        try:
                            ev = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        if ev.get("event") in ("done", "result") and ev.get("image_b64"):
                            image_bytes = base64.b64decode(ev["image_b64"])
                        elif ev.get("event") == "error":
                            logger.error(f"[HordeDreamer] Modal stream error: {ev.get('error') or ev.get('detail')}")
                if buf.strip() and not image_bytes:
                    try:
                        ev = json.loads(bytes(buf).strip().decode("utf-8"))
                        if ev.get("event") in ("done", "result") and ev.get("image_b64"):
                            image_bytes = base64.b64decode(ev["image_b64"])
                    except Exception:
                        pass
                return image_bytes
    except Exception as e:
        logger.error(f"[HordeDreamer] Error in _run_modal_krea_silent: {e}")
        return None


def _convert_png_to_webp(img_bytes: bytes, quality: int = 92) -> bytes:
    """Converts PNG/JPEG bytes into AI Horde standard WebP bytes."""
    with Image.open(io.BytesIO(img_bytes)) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="WEBP", quality=quality, method=4)
        return out.getvalue()


async def _upload_webp_to_r2(session: aiohttp.ClientSession, r2_url: str, webp_bytes: bytes) -> bool:
    """
    Uploads WebP image bytes to AI Horde's presigned Cloudflare R2 URL.
    Critically uses `skip_auto_headers=["Content-Type"]` so S3 signature verification succeeds.
    """
    for attempt in range(2):
        try:
            async with session.put(
                r2_url,
                data=webp_bytes,
                skip_auto_headers=["Content-Type", "content-type"],
                timeout=aiohttp.ClientTimeout(total=45.0),
            ) as resp:
                if resp.status in (200, 201, 204):
                    return True
                body = await resp.text()
                logger.warning(f"[HordeDreamer] R2 PUT attempt {attempt + 1} returned HTTP {resp.status}: {body[:200]}")
        except Exception as e:
            logger.warning(f"[HordeDreamer] R2 PUT attempt {attempt + 1} error: {e}")
        await asyncio.sleep(1.0)
    return False


async def fetch_horde_user_info(api_key: Optional[str] = None, force: bool = False) -> Optional[dict[str, Any]]:
    """Fetches `/api/v2/find_user` for the configured worker API key (cached for 120s unless forced)."""
    now = time.time()
    if not force and _runtime_status.get("user_info") and (now - float(_runtime_status.get("user_info_ts") or 0)) < 120.0:
        return _runtime_status["user_info"]

    eff_key = api_key or get_effective_api_key()
    if not eff_key:
        return None

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0)) as session:
            async with session.get(
                f"{HORDE_API_BASE}/find_user",
                headers={"apikey": eff_key, "Client-Agent": CLIENT_AGENT},
            ) as resp:
                if resp.status == 200:
                    uinfo = await resp.json()
                    _runtime_status["user_info"] = uinfo
                    _runtime_status["user_info_ts"] = now
                    return uinfo
                else:
                    txt = await resp.text()
                    logger.warning(f"[HordeDreamer] /find_user returned HTTP {resp.status}: {txt[:160]}")
    except Exception as e:
        logger.warning(f"[HordeDreamer] /find_user error: {e}")
    return _runtime_status.get("user_info")


async def run_single_dreamer_cycle(ignore_enabled: bool = False) -> dict[str, Any]:
    """
    Executes a single `pop -> render -> upload R2 -> submit` cycle.
    Protected by `_worker_lock` so manual trigger and background loop never overlap.
    """
    async with _worker_lock:
        data = load_dreamer_data()
        if not ignore_enabled and not data.get("enabled", True):
            _runtime_status["state"] = "paused"
            return {"status": "paused", "detail": "Worker is paused"}

        api_key = get_effective_api_key(data)
        if not api_key:
            _runtime_status["state"] = "error"
            _runtime_status["last_error"] = "Missing HORDE_WORKER_API_KEY"
            return {"status": "error", "detail": "Missing HORDE_WORKER_API_KEY"}

        models_to_offer = get_active_models_for_pop(data)
        _runtime_status["active_models_offered"] = models_to_offer
        if not models_to_offer:
            _runtime_status["state"] = "paused"
            return {"status": "no_models", "detail": "No active models available to offer (check SDXL or Krea daily limit)"}

        # Check local ComfyUI idle status if any SDXL model is offered
        has_sdxl_offer = any(m != KREA_HORDE_MODEL_NAME for m in models_to_offer)
        if has_sdxl_offer and not await is_local_comfy_idle():
            _runtime_status["state"] = "waiting_local_busy"
            return {"status": "local_busy", "detail": "Local ComfyUI or Autopilot is currently busy"}

        worker_name = str(data.get("worker_name") or "OnMyChat-Dreamer").strip()[:100]
        max_pixels = int(data.get("max_pixels") or (1024 * 1024))
        nsfw_flag = bool(data.get("nsfw", True))

        pop_payload = {
            "name": worker_name,
            "models": models_to_offer,
            "max_pixels": max_pixels,
            "nsfw": nsfw_flag,
            "bridge_agent": BRIDGE_AGENT,
            "threads": 1,
            "require_upfront_kudos": False,
            "amount": 1,
            "allow_img2img": False,
            "allow_painting": False,
            "allow_unsafe_ipaddr": True,
            "allow_post_processing": False,
            "allow_controlnet": False,
            "allow_extended_controlnet": False,
            "allow_sdxl_controlnet": False,
            "allow_lora": False,
            "limit_max_steps": False,
        }

        headers = {
            "Content-Type": "application/json",
            "apikey": api_key,
            "Client-Agent": BRIDGE_AGENT,
        }

        _runtime_status["state"] = "idle_polling"
        _runtime_status["last_pop_ts"] = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60.0)) as session:
                async with session.post(f"{HORDE_API_BASE}/generate/pop", json=pop_payload, headers=headers) as pop_resp:
                    if pop_resp.status != 200:
                        err_text = await pop_resp.text()
                        _runtime_status["state"] = "error"
                        if "UnsafeIP" in err_text:
                            _runtime_status["last_error"] = (
                                "UnsafeIP (403): AI Horde flagged ISP IP as VPN. "
                                "Ask a mod in AI Horde Discord (#horde-support) to enable the 'vpn' flag on ShinyLark#605378."
                            )
                            logger.warning(f"[HordeDreamer] {_runtime_status['last_error']}")
                            return {"status": "unsafe_ip", "detail": _runtime_status["last_error"]}
                        if "Maintenance mode" in err_text or "WorkerMaintenance" in err_text:
                            uinfo = await fetch_horde_user_info(api_key, force=True)
                            for wid in (uinfo or {}).get("worker_ids") or []:
                                try:
                                    async with session.put(
                                        f"{HORDE_API_BASE}/workers/{wid}",
                                        json={"maintenance": False},
                                        headers=headers,
                                    ) as m_resp:
                                        logger.info(f"[HordeDreamer] Cleared maintenance mode on worker {wid}: HTTP {m_resp.status}")
                                except Exception as m_err:
                                    logger.warning(f"[HordeDreamer] Failed clearing maintenance on {wid}: {m_err}")
                        _runtime_status["last_error"] = f"Pop HTTP {pop_resp.status}: {err_text[:180]}"
                        logger.warning(f"[HordeDreamer] Pop failed ({pop_resp.status}): {err_text[:200]}")
                        return {"status": "error", "detail": _runtime_status["last_error"]}

                    pop_data = await pop_resp.json()

                ids = pop_data.get("ids") or ([pop_data["id"]] if pop_data.get("id") else [])
                if not ids:
                    skipped = pop_data.get("skipped") or {}
                    _runtime_status["last_skipped"] = skipped
                    _runtime_status["last_error"] = ""
                    return {"status": "empty_queue", "skipped": skipped, "models_offered": models_to_offer}

                job_id = str(ids[0])
                r2_uploads = pop_data.get("r2_uploads") or ([pop_data["r2_upload"]] if pop_data.get("r2_upload") else [])
                r2_url = str(r2_uploads[0]) if r2_uploads else ""
                job_model = str(pop_data.get("model") or models_to_offer[0])
                gen_payload = pop_data.get("payload") or {}

                raw_prompt = str(gen_payload.get("prompt") or "")
                pos_prompt, neg_prompt = _split_horde_prompt(raw_prompt)
                width = int(gen_payload.get("width") or 1024)
                height = int(gen_payload.get("height") or 1024)
                steps = int(gen_payload.get("ddim_steps") or gen_payload.get("steps") or 20)
                cfg_scale = float(gen_payload.get("cfg_scale") or 7.0)

                raw_seed = gen_payload.get("seed")
                try:
                    seed_int = int(str(raw_seed).strip()) if raw_seed is not None and str(raw_seed).strip() != "" else 0
                except Exception:
                    seed_int = 0
                if seed_int <= 0:
                    seed_int = random.randint(1, 2147483646)
                else:
                    seed_int = seed_int % 2147483647 or 1

                is_krea_job = (job_model == KREA_HORDE_MODEL_NAME)
                backend_label = "Modal Krea 2" if is_krea_job else "Local SDXL"

                _runtime_status["state"] = "rendering"
                _runtime_status["current_job_id"] = job_id
                _runtime_status["current_model"] = job_model
                _runtime_status["current_backend"] = backend_label

                logger.info(
                    f"[HordeDreamer] Popped job {job_id} | model='{job_model}' ({backend_label}) | "
                    f"{width}x{height} | steps={steps} | prompt='{pos_prompt[:70]}'"
                )

                t0 = time.time()
                png_bytes: Optional[bytes] = None

                if is_krea_job:
                    # Enforce hard daily limit for Krea 2 on Modal (counted on invocation so failed calls also count toward budget)
                    fresh_data = load_dreamer_data()
                    k_limit = int(fresh_data.get("krea_daily_limit", 10))
                    k_used = int(fresh_data.get("krea_daily_count", 0))
                    if k_limit >= 0 and k_used >= k_limit:
                        logger.warning(f"[HordeDreamer] Krea daily limit ({k_used}/{k_limit}) reached; faulting job {job_id}")
                        png_bytes = None
                    else:
                        fresh_data["krea_daily_count"] = k_used + 1
                        save_dreamer_data(fresh_data)
                        wf = _build_krea_horde_workflow(
                            positive_prompt=pos_prompt,
                            negative_prompt=neg_prompt,
                            width=width,
                            height=height,
                            seed=seed_int,
                            steps=steps,
                            cfg=cfg_scale,
                        )
                        png_bytes = await _run_modal_krea_silent(wf)
                else:
                    cat_entry = LOCAL_SDXL_MODEL_CATALOG.get(job_model) or LOCAL_SDXL_MODEL_CATALOG["CyberRealistic Pony"]
                    ckpt_name = cat_entry["ckpt"]
                    wf = _build_sdxl_horde_workflow(
                        positive_prompt=pos_prompt,
                        negative_prompt=neg_prompt,
                        ckpt_name=ckpt_name,
                        width=width,
                        height=height,
                        seed=seed_int,
                    )
                    png_bytes = await _run_local_comfy_silent(wf)

                render_sec = round(time.time() - t0, 2)

                if not png_bytes:
                    # Report faulted so AI Horde immediately re-queues the job for another worker
                    fault_body = {
                        "id": job_id,
                        "generation": "faulted",
                        "state": "faulted",
                        "seed": seed_int,
                        "censored": False,
                    }
                    try:
                        await session.post(f"{HORDE_API_BASE}/generate/submit", json=fault_body, headers=headers)
                    except Exception:
                        pass
                    upd = load_dreamer_data()
                    upd["total_jobs_faulted"] = int(upd.get("total_jobs_faulted", 0)) + 1
                    upd.setdefault("recent_jobs", []).insert(0, {
                        "id": job_id,
                        "ts": int(time.time()),
                        "model": job_model,
                        "backend": backend_label,
                        "resolution": f"{width}x{height}",
                        "steps": steps,
                        "duration_sec": render_sec,
                        "kudos": 0.0,
                        "status": "faulted",
                        "prompt_preview": pos_prompt[:100],
                    })
                    save_dreamer_data(upd)
                    _runtime_status["state"] = "error"
                    _runtime_status["last_error"] = f"Render failed on {backend_label} for job {job_id}"
                    _runtime_status["current_job_id"] = None
                    return {"status": "faulted", "job_id": job_id}

                webp_bytes = _convert_png_to_webp(png_bytes, quality=92)
                if r2_url and r2_url.startswith("http"):
                    uploaded = await _upload_webp_to_r2(session, r2_url, webp_bytes)
                    if not uploaded:
                        logger.warning(f"[HordeDreamer] R2 upload failed for {job_id}, falling back to inline base64 submit")
                        generation_field = base64.b64encode(webp_bytes).decode("ascii")
                    else:
                        generation_field = "R2"
                else:
                    generation_field = base64.b64encode(webp_bytes).decode("ascii")

                submit_body = {
                    "id": job_id,
                    "generation": generation_field,
                    "state": "ok",
                    "seed": seed_int,
                    "censored": False,
                }

                reward = 0.0
                async with session.post(f"{HORDE_API_BASE}/generate/submit", json=submit_body, headers=headers) as sub_resp:
                    sub_txt = await sub_resp.text()
                    if sub_resp.status in (200, 201, 202):
                        try:
                            sub_json = json.loads(sub_txt)
                            reward = float(sub_json.get("reward") or 0.0)
                        except Exception:
                            reward = 0.0
                    else:
                        logger.error(f"[HordeDreamer] Submit failed ({sub_resp.status}) for {job_id}: {sub_txt[:200]}")
                        _runtime_status["last_error"] = f"Submit HTTP {sub_resp.status}: {sub_txt[:150]}"

                mps = round((width * height * steps) / 1_000_000.0, 2)
                upd = load_dreamer_data()
                upd["total_jobs_completed"] = int(upd.get("total_jobs_completed", 0)) + 1
                if is_krea_job:
                    upd["krea_jobs_completed"] = int(upd.get("krea_jobs_completed", 0)) + 1
                else:
                    upd["sdxl_jobs_completed"] = int(upd.get("sdxl_jobs_completed", 0)) + 1
                upd["total_kudos_earned"] = round(float(upd.get("total_kudos_earned", 0.0)) + reward, 2)
                upd["total_megapixelsteps"] = round(float(upd.get("total_megapixelsteps", 0.0)) + mps, 2)

                job_record = {
                    "id": job_id,
                    "ts": int(time.time()),
                    "model": job_model,
                    "backend": backend_label,
                    "resolution": f"{width}x{height}",
                    "steps": steps,
                    "duration_sec": render_sec,
                    "kudos": round(reward, 2),
                    "status": "ok",
                    "prompt_preview": pos_prompt[:100],
                }
                upd.setdefault("recent_jobs", []).insert(0, job_record)
                save_dreamer_data(upd)

                _runtime_status["last_submit_ts"] = time.time()
                _runtime_status["last_error"] = ""
                _runtime_status["current_job_id"] = None
                _runtime_status["current_model"] = None
                _runtime_status["current_backend"] = None
                _runtime_status["state"] = "idle_polling"

                logger.info(
                    f"[HordeDreamer] Submitted job {job_id} ({job_model} via {backend_label}) in {render_sec}s -> "
                    f"+{reward:.1f} Kudos (Krea today: {upd['krea_daily_count']}/{upd['krea_daily_limit']})"
                )
                return {"status": "ok", "job": job_record}

        except asyncio.CancelledError:
            raise
        except Exception as e:
            _runtime_status["state"] = "error"
            _runtime_status["last_error"] = str(e)
            _runtime_status["current_job_id"] = None
            logger.error(f"[HordeDreamer] Cycle exception: {e}", exc_info=True)
            return {"status": "error", "detail": str(e)}


async def _dreamer_worker_loop() -> None:
    logger.info("[HordeDreamer] Background AI Horde Dreamer worker loop started.")
    await asyncio.sleep(3.0)

    while True:
        try:
            data = load_dreamer_data()
            if not data.get("enabled", True):
                _runtime_status["state"] = "paused"
                _wake_event.clear()
                try:
                    await asyncio.wait_for(_wake_event.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    pass
                continue

            res = await run_single_dreamer_cycle(ignore_enabled=False)
            st = res.get("status")

            if st == "ok":
                sleep_sec = float(data.get("poll_interval_busy") or 2.0)
            elif st == "local_busy":
                sleep_sec = 6.0
            elif st == "unsafe_ip":
                sleep_sec = 300.0
            elif st == "error":
                sleep_sec = 20.0
            else:
                sleep_sec = float(data.get("poll_interval_idle") or 10.0)

            _wake_event.clear()
            try:
                await asyncio.wait_for(_wake_event.wait(), timeout=max(1.5, sleep_sec))
            except asyncio.TimeoutError:
                pass
        except asyncio.CancelledError:
            logger.info("[HordeDreamer] Background worker loop cancelled.")
            break
        except Exception as e:
            logger.error(f"[HordeDreamer] Unexpected loop error: {e}", exc_info=True)
            await asyncio.sleep(15.0)


async def start_horde_dreamer() -> None:
    global _worker_task
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_dreamer_worker_loop())


# ============================================================================
# AI Horde Image Generation Client (for OnMyChat users in Horde mode)
# ============================================================================

async def horde_generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 25,
    cfg_scale: float = 7.0,
    seed: Optional[int] = None,
    model: str = "CyberRealistic Pony",
    api_key: str = "",
    max_wait_seconds: int = 180,
) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
    """
    Submits an asynchronous image generation request to AI Horde (`/api/v2/generate/async`),
    polls `/generate/check/{id}` and `/generate/status/{id}`, downloads the generated WebP,
    and converts it to PNG bytes for seamless display in OnMyChat.
    Returns `(png_bytes, worker_name, actual_model)` or `(None, None, error_str)`.
    """
    eff_key = (api_key or "").strip() or get_effective_api_key() or horde_client.ANONYMOUS_API_KEY
    full_prompt = f"{prompt.strip()} {PROMPT_SEPARATOR} {negative_prompt.strip()}" if negative_prompt.strip() else prompt.strip()

    w_aligned = max(512, min(1024, (int(width or 1024) // 64) * 64))
    h_aligned = max(512, min(1024, (int(height or 1024) // 64) * 64))

    params: dict[str, Any] = {
        "sampler_name": "k_euler_a",
        "cfg_scale": float(cfg_scale or 7.0),
        "width": w_aligned,
        "height": h_aligned,
        "steps": min(30, max(15, int(steps or 25))),
        "karras": True,
        "n": 1,
    }
    if seed is not None and int(seed) > 0:
        params["seed"] = str(int(seed))

    req_body = {
        "prompt": full_prompt,
        "params": params,
        "nsfw": True,
        "censor_nsfw": False,
        "trusted_workers": False,
        "slow_workers": True,
        "models": [model] if model else ["CyberRealistic Pony", "AlbedoBase XL (SDXL)"],
        "r2": True,
    }

    headers = {
        "Content-Type": "application/json",
        "apikey": eff_key,
        "Client-Agent": CLIENT_AGENT,
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=max_wait_seconds)) as session:
            async with session.post(f"{HORDE_API_BASE}/generate/async", json=req_body, headers=headers) as resp:
                if resp.status not in (200, 202):
                    err = await resp.text()
                    return None, None, f"AI Horde image submit error ({resp.status}): {err[:140]}"
                res_json = await resp.json()
                task_id = res_json.get("id")

            if not task_id:
                return None, None, "AI Horde did not return an image task ID"

            elapsed = 0.0
            while elapsed < max_wait_seconds:
                await asyncio.sleep(2.5)
                elapsed += 2.5

                async with session.get(f"{HORDE_API_BASE}/generate/check/{task_id}", headers={"Client-Agent": CLIENT_AGENT}) as c_resp:
                    if c_resp.status != 200:
                        continue
                    chk = await c_resp.json()

                if chk.get("faulted"):
                    return None, None, "AI Horde worker faulted while generating image"
                if chk.get("is_possible") is False:
                    return None, None, "No AI Horde image workers available for the selected model"

                if chk.get("done"):
                    async with session.get(f"{HORDE_API_BASE}/generate/status/{task_id}", headers={"Client-Agent": CLIENT_AGENT}) as s_resp:
                        if s_resp.status != 200:
                            break
                        st = await s_resp.json()
                    gens = st.get("generations") or []
                    if not gens:
                        return None, None, "AI Horde returned empty generations list"
                    gen0 = gens[0]
                    img_ref = str(gen0.get("img") or "")
                    worker_name = str(gen0.get("worker_name") or "AI Horde")
                    used_model = str(gen0.get("model") or model)

                    raw_bytes: Optional[bytes] = None
                    if img_ref.startswith("http"):
                        async with session.get(img_ref) as dl_resp:
                            if dl_resp.status == 200:
                                raw_bytes = await dl_resp.read()
                    elif img_ref:
                        raw_bytes = base64.b64decode(img_ref)

                    if not raw_bytes:
                        return None, None, "Failed to download generated image from AI Horde"

                    # Convert WebP to PNG for OnMyChat storage & pipeline compatibility
                    with Image.open(io.BytesIO(raw_bytes)) as im:
                        out_png = io.BytesIO()
                        im.save(out_png, format="PNG")
                        return out_png.getvalue(), worker_name, used_model

            return None, None, f"Timed out waiting for AI Horde image ({max_wait_seconds}s)"
    except Exception as e:
        logger.error(f"[HordeDreamer] horde_generate_image error: {e}")
        return None, None, str(e)


def install_horde_client_hooks() -> None:
    """
    Attach `horde_generate_image` onto `horde_client` and normalize `ctx.settings['ai_mode']`
    to `ctx.settings['ai_provider']` in `core_service.perform_prompt_stream`.
    """
    setattr(horde_client, "horde_generate_image", horde_generate_image)
    setattr(horde_client, "DEFAULT_CLIENT_AGENT", CLIENT_AGENT)
    try:
        import core_service as _cs
        orig_stream = getattr(_cs, "perform_prompt_stream", None)
        if callable(orig_stream) and not getattr(orig_stream, "_horde_hooked", False):
            async def _hooked_perform_prompt_stream(ctx, *args, **kwargs):
                if ctx is not None and isinstance(getattr(ctx, "settings", None), dict):
                    if not ctx.settings.get("ai_provider") and ctx.settings.get("ai_mode"):
                        ctx.settings["ai_provider"] = ctx.settings["ai_mode"]
                async for item in orig_stream(ctx, *args, **kwargs):
                    yield item
            _hooked_perform_prompt_stream._horde_hooked = True
            _cs.perform_prompt_stream = _hooked_perform_prompt_stream
    except Exception as e:
        logger.warning(f"[HordeDreamer] Could not hook perform_prompt_stream: {e}")


# ============================================================================
# Owner Auth & HTTP Console / API Routes
# ============================================================================

def _is_loopback_request(request: Request) -> bool:
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    xri = (request.headers.get("x-real-ip") or "").strip()
    if xff or xri:
        first_ip = (xff.split(",")[0].strip() if xff else xri)
        return first_ip in ("127.0.0.1", "::1", "localhost")
    client_host = (request.client.host if request.client else "") or ""
    return client_host in ("127.0.0.1", "::1", "localhost")


def _check_owner(request: Request) -> bool:
    try:
        from extensions.autopilot import _check_owner_request
        if _check_owner_request(request):
            return True
    except Exception:
        pass
    if _is_loopback_request(request):
        return True
    role = (request.headers.get("x-omd-caller-role") or "").strip().lower()
    if role == "owner":
        return True
    try:
        import api as _api_mod
        if hasattr(_api_mod, "trusted_owner_access") and _api_mod.trusted_owner_access(request):
            return True
    except Exception:
        pass
    return False


def _require_owner(request: Request) -> None:
    if not _check_owner(request):
        raise HTTPException(status_code=404, detail="Not Found")


router = APIRouter()


async def _build_state_payload(refresh_user: bool = False) -> dict[str, Any]:
    data = load_dreamer_data()
    uinfo = await fetch_horde_user_info(data.get("api_key"), force=refresh_user)
    raw_key = get_effective_api_key(data)
    masked_key = (raw_key[:4] + "••••" + raw_key[-4:]) if len(raw_key) > 8 else ("••••" if raw_key else "")

    return {
        "config": {
            "enabled": bool(data.get("enabled", True)),
            "worker_name": data.get("worker_name", "OnMyChat-Dreamer"),
            "has_api_key": bool(raw_key),
            "masked_api_key": masked_key,
            "nsfw": bool(data.get("nsfw", True)),
            "max_pixels": int(data.get("max_pixels", 1024 * 1024)),
            "poll_interval_idle": float(data.get("poll_interval_idle", 10.0)),
            "poll_interval_busy": float(data.get("poll_interval_busy", 2.0)),
            "sdxl_enabled": bool(data.get("sdxl_enabled", True)),
            "sdxl_models": data.get("sdxl_models") or ["CyberRealistic Pony"],
            "sdxl_catalog": LOCAL_SDXL_MODEL_CATALOG,
            "krea_enabled": bool(data.get("krea_enabled", True)),
            "krea_daily_limit": int(data.get("krea_daily_limit", 10)),
            "krea_daily_date": data.get("krea_daily_date", _utc_today_str()),
            "krea_daily_count": int(data.get("krea_daily_count", 0)),
            "modal_available": is_modal_enabled(),
        },
        "stats": {
            "total_jobs_completed": int(data.get("total_jobs_completed", 0)),
            "sdxl_jobs_completed": int(data.get("sdxl_jobs_completed", 0)),
            "krea_jobs_completed": int(data.get("krea_jobs_completed", 0)),
            "total_jobs_faulted": int(data.get("total_jobs_faulted", 0)),
            "total_kudos_earned": round(float(data.get("total_kudos_earned", 0.0)), 2),
            "total_megapixelsteps": round(float(data.get("total_megapixelsteps", 0.0)), 2),
            "recent_jobs": (data.get("recent_jobs") or [])[:30],
        },
        "runtime": {
            "state": _runtime_status.get("state", "stopped"),
            "current_job_id": _runtime_status.get("current_job_id"),
            "current_model": _runtime_status.get("current_model"),
            "current_backend": _runtime_status.get("current_backend"),
            "last_pop_ts": _runtime_status.get("last_pop_ts", 0.0),
            "last_submit_ts": _runtime_status.get("last_submit_ts", 0.0),
            "last_error": _runtime_status.get("last_error", ""),
            "last_skipped": _runtime_status.get("last_skipped") or {},
            "active_models_offered": get_active_models_for_pop(data),
            "bridge_agent": BRIDGE_AGENT,
        },
        "account": {
            "username": (uinfo or {}).get("username", ""),
            "id": (uinfo or {}).get("id"),
            "kudos": (uinfo or {}).get("kudos", 0.0),
            "trusted": (uinfo or {}).get("trusted", False),
            "worker_count": (uinfo or {}).get("worker_count", 0),
            "worker_ids": (uinfo or {}).get("worker_ids") or [],
            "records": (uinfo or {}).get("records") or {},
        } if uinfo else None,
    }


@router.get("/horde", response_class=HTMLResponse)
async def horde_console_page(request: Request):
    _require_owner(request)
    return HTMLResponse(content=HORDE_CONSOLE_HTML)


@router.get("/horde/api/state")
@router.get("/assistant/horde_worker")
async def horde_get_state(request: Request, refresh: bool = False):
    _require_owner(request)
    return JSONResponse(await _build_state_payload(refresh_user=refresh))


@router.post("/horde/api/config")
@router.post("/assistant/horde_worker")
async def horde_update_config(request: Request):
    _require_owner(request)
    body = await request.json()
    data = load_dreamer_data()

    if "enabled" in body:
        data["enabled"] = bool(body["enabled"])
    if "worker_name" in body and str(body["worker_name"]).strip():
        data["worker_name"] = str(body["worker_name"]).strip()[:100]
    if "api_key" in body:
        new_key = str(body["api_key"]).strip()
        if new_key and "••••" not in new_key:
            data["api_key"] = new_key
            _runtime_status["user_info"] = None
    if "nsfw" in body:
        data["nsfw"] = bool(body["nsfw"])
    if "max_pixels" in body:
        try:
            data["max_pixels"] = max(262144, min(2097152, int(body["max_pixels"])))
        except Exception:
            pass
    if "sdxl_enabled" in body:
        data["sdxl_enabled"] = bool(body["sdxl_enabled"])
    if "sdxl_models" in body and isinstance(body["sdxl_models"], list):
        valid = [str(m).strip() for m in body["sdxl_models"] if str(m).strip() in LOCAL_SDXL_MODEL_CATALOG]
        if valid:
            data["sdxl_models"] = valid
    if "krea_enabled" in body:
        data["krea_enabled"] = bool(body["krea_enabled"])
    if "krea_daily_limit" in body:
        try:
            data["krea_daily_limit"] = max(-1, min(1000, int(body["krea_daily_limit"])))
        except Exception:
            pass

    save_dreamer_data(data)
    _wake_event.set()
    return JSONResponse(await _build_state_payload(refresh_user=True))


@router.post("/horde/api/toggle")
async def horde_toggle_worker(request: Request):
    _require_owner(request)
    data = load_dreamer_data()
    data["enabled"] = not bool(data.get("enabled", True))
    save_dreamer_data(data)
    _wake_event.set()
    return JSONResponse(await _build_state_payload(refresh_user=False))


@router.post("/horde/api/step")
async def horde_trigger_single_step(request: Request):
    _require_owner(request)
    res = await run_single_dreamer_cycle(ignore_enabled=True)
    state = await _build_state_payload(refresh_user=True)
    return JSONResponse({"step_result": res, "state": state})


@router.post("/horde/api/reset_krea_quota")
async def horde_reset_krea_quota(request: Request):
    _require_owner(request)
    data = load_dreamer_data()
    data["krea_daily_date"] = _utc_today_str()
    data["krea_daily_count"] = 0
    save_dreamer_data(data)
    _wake_event.set()
    return JSONResponse(await _build_state_payload(refresh_user=False))


@router.post("/horde/api/test_local_sdxl")
async def horde_test_local_sdxl(request: Request):
    _require_owner(request)
    t0 = time.time()
    wf = _build_sdxl_horde_workflow(
        positive_prompt="cinematic portrait of a cyberpunk hacker girl in neon rain, masterpiece",
        negative_prompt="blurry, low quality",
        ckpt_name="cyberrealisticPony_v180Coreshift.safetensors",
        width=896,
        height=1152,
        seed=random.randint(1, 2147483646),
    )
    png_bytes = await _run_local_comfy_silent(wf)
    dt = round(time.time() - t0, 2)
    if not png_bytes:
        return JSONResponse({"ok": False, "error": "Local ComfyUI render failed"}, status_code=500)
    webp_bytes = _convert_png_to_webp(png_bytes)
    return JSONResponse({
        "ok": True,
        "duration_sec": dt,
        "png_bytes": len(png_bytes),
        "webp_bytes": len(webp_bytes),
    })


def mount_horde_dreamer(app) -> None:
    app.include_router(router)


HORDE_CONSOLE_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OnMyChat · AI Horde Dreamer Worker</title>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --purple: #a371f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px; margin-bottom: 20px;
    padding-bottom: 16px; border-bottom: 1px solid var(--border);
  }
  h1 { margin: 0; font-size: 22px; display: flex; align-items: center; gap: 10px; }
  .badge {
    font-size: 12px; padding: 4px 10px; border-radius: 999px; font-weight: 600;
    background: #21262d; border: 1px solid var(--border);
  }
  .badge.ok { color: var(--green); border-color: rgba(63,185,80,0.4); }
  .badge.warn { color: var(--yellow); border-color: rgba(210,153,34,0.4); }
  .badge.err { color: var(--red); border-color: rgba(248,81,73,0.4); }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 14px; margin-bottom: 20px;
  }
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .stat-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat-val { font-size: 26px; font-weight: 700; margin-top: 6px; }
  .stat-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  @media (max-width: 800px) { .row { grid-template-columns: 1fr; } }
  h2 { margin: 0 0 12px 0; font-size: 16px; }
  label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 4px; }
  input[type="text"], input[type="password"], input[type="number"] {
    width: 100%; padding: 8px 10px; background: #0d1117; border: 1px solid var(--border);
    color: var(--text); border-radius: 6px; font-size: 14px; margin-bottom: 12px;
  }
  .chk-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 14px; }
  .btns { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
  button {
    cursor: pointer; border: 1px solid var(--border); background: #21262d;
    color: var(--text); padding: 8px 14px; border-radius: 6px; font-size: 13px; font-weight: 600;
  }
  button.primary { background: #238636; border-color: rgba(240,246,252,0.1); color: #fff; }
  button.accent { background: #1f6feb; border-color: rgba(240,246,252,0.1); color: #fff; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; }
  .pill { padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }
  .pill.sdxl { background: rgba(63,185,80,0.15); color: var(--green); }
  .pill.krea { background: rgba(163,113,247,0.15); color: var(--purple); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>
        <span>⚡ AI Horde Dreamer Worker</span>
        <span id="stateBadge" class="badge">Loading...</span>
      </h1>
      <div class="stat-sub mono" id="bridgeAgent"></div>
    </div>
    <div class="btns">
      <button id="toggleBtn" onclick="toggleWorker()">Pause / Resume</button>
      <button class="accent" id="stepBtn" onclick="runSingleStep()">▶ Run 1 Job Now</button>
      <button onclick="loadState(true)">↻ Refresh</button>
    </div>
  </header>

  <div class="grid">
    <div class="card">
      <div class="stat-label">AI Horde Account</div>
      <div class="stat-val" id="accName">—</div>
      <div class="stat-sub" id="accKudos">Balance: 0 Kudos</div>
    </div>
    <div class="card">
      <div class="stat-label">Local SDXL (Unlimited)</div>
      <div class="stat-val" id="sdxlCount">0</div>
      <div class="stat-sub">Jobs rendered on local RTX GPU ($0 cost)</div>
    </div>
    <div class="card">
      <div class="stat-label">Modal Krea 2 Daily Quota</div>
      <div class="stat-val" id="kreaQuota">0 / 10</div>
      <div class="stat-sub" id="kreaSub">Auto-stops Krea2 at limit</div>
    </div>
    <div class="card">
      <div class="stat-label">Session Kudos Earned</div>
      <div class="stat-val" id="earnedKudos">0</div>
      <div class="stat-sub" id="mpsTotal">0 Megapixelsteps</div>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>Worker Settings & Modal Krea 2 Cap</h2>
      <label>Worker Name</label>
      <input type="text" id="workerName" placeholder="OnMyChat-Dreamer">
      <label>AI Horde Worker API Key</label>
      <input type="password" id="apiKey" placeholder="Leave unchanged or paste new API key">
      <label>Krea 2 (Modal) Daily Generation Limit (0 = off, 10 = max 10/day)</label>
      <input type="number" id="kreaLimit" min="0" max="500" value="10">
      <div class="chk-row">
        <input type="checkbox" id="kreaEnabled">
        <span>Offer <b>Krea2-Turbo_fp8</b> while under daily limit</span>
      </div>
      <div class="chk-row">
        <input type="checkbox" id="nsfwEnabled">
        <span>Allow NSFW generations</span>
      </div>
      <div class="btns">
        <button class="primary" onclick="saveConfig()">Save Settings</button>
        <button onclick="resetKreaQuota()">Reset Today's Krea Counter</button>
      </div>
    </div>

    <div class="card">
      <h2>Unlimited Local SDXL Models (`http://localhost:8188`)</h2>
      <div class="chk-row">
        <input type="checkbox" id="sdxlEnabled">
        <span><b>Enable Unlimited Local SDXL Worker (24/7 when idle)</b></span>
      </div>
      <div id="sdxlModelsList" style="margin-top:10px;"></div>
      <div class="stat-sub" style="margin-top:10px;">
        Active models currently advertised to <code>/v2/generate/pop</code>:
        <div class="mono" id="activeModels" style="margin-top:4px;color:var(--accent);"></div>
      </div>
      <div class="stat-sub" id="lastErrorBox" style="margin-top:8px;color:var(--red);"></div>
    </div>
  </div>

  <div class="card">
    <h2>Recent Fulfilled Jobs</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Job ID</th>
          <th>Model</th>
          <th>Backend</th>
          <th>Size</th>
          <th>Time</th>
          <th>Kudos</th>
          <th>Prompt</th>
        </tr>
      </thead>
      <tbody id="jobsTable">
        <tr><td colspan="8" style="color:var(--muted);">No jobs recorded yet.</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
let catalogData = {};
let formDirty = false;

async function loadState(refresh=false) {
  const r = await fetch('/horde/api/state' + (refresh ? '?refresh=true' : ''));
  if (!r.ok) return;
  const d = await r.json();
  renderState(d);
}

function renderState(d) {
  const cfg = d.config || {};
  const st = d.stats || {};
  const rt = d.runtime || {};
  const acc = d.account || {};

  const badge = document.getElementById('stateBadge');
  badge.textContent = (cfg.enabled ? rt.state : 'paused').toUpperCase();
  badge.className = 'badge ' + (cfg.enabled && rt.state !== 'error' ? 'ok' : (rt.state === 'error' ? 'err' : 'warn'));

  document.getElementById('toggleBtn').textContent = cfg.enabled ? '⏸ Pause Worker' : '▶ Resume Worker';
  document.getElementById('bridgeAgent').textContent = rt.bridge_agent || '';

  document.getElementById('accName').textContent = acc.username || 'Not fetched';
  document.getElementById('accKudos').textContent = 'Account Balance: ' + Math.round(acc.kudos || 0).toLocaleString() + ' Kudos';
  document.getElementById('sdxlCount').textContent = st.sdxl_jobs_completed || 0;
  document.getElementById('kreaQuota').textContent = (cfg.krea_daily_count || 0) + ' / ' + (cfg.krea_daily_limit ?? 10);
  document.getElementById('kreaSub').textContent = 'Total Krea jobs: ' + (st.krea_jobs_completed || 0) + ' (' + (cfg.krea_daily_date || '') + ' UTC)';
  document.getElementById('earnedKudos').textContent = '+' + (st.total_kudos_earned || 0).toLocaleString();
  document.getElementById('mpsTotal').textContent = (st.total_megapixelsteps || 0).toLocaleString() + ' MPS (' + (st.total_jobs_completed || 0) + ' total jobs)';

  if (!formDirty) {
    document.getElementById('workerName').value = cfg.worker_name || '';
    document.getElementById('apiKey').placeholder = cfg.masked_api_key || 'Enter API Key';
    document.getElementById('kreaLimit').value = cfg.krea_daily_limit ?? 10;
    document.getElementById('kreaEnabled').checked = !!cfg.krea_enabled;
    document.getElementById('nsfwEnabled').checked = !!cfg.nsfw;
    document.getElementById('sdxlEnabled').checked = !!cfg.sdxl_enabled;

    catalogData = cfg.sdxl_catalog || {};
    const selected = new Set(cfg.sdxl_models || []);
    const box = document.getElementById('sdxlModelsList');
    box.innerHTML = '';
    Object.keys(catalogData).forEach(mName => {
      const info = catalogData[mName];
      const div = document.createElement('div');
      div.className = 'chk-row';
      div.innerHTML = `<label style="display:flex;align-items:center;gap:8px;color:var(--text);margin:0;cursor:pointer;">
        <input type="checkbox" class="sdxl-model-chk" value="${mName}" ${selected.has(mName) ? 'checked' : ''}>
        <span><b>${mName}</b> <span style="color:var(--muted);font-size:12px;">(${info.ckpt})</span></span>
      </label>`;
      box.appendChild(div);
    });
  }

  document.getElementById('activeModels').textContent = (rt.active_models_offered || []).join(', ') || 'None';
  document.getElementById('lastErrorBox').textContent = rt.last_error ? ('Last Error: ' + rt.last_error) : '';

  const tbody = document.getElementById('jobsTable');
  const jobs = st.recent_jobs || [];
  if (!jobs.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted);">No jobs recorded yet.</td></tr>';
  } else {
    tbody.innerHTML = jobs.map(j => {
      const dt = new Date((j.ts || 0) * 1000).toLocaleTimeString();
      const pillCls = (j.backend || '').includes('Krea') ? 'krea' : 'sdxl';
      return `<tr>
        <td class="mono">${dt}</td>
        <td class="mono">${(j.id || '').slice(0, 8)}</td>
        <td><b>${j.model || ''}</b></td>
        <td><span class="pill ${pillCls}">${j.backend || ''}</span></td>
        <td class="mono">${j.resolution || ''}</td>
        <td class="mono">${j.duration_sec || 0}s</td>
        <td class="mono" style="color:var(--green);">+${j.kudos || 0}</td>
        <td style="color:var(--muted);max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${(j.prompt_preview || '').replace(/</g,'&lt;')}</td>
      </tr>`;
    }).join('');
  }
}

document.addEventListener('input', () => { formDirty = true; });

async function saveConfig() {
  const models = Array.from(document.querySelectorAll('.sdxl-model-chk:checked')).map(el => el.value);
  const body = {
    worker_name: document.getElementById('workerName').value,
    krea_daily_limit: parseInt(document.getElementById('kreaLimit').value || '10', 10),
    krea_enabled: document.getElementById('kreaEnabled').checked,
    nsfw: document.getElementById('nsfwEnabled').checked,
    sdxl_enabled: document.getElementById('sdxlEnabled').checked,
    sdxl_models: models
  };
  const apiKeyVal = document.getElementById('apiKey').value.trim();
  if (apiKeyVal) body.api_key = apiKeyVal;

  const r = await fetch('/horde/api/config', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (r.ok) {
    document.getElementById('apiKey').value = '';
    formDirty = false;
    renderState(await r.json());
  }
}

async function toggleWorker() {
  const r = await fetch('/horde/api/toggle', {method: 'POST'});
  if (r.ok) renderState(await r.json());
}

async function resetKreaQuota() {
  const r = await fetch('/horde/api/reset_krea_quota', {method: 'POST'});
  if (r.ok) renderState(await r.json());
}

async function runSingleStep() {
  const btn = document.getElementById('stepBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Popping & Rendering...';
  try {
    const r = await fetch('/horde/api/step', {method: 'POST'});
    if (r.ok) {
      const res = await r.json();
      renderState(res.state);
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run 1 Job Now';
  }
}

loadState(true);
setInterval(() => loadState(false), 6000);
</script>
</body>
</html>
"""
