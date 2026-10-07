"""
device_push.py — push generated artifacts to the user's own device.

The public AI node generates chat images and avatar candidates, but ephemeral
storage (`store_ephemeral_image`, TTL 5 min) and node gateway storage are the
wrong home for them: the picture must live on the *user's* device, in the
folder declared by the profile (`defaultStorage`, e.g. `/Private/onmychat`),
under `generated/` — the legacy `<share>/onmychat/generated` layout the
filemanager used to create.

Transport is the existing p2p tunnel (`p2p_client.DeviceClient`): the browser
mints a one-shot consent grant for the target folder right before the request
(`issueConsent(folder, {writable: true, openWindow: false})` in svar) and sends
it as `device_push = {linkId, consentToken, folder}`; the node redeems the
grant, lands inside the folder jail and writes with MkDir + chunked Write.

Pushes run fire-and-forget: a failed tunnel must never fail the generation
response — the ephemeral copy + `/chat/image/` fallback still render, and the
client shows an "image expired" placeholder once it is gone.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import p2p_client

log = logging.getLogger("device_push")

# ICE + consent auth + directory creation + writes. Generation itself takes
# far longer, and this runs in the background, so be generous.
PUSH_TIMEOUT = 90.0

# (filename, bytes) — written into the granted folder as-is.
FileSpec = Tuple[str, bytes]


def _device_push_ctx(device_push: Any) -> Optional[Dict[str, Any]]:
    """Validate the client-supplied push descriptor; None when unusable."""
    if not isinstance(device_push, dict):
        log.warning("device_push: descriptor is not a dict: %r", type(device_push))
        return None
    link_id = str(device_push.get("linkId") or "").strip()
    token = str(device_push.get("consentToken") or "").strip()
    folder = _clean_folder(device_push.get("folder"))
    if not link_id or not token or not folder:
        log.warning("device_push: missing required fields: linkId=%r, token=%r, folder=%r", link_id, bool(token), folder)
        return None
    return {
        "linkId": link_id,
        "consentToken": token,
        "folder": folder,
        "didPub": str(device_push.get("didPub") or "").strip(),
    }


def _clean_folder(folder: Any) -> str:
    """Rooted, dot-segment-free folder path — no leading/trailing slashes."""
    parts: List[str] = []
    for seg in str(folder or "").split("/"):
        if not seg or seg == ".":
            continue
        if seg == "..":
            # Never let a client-supplied path climb out of its own grant.
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def _safe_name(name: str) -> str:
    return os.path.basename(str(name or "")).strip()


async def push_files_to_device(
    omd_key: str,
    device_push: Any,
    files: Sequence[FileSpec],
) -> bool:
    """Write every file into the granted folder on the user's device."""
    ctx_info = _device_push_ctx(device_push)
    if not ctx_info:
        log.warning("device_push: missing/invalid device_push descriptor: %r, skipping", device_push)
        return False
    if not omd_key:
        log.warning("device_push: no omd_key for signaling, skipping")
        return False

    payloads: List[FileSpec] = []
    for name, data in files:
        safe = _safe_name(name)
        if not safe or not isinstance(data, (bytes, bytearray)) or not data:
            log.warning("device_push: skipping bad file entry %r", name)
            continue
        payloads.append((safe, bytes(data)))
    if not payloads:
        return False

    folder = ctx_info["folder"]
    link_id = ctx_info["linkId"]

    settings = p2p_client.P2PSettings.from_settings(_settings())
    settings.token = omd_key
    client = p2p_client.P2PClient(settings)
    try:
        device = await client.device(
            link_id,
            ctx_info["consentToken"],
            did_pub=ctx_info["didPub"],
            scope_path=folder,
            writable=True,
        )
        await device.ensure_dir(folder)
        for name, data in payloads:
            await device.write_file(f"{folder}/{name}", data)
        log.info(
            "device_push: %d file(s) -> %s on %s",
            len(payloads), folder, link_id,
        )
        return True
    except Exception as exc:
        log.warning(
            "device_push: failed to push %d file(s) to %s on %s: %s",
            len(payloads), folder, link_id, exc,
        )
        return False
    finally:
        try:
            await client.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _settings():
    from config import SETTINGS
    return SETTINGS


def schedule_push(
    omd_key: str,
    device_push: Any,
    files: Sequence[FileSpec],
) -> bool:
    """Fire-and-forget wrapper: never blocks or fails the caller's response."""
    if not _device_push_ctx(device_push):
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        log.warning("device_push: no running event loop, push skipped")
        return False

    async def _run() -> None:
        try:
            await asyncio.wait_for(
                push_files_to_device(omd_key, device_push, files),
                timeout=PUSH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("device_push: push to %s timed out after %.0fs",
                        (device_push or {}).get("linkId"), PUSH_TIMEOUT)
        except Exception as exc:  # pragma: no cover - already guarded inside
            log.warning("device_push: background push failed: %s", exc)

    task = loop.create_task(_run())
    task.add_done_callback(lambda t: t.exception())
    return True


def schedule_generated_image_push(
    omd_key: str,
    device_push: Any,
    filename: str,
    image_data: bytes,
    title: str = "",
    description: str = "",
) -> bool:
    """Chat image → `<folder>/IMG_x.png` + companion Readme (legacy habit)."""
    files: List[FileSpec] = [(filename, image_data)]
    base = os.path.splitext(_safe_name(filename))[0]
    if base and title:
        readme = f"# {title}\n\n{description or ''}\n".encode("utf-8")
        files.append((f"{base}.Readme.md", readme))
    return schedule_push(omd_key, device_push, files)


def schedule_avatar_push(
    omd_key: str,
    device_push: Any,
    filename: str,
    image_data: bytes,
) -> bool:
    """Avatar candidate → `<folder>/generated/avatars/<file>`.

    The descriptor's folder is the *generated* root; avatars live one level
    deeper (same layout the filemanager's ensureOnMyChat created).
    """
    ctx_info = _device_push_ctx(device_push)
    if not ctx_info:
        return False
    avatar_push = dict(device_push)
    avatar_push["folder"] = ctx_info["folder"].rstrip("/") + "/avatars"
    return schedule_push(omd_key, avatar_push, [(filename, image_data)])
