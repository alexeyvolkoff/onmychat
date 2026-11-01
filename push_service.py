"""
Custom Web Push Notification Service (RFC 8291 VAPID) for OnMyChat & Svar PWA.
Provides peer-to-peer / self-hosted rich push notifications with preview images
(e.g. /p2p/{defaultStorage}/generated/...) and interactive action buttons.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid
from pywebpush import WebPushException, webpush

logger = logging.getLogger("push_service")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAPID_KEYS_PATH = os.path.join(BASE_DIR, "vapid_keys.json")
SUBSCRIPTIONS_PATH = os.path.join(BASE_DIR, "push_subscriptions.json")

_lock = threading.Lock()
_vapid_cache: Optional[dict[str, str]] = None


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def ensure_vapid_keys() -> dict[str, str]:
    """Loads or generates ECDSA P-256 VAPID keypair on first use."""
    global _vapid_cache
    with _lock:
        if _vapid_cache:
            return _vapid_cache

        if os.path.isfile(VAPID_KEYS_PATH):
            try:
                with open(VAPID_KEYS_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if loaded.get("public_key") and loaded.get("private_key"):
                    _vapid_cache = loaded
                    return _vapid_cache
            except Exception as e:
                logger.warning(f"[WebPush] Failed reading {VAPID_KEYS_PATH}, regenerating: {e}")

        vapid = Vapid()
        vapid.generate_keys()
        raw_pub = vapid.public_key.public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        raw_priv = vapid.private_key.private_numbers().private_value.to_bytes(32, "big")
        keys_obj = {
            "public_key": _b64url_encode(raw_pub),
            "private_key": _b64url_encode(raw_priv),
            "subject": "mailto:admin@onmydisk.net",
            "created_at": int(time.time()),
        }
        tmp_path = VAPID_KEYS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(keys_obj, f, indent=2)
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        os.replace(tmp_path, VAPID_KEYS_PATH)
        _vapid_cache = keys_obj
        logger.info("[WebPush] Generated new VAPID keypair.")
        return _vapid_cache


def get_public_vapid_key() -> str:
    return ensure_vapid_keys()["public_key"]


def _load_subscriptions_unlocked() -> dict[str, dict[str, Any]]:
    if not os.path.isfile(SUBSCRIPTIONS_PATH):
        return {}
    try:
        with open(SUBSCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning(f"[WebPush] Could not load subscriptions: {e}")
    return {}


def _save_subscriptions_unlocked(subs: dict[str, dict[str, Any]]) -> None:
    tmp_path = SUBSCRIPTIONS_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(subs, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    os.replace(tmp_path, SUBSCRIPTIONS_PATH)


def _hash_key(omd_key: str) -> str:
    clean = (omd_key or "").strip()
    if not clean:
        return ""
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()[:24]


def normalize_storage_prefix(default_storage: str) -> str:
    """Normalizes storage path like '/Laptop/OnMyChat/' -> 'Laptop/OnMyChat' for /p2p/<storage>/... URLs."""
    s = (default_storage or "").strip()
    if s.startswith("/p2p/"):
        s = s[5:]
    return s.strip("/")


def build_p2p_url(default_storage: str, rel_path: str) -> str:
    """Builds a clean /p2p/{defaultStorage}/{rel_path} URL."""
    rel = (rel_path or "").strip().lstrip("/")
    prefix = normalize_storage_prefix(default_storage)
    if prefix:
        return f"/p2p/{prefix}/{rel}" if rel else f"/p2p/{prefix}"
    return f"/p2p/{rel}" if rel else "/"


def save_subscription(
    subscription: dict[str, Any],
    user_id: str = "",
    omd_key: str = "",
    default_storage: str = "",
    device_label: str = "",
    user_agent: str = "",
    is_owner: bool = False,
) -> dict[str, Any]:
    """Saves or updates a browser PushSubscription."""
    if not isinstance(subscription, dict):
        raise ValueError("Invalid subscription object")
    endpoint = str(subscription.get("endpoint") or "").strip()
    keys = subscription.get("keys")
    if not endpoint or not isinstance(keys, dict) or not keys.get("p256dh") or not keys.get("auth"):
        raise ValueError("Subscription must contain endpoint and keys (p256dh, auth)")

    now = int(time.time())
    with _lock:
        subs = _load_subscriptions_unlocked()
        existing = subs.get(endpoint) or {}
        entry = {
            "endpoint": endpoint,
            "keys": {
                "p256dh": str(keys["p256dh"]),
                "auth": str(keys["auth"]),
            },
            "expirationTime": subscription.get("expirationTime"),
            "user_id": user_id or existing.get("user_id") or "",
            "omd_key_hash": _hash_key(omd_key) or existing.get("omd_key_hash") or "",
            "default_storage": default_storage.strip() if default_storage and default_storage.strip() else (existing.get("default_storage") or ""),
            "device_label": (device_label or existing.get("device_label") or "")[:120],
            "user_agent": (user_agent or existing.get("user_agent") or "")[:240],
            "is_owner": bool(is_owner or existing.get("is_owner")),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
        }
        subs[endpoint] = entry
        _save_subscriptions_unlocked(subs)
    logger.info(f"[WebPush] Registered subscription (owner={entry['is_owner']}, storage='{entry['default_storage']}', label='{entry['device_label']}')")
    return entry


def remove_subscription(endpoint: str) -> bool:
    ep = (endpoint or "").strip()
    if not ep:
        return False
    with _lock:
        subs = _load_subscriptions_unlocked()
        if ep in subs:
            subs.pop(ep, None)
            _save_subscriptions_unlocked(subs)
            logger.info("[WebPush] Unsubscribed endpoint.")
            return True
    return False


def list_subscriptions(owner_only: bool = False, user_id: str = "", omd_key: str = "") -> list[dict[str, Any]]:
    key_hash = _hash_key(omd_key)
    with _lock:
        subs = list(_load_subscriptions_unlocked().values())
    out = []
    for s in subs:
        if owner_only and not s.get("is_owner"):
            continue
        if user_id and s.get("user_id") != user_id:
            if not (key_hash and s.get("omd_key_hash") == key_hash):
                continue
        elif key_hash and not owner_only and s.get("omd_key_hash") != key_hash:
            continue
        out.append(s)
    return out


def get_owner_default_storage() -> str:
    """Returns the most recently updated default_storage registered by an owner subscription."""
    owners = list_subscriptions(owner_only=True)
    if not owners:
        all_subs = list_subscriptions(owner_only=False)
        owners = all_subs
    owners.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    for s in owners:
        ds = (s.get("default_storage") or "").strip()
        if ds:
            return ds
    return ""


def _resolve_placeholder_storage(val: str, storage: str) -> str:
    if not val or "{defaultStorage}" not in val:
        return val
    norm = normalize_storage_prefix(storage)
    if norm:
        replaced = val.replace("/{defaultStorage}/", f"/{norm}/").replace("{defaultStorage}", norm)
    else:
        replaced = val.replace("/{defaultStorage}/", "/").replace("{defaultStorage}", "")
    return replaced


def _send_single_webpush_sync(sub_entry: dict[str, Any], payload_str: str, vapid_keys: dict[str, str], ttl: int = 86400) -> tuple[bool, int, str]:
    endpoint = sub_entry["endpoint"]
    parsed = urlparse(endpoint)
    aud = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    claims = {
        "sub": vapid_keys.get("subject", "mailto:admin@onmydisk.net"),
    }
    if aud:
        claims["aud"] = aud

    sub_info = {
        "endpoint": endpoint,
        "keys": sub_entry["keys"],
    }
    try:
        resp = webpush(
            subscription_info=sub_info,
            data=payload_str,
            vapid_private_key=vapid_keys["private_key"],
            vapid_claims=claims,
            ttl=ttl,
            timeout=15,
            headers={"Urgency": "high"},
        )
        status = getattr(resp, "status_code", 201) or 201
        return True, int(status), ""
    except WebPushException as wpe:
        status = getattr(getattr(wpe, "response", None), "status_code", 0) or 0
        body = ""
        if getattr(wpe, "response", None) is not None:
            try:
                body = (wpe.response.text or "")[:240]
            except Exception:
                pass
        return False, int(status), body or str(wpe)[:240]
    except Exception as exc:
        return False, 0, str(exc)[:240]


async def send_custom_push(
    title: str,
    body: str,
    image: str = "",
    image_rel_path: str = "",
    icon: str = "/customtheme/onmydisk.svg",
    badge: str = "/customtheme/onmydisk.svg",
    url: str = "/",
    tag: str = "",
    actions: Optional[list[dict[str, str]]] = None,
    data: Optional[dict[str, Any]] = None,
    require_interaction: bool = True,
    silent_when_focused: bool = False,
    owner_only: bool = False,
    user_id: str = "",
    omd_key: str = "",
    target_endpoint: str = "",
    fallback_storage: str = "",
) -> dict[str, Any]:
    """
    Sends a custom Rich Web Push notification to matching subscribed devices.
    Supports {defaultStorage} resolution per device or via image_rel_path.
    """
    vapid_keys = ensure_vapid_keys()
    if target_endpoint:
        with _lock:
            all_map = _load_subscriptions_unlocked()
        sub = all_map.get(target_endpoint)
        targets = [sub] if sub else []
    else:
        targets = list_subscriptions(owner_only=owner_only, user_id=user_id, omd_key=omd_key)
        # Fallback: if owner_only=True and no subscription explicitly got flagged yet on localhost,
        # check if there are any subscriptions at all on this private node when NODE_OWNER isn't separate
        if owner_only and not targets:
            with _lock:
                all_subs = list(_load_subscriptions_unlocked().values())
            if len(all_subs) == 1:
                targets = all_subs

    if not targets:
        return {"sent": 0, "failed": 0, "total": 0, "detail": "No subscribed devices"}

    sent = 0
    failed = 0
    stale_endpoints: list[str] = []
    errors: list[str] = []

    for sub in targets:
        dev_storage = (sub.get("default_storage") or fallback_storage or "").strip()
        resolved_image = image
        if not resolved_image and image_rel_path:
            resolved_image = build_p2p_url(dev_storage, image_rel_path)
        elif resolved_image:
            resolved_image = _resolve_placeholder_storage(resolved_image, dev_storage)

        resolved_url = _resolve_placeholder_storage(url or "/", dev_storage)
        extra_data = dict(data or {})
        if dev_storage and not extra_data.get("default_storage"):
            extra_data["default_storage"] = dev_storage
        if resolved_image and not extra_data.get("image"):
            extra_data["image"] = resolved_image

        payload_obj: dict[str, Any] = {
            "type": "custom_push",
            "title": title or "On My Disk",
            "body": body or "",
            "icon": icon or "/customtheme/onmydisk.svg",
            "badge": badge or "/customtheme/onmydisk.svg",
            "url": resolved_url,
            "requireInteraction": bool(require_interaction),
            "silentWhenFocused": bool(silent_when_focused),
            "timestamp": int(time.time() * 1000),
            "data": extra_data,
        }
        if resolved_image:
            payload_obj["image"] = resolved_image
        if tag:
            payload_obj["tag"] = tag
        if actions:
            payload_obj["actions"] = actions

        payload_str = json.dumps(payload_obj, ensure_ascii=False)
        ok, status_code, err_msg = await asyncio.to_thread(
            _send_single_webpush_sync, sub, payload_str, vapid_keys
        )
        if ok:
            sent += 1
        else:
            failed += 1
            errors.append(f"HTTP {status_code}: {err_msg}")
            if status_code in (404, 410):
                stale_endpoints.append(sub["endpoint"])

    for ep in stale_endpoints:
        remove_subscription(ep)

    logger.info(f"[WebPush] Push dispatched '{title}': sent={sent}, failed={failed}, pruned={len(stale_endpoints)}")
    return {
        "sent": sent,
        "failed": failed,
        "pruned": len(stale_endpoints),
        "total": len(targets),
        "errors": errors[:3],
    }

