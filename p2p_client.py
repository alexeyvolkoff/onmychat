"""
p2p_client.py — WebRTC P2P initiator (offerer) for OnMyDisk.

Plays the OFFERER role in Python: the exact mirror image of the browser client
`svar/src/lib/webrtc-drive.js`. The remote peer is a user client device
(PWA / native app) which implements the ANSWERER role in
`svar/src/lib/client-device-server.js`.

Why this exists
---------------
Knowledge import today is browser-driven: the browser reads document bytes and
uploads them to the node (`POST /rag/import/raw`). This module lets the node
pull documents itself, straight from the user's device, and push the resulting
knowledge cards back over the very same tunnel — no client upload, no file
bytes through the gateway.

Signaling
---------
1. `RTCPeerConnection` + `createDataChannel("p2p-fs")`. aiortc gathers ICE
   candidates before `setLocalDescription()` returns (non-trickle), so the
   offer SDP already carries every candidate.
2. Offer is POSTed to `{gateway}/api/signaling`:
   `{linkID: <client linkID>, ID: <tunnelID>, kind: "offer", payload: b64(sdp),
     port: 0, service: "p2p-fs", senderId: <ours>, token: <omd_key>}`
3. The gateway routes it to the client (it registered itself as a web device via
   `POST /api/device/register`) and the client answers the same way. The answer
   comes back through the gateway signal queue, drained with
   `{kind: "poll", ID: <tunnelID>}` (WebSocket push is browser-only, so this
   module polls).

Wire protocol
-------------
Text frames, JSON, matching the C++ node and the browser client:

    request:  {"id": 1, "cmd": "FileInfo", "path": "/Docs"}
    response: {"id": 1, "cmd": "FileInfo", "status": "ok", "result": {...}}

Several JSON objects may arrive inside one frame, so frames are split with a
brace-depth scanner (string/escape aware) — the same approach as
`webrtc-drive.js` (`_handleMessage`) and `onmydisklinkclient.cpp`
(`QJsonParseError::GarbageAtEnd`).

Control frames: `__ready__` (peer signalling it is listening), `__ping__` /
`__pong__` (keepalive).

Dependencies: aiortc, aiohttp. aiortc is imported lazily so the rest of the
service keeps working when it is not installed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import aiohttp

log = logging.getLogger("p2p")

# --- framing / protocol constants (keep in sync with webrtc-drive.js) --------

DATA_CHANNEL_LABEL = "p2p-fs"
FRAME_READY = "__ready__"
FRAME_PING = "__ping__"
FRAME_PONG = "__pong__"

DEFAULT_CHUNK_SIZE = 65536          # node uses 64 KiB; camera sync uses 16 KiB
DEFAULT_RPC_TIMEOUT = 30.0
DEFAULT_HANDSHAKE_TIMEOUT = 15.0    # same deadline as webrtc-drive.js
DEFAULT_KEEPALIVE_INTERVAL = 30.0
DEFAULT_KEEPALIVE_MISSES = 3
MAX_BUFFER = 16 * 1024 * 1024       # node uses a 16 MB cap


class P2PError(Exception):
    """Any failure of the P2P subsystem (signaling, ICE, RPC, auth)."""


class P2PTimeout(P2PError):
    pass


class P2PAuthError(P2PError):
    pass


def _random_id(prefix: str, length: int = 9) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choice(alphabet) for _ in range(length))


def default_ice_servers(host: str = "direct.onmydisk.net") -> List[Dict[str, Any]]:
    """ICE servers mirroring `turnServers()` in svar/src/lib/webrtc-drive.js."""
    return [
        {"urls": f"stun:{host}:3478"},
        {"urls": "stun:stun.cloudflare.com:3478"},
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": f"turn:{host}:3478?transport=udp", "username": "turnuser", "credential": "turnpass"},
        {"urls": f"turn:{host}:3478?transport=tcp", "username": "turnuser", "credential": "turnpass"},
    ]


@dataclass
class P2PSettings:
    """Runtime configuration, normally sourced from config.ini [settings]."""

    signaling_url: str = "https://direct.onmydisk.net/api/signaling"
    token: str = ""                 # omd_key of the user whose data we access
    sender_id: str = ""             # our linkID-like id for gateway routing
    ice_servers: List[Dict[str, Any]] = field(default_factory=default_ice_servers)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    rpc_timeout: float = DEFAULT_RPC_TIMEOUT
    handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL
    keepalive_misses: int = DEFAULT_KEEPALIVE_MISSES
    poll_interval: float = 0.3

    @classmethod
    def from_settings(cls, settings) -> "P2PSettings":
        """Build from onmychat's SETTINGS dict (configparser section)."""
        def get(key: str, default=None):
            try:
                return settings.get(key, default)
            except AttributeError:
                return default

        def getf(key: str, default: float) -> float:
            raw = get(key, None)
            if raw in (None, ""):
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def geti(key: str, default: int) -> int:
            raw = get(key, None)
            if raw in (None, ""):
                return default
            try:
                return int(raw)
            except (TypeError, ValueError):
                return default

        host = str(get("P2P_ICE_HOST", "") or "direct.onmydisk.net")
        return cls(
            signaling_url=str(get("P2P_SIGNALING_URL", "") or cls.signaling_url),
            # token намеренно не берётся из конфига: omd_key живёт в запросе
            # конкретного юзера, глобального токена ноды нет.
            token="",
            sender_id=str(get("P2P_SENDER_ID", "") or ""),
            ice_servers=default_ice_servers(host),
            chunk_size=geti("P2P_CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
            rpc_timeout=getf("P2P_RPC_TIMEOUT", DEFAULT_RPC_TIMEOUT),
            handshake_timeout=getf("P2P_HANDSHAKE_TIMEOUT", DEFAULT_HANDSHAKE_TIMEOUT),
            keepalive_interval=getf("P2P_KEEPALIVE_INTERVAL", DEFAULT_KEEPALIVE_INTERVAL),
            keepalive_misses=geti("P2P_KEEPALIVE_MISSES", DEFAULT_KEEPALIVE_MISSES),
            poll_interval=getf("P2P_POLL_INTERVAL", 0.3),
        )


# --------------------------------------------------------------------------- #
# Signaling over the public gateway HTTP API
# --------------------------------------------------------------------------- #

class SignalingClient:
    """Thin async wrapper around `POST /api/signaling`.

    The endpoint is whitelisted for guest sessions
    (Node/src/httpnode/sessionauth.cpp); real authorization happens in-band over
    the data channel (`ConsentAuth`), which is what this module relies on.
    """

    def __init__(self, settings: P2PSettings, session: Optional[aiohttp.ClientSession] = None):
        self._settings = settings
        self._session = session
        self._owns_session = session is None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.rpc_timeout)
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._settings.token:
            # Same header the browser sends (ensureChannelToken -> omd_key).
            headers["Authorization"] = f"token {self._settings.token}"
        return headers

    async def send(
        self,
        link_id: str,
        tunnel_id: str,
        kind: str,
        payload: str = "",
        port: int = 0,
        service: str = DATA_CHANNEL_LABEL,
    ) -> Dict[str, Any]:
        body = {
            "linkID": link_id,
            "ID": tunnel_id,
            "kind": kind,
            "payload": payload,
            "port": port,
            "service": service,
            "senderId": self._settings.sender_id,
            "token": self._settings.token,
        }
        session = await self._ensure_session()
        try:
            async with session.post(
                self._settings.signaling_url, json=body, headers=self._headers()
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise P2PError(
                        f"signaling {kind} failed: HTTP {resp.status} {_short(text)}"
                    )
                try:
                    return json.loads(text) if text else {}
                except ValueError:
                    return {"status": "ok", "raw": text}
        except asyncio.TimeoutError as exc:
            raise P2PTimeout(f"signaling {kind} timed out") from exc
        except aiohttp.ClientError as exc:
            raise P2PError(f"signaling {kind} failed: {exc}") from exc

    async def poll(self, tunnel_id: str) -> List[Dict[str, Any]]:
        """Drain queued signals for our tunnel (answer / candidate / closed)."""
        data = await self.send(self._settings.sender_id, tunnel_id, "poll")
        signals = data.get("signals")
        return list(signals) if isinstance(signals, list) else []


def _short(text: str, limit: int = 200) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


# --------------------------------------------------------------------------- #
# One tunnel to a client device
# --------------------------------------------------------------------------- #

class P2PConnection:
    """A single offerer-side tunnel: signaling + data channel + JSON-RPC."""

    def __init__(self, settings: P2PSettings, signaling: SignalingClient,
                 target_link_id: str):
        self._settings = settings
        self._signaling = signaling
        self.target_link_id = target_link_id
        self.tunnel_id = _random_id("p2p-tunnel-")
        self.state = "new"              # new | connecting | open | closed | failed
        self.device_name: str = ""
        self.auth_level: str = ""
        self.peer_ready = False
        self.last_error: str = ""
        self.created_at = time.time()
        self.last_activity = self.created_at

        self._pc = None
        self._dc = None
        self._buffer = ""
        self._next_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._keepalive_task: Optional[asyncio.Task] = None
        self._last_pong = time.monotonic()
        self._ready_waiter: Optional[asyncio.Future] = None
        self._close_reason = ""

    # -- lifecycle -------------------------------------------------------- #

    async def open(self) -> "P2PConnection":
        if self.state in ("open", "connecting"):
            return self
        try:
            self.state = "connecting"
            self._ready_waiter = asyncio.get_running_loop().create_future()
            await self._create_channel()
            await asyncio.wait_for(
                self._handshake_signaling(), timeout=self._settings.handshake_timeout
            )

            self.state = "open"
            self.last_activity = time.time()
            self._keepalive_task = asyncio.create_task(self._keepalive())

            # `__ready__` is a courtesy from newer peers (C++ node sends it);
            # the browser answerer historically did not. Auth below is the real
            # liveness proof, so never block on it.
            if self._ready_waiter is not None and not self._ready_waiter.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._ready_waiter), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            return self
        except Exception as exc:
            self.last_error = str(exc)
            self.state = "failed"
            await self._teardown()
            raise

    async def _create_channel(self) -> None:
        """Build the peer connection + data channel (lazy aiortc import)."""
        try:
            from aiortc import (  # type: ignore
                RTCConfiguration,
                RTCIceServer,
                RTCPeerConnection,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise P2PError(
                "aiortc is not installed — run: venv/bin/pip install aiortc"
            ) from exc

        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=server["urls"],
                    username=server.get("username"),
                    credential=server.get("credential"),
                )
                for server in self._settings.ice_servers
            ]
        )
        self._pc = RTCPeerConnection(configuration=config)
        self._pc.on("connectionstatechange", self._on_connection_state)

        self._dc = self._pc.createDataChannel(DATA_CHANNEL_LABEL)
        self._dc.on("message", self._on_message)
        self._dc.on("close", lambda: self._on_closed("peer closed the channel"))

    async def _teardown(self) -> None:
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        self._fail_pending(f"tunnel {self._close_reason or self.last_error or 'closed'}")
        for closer in (getattr(self._dc, "close", None), getattr(self._pc, "close", None)):
            if closer is None:
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # pragma: no cover - best effort teardown
                pass
        self._dc = None
        self._pc = None

    # -- signaling -------------------------------------------------------- #

    async def _handshake_signaling(self) -> None:
        from aiortc import RTCSessionDescription  # type: ignore

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        sdp = self._pc.localDescription.sdp

        await self._signaling.send(
            self.target_link_id,
            self.tunnel_id,
            "offer",
            payload=base64.b64encode(sdp.encode("utf-8")).decode("ascii"),
        )
        log.info("P2P: offer sent to %s tunnel=%s", self.target_link_id, self.tunnel_id)

        deadline = time.monotonic() + self._settings.handshake_timeout
        got_answer = False
        while time.monotonic() < deadline:
            for sig in await self._signaling.poll(self.tunnel_id):
                kind = sig.get("kind")
                payload = sig.get("payload") or ""
                if kind in ("answer", "offer"):
                    remote_sdp = base64.b64decode(payload).decode("utf-8", "replace")
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=remote_sdp, type="answer")
                    )
                    got_answer = True
                elif kind == "candidate" and payload:
                    self._add_candidate(base64.b64decode(payload).decode("utf-8", "replace"))
                elif kind in ("tunnel-closed", "closed"):
                    raise P2PError(f"gateway reported tunnel {kind}")
            if got_answer:
                break
            await asyncio.sleep(self._settings.poll_interval)

        if not got_answer:
            raise P2PTimeout(
                f"no answer from {self.target_link_id} within "
                f"{self._settings.handshake_timeout:.0f}s"
            )

        # Wait for the data channel itself (ICE + DTLS + SCTP). Own deadline: the
        # answer may have arrived late and must not eat the channel budget.
        dc_deadline = time.monotonic() + self._settings.handshake_timeout
        while self._dc_ready_state() != "open":
            if time.monotonic() > dc_deadline:
                raise P2PTimeout("data channel did not open in time")
            if self._pc.connectionState in ("failed", "closed"):
                raise P2PError(f"peer connection {self._pc.connectionState}")
            await asyncio.sleep(0.05)

    def _dc_ready_state(self) -> str:
        return getattr(self._dc, "readyState", "closed") if self._dc else "closed"

    def _add_candidate(self, candidate: str) -> None:
        # aiortc gathers non-trickle, so remote candidates are normally already
        # inside the answer SDP. Kept for peers that trickle anyway.
        log.debug("P2P: remote candidate (embedded SDP expected): %s", candidate[:80])

    async def close(self, reason: str = "closed by local") -> None:
        self._close_reason = reason
        self.state = "closed"
        await self._teardown()

    # -- JSON-RPC --------------------------------------------------------- #

    async def rpc(self, cmd: str, timeout: Optional[float] = None, **params: Any) -> Dict[str, Any]:
        """Send one request and return the full response envelope."""
        if self.state != "open":
            raise P2PError(f"tunnel not open (state={self.state})")
        timeout = timeout or self._settings.rpc_timeout
        request_id = self._next_id
        self._next_id += 1

        envelope = {"id": request_id, "cmd": cmd}
        envelope.update(params)
        payload = json.dumps(envelope, separators=(",", ":"))

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        try:
            self._dc.send(payload)
        except Exception as exc:
            self._pending.pop(request_id, None)
            raise P2PError(f"send {cmd} failed: {exc}") from exc

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise P2PTimeout(f"RPC timeout for {cmd}") from exc
        finally:
            self._pending.pop(request_id, None)

        self.last_activity = time.time()
        if response.get("status") == "error":
            raise P2PError(f"{cmd}: {response.get('error') or 'unknown error'}")
        return response

    async def call(self, cmd: str, timeout: Optional[float] = None, **params: Any) -> Dict[str, Any]:
        """`rpc` but returning only `result` (empty dict when absent)."""
        response = await self.rpc(cmd, timeout=timeout, **params)
        result = response.get("result")
        return result if isinstance(result, dict) else {}

    # -- incoming frames -------------------------------------------------- #

    def _on_message(self, data: Any) -> None:
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8", "replace")
        if not isinstance(data, str):
            return
        self.last_activity = time.time()
        self._buffer += data

        if len(self._buffer) > MAX_BUFFER:
            log.warning("P2P: buffer overflow on %s, resetting", self.target_link_id)
            self._buffer = ""

        while True:
            frame = _extract_json(self._buffer)
            if frame is None:
                break
            payload, consumed = frame
            self._buffer = self._buffer[consumed:]
            self._handle_payload(payload)

    def _handle_payload(self, payload: str) -> None:
        stripped = payload.strip()
        if stripped in (FRAME_READY, FRAME_PING, FRAME_PONG):
            if stripped == FRAME_READY:
                self.peer_ready = True
                if self._ready_waiter and not self._ready_waiter.done():
                    self._ready_waiter.set_result(True)
            elif stripped == FRAME_PING:
                self._send_raw(FRAME_PONG)
            else:
                self._last_pong = time.monotonic()
            return

        try:
            message = json.loads(stripped)
        except ValueError:
            log.debug("P2P: non-JSON frame from %s: %s", self.target_link_id, _short(stripped, 80))
            return
        if not isinstance(message, dict):
            return

        message_id = message.get("id")
        if isinstance(message_id, int) and message_id in self._pending:
            future = self._pending.pop(message_id, None)
            if future is not None and not future.done():
                future.set_result(message)
            return
        log.debug("P2P: unsolicited frame %s from %s", message.get("cmd"), self.target_link_id)

    def _send_raw(self, text: str) -> None:
        try:
            if self._dc is not None and self._dc.readyState == "open":
                self._dc.send(text)
        except Exception:  # pragma: no cover - best effort keepalive
            pass

    # -- keepalive / failure --------------------------------------------- #

    async def _keepalive(self) -> None:
        interval = self._settings.keepalive_interval
        limit = interval * self._settings.keepalive_misses
        try:
            while self.state == "open":
                await asyncio.sleep(interval)
                if self.state != "open":
                    return
                self._send_raw(FRAME_PING)
                if time.monotonic() - self._last_pong > limit:
                    self.last_error = "keepalive timeout"
                    await self._on_closed("keepalive timeout")
                    return
        except asyncio.CancelledError:
            return

    def _on_connection_state(self) -> None:
        state = getattr(self._pc, "connectionState", "")
        if state in ("failed", "closed", "disconnected"):
            asyncio.create_task(self._on_closed(f"peer connection {state}"))

    async def _on_closed(self, reason: str) -> None:
        if self.state == "closed":
            return
        self.state = "closed"
        self._close_reason = reason
        self.last_error = reason
        self._fail_pending(f"tunnel {reason}")
        log.info("P2P: tunnel to %s closed: %s", self.target_link_id, reason)

    def _fail_pending(self, reason: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(P2PError(reason))
        self._pending.clear()


def _extract_json(buffer: str):
    """Return `(json_text, consumed_chars)` for the first complete object."""
    start = buffer.find("{")
    if start == -1:
        if buffer.strip():
            # No object start yet; drop control frames that may precede it.
            for frame in (FRAME_READY, FRAME_PING, FRAME_PONG):
                if buffer.lstrip().startswith(frame):
                    return frame, len(buffer) - len(buffer.lstrip()) + len(frame)
            return None
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(buffer)):
        char = buffer[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return buffer[start:index + 1], index + 1
    return None


# --------------------------------------------------------------------------- #
# Device-level operations (what the reindex job actually needs)
# --------------------------------------------------------------------------- #

class DeviceClient:
    """High level API over a client device: auth, browse, read, write cards."""

    def __init__(self, settings: P2PSettings, connection: P2PConnection):
        self._settings = settings
        self.conn = connection
        self.authenticated = False

    @property
    def link_id(self) -> str:
        return self.conn.target_link_id

    async def authenticate(self, consent_token: str, did_pub: str = "",
                           scope_path: str = "", writable: bool = False) -> Dict[str, Any]:
        """First RPC on every tunnel: prove the user granted access.

        The browser generates the consent token when the user picks a document
        or folder; this module only carries it back to the device that issued
        it. `AuthInfo` is sent first so the device can report its policy.
        """
        info = await self.conn.call("AuthInfo", didPub=did_pub)
        self.device_name = info.get("name") or info.get("user") or self.link_id

        result = await self.conn.call(
            "ConsentAuth",
            token=consent_token,
            didPub=did_pub,
            path=scope_path,
            writable=bool(writable),
        )
        self.conn.auth_level = result.get("authLevel", "")
        if self.conn.auth_level in ("", "denied", "unauthenticated"):
            raise P2PAuthError(
                f"device {self.link_id} refused consent "
                f"(authLevel={self.conn.auth_level or 'none'})"
            )
        self.authenticated = True
        log.info(
            "P2P: authenticated to %s as %s (%s)",
            self.link_id, self.device_name, self.conn.auth_level,
        )
        return result

    async def list_dir(self, path: str) -> Dict[str, Any]:
        result = await self.conn.call("FileInfo", path=path or "/")
        return result if isinstance(result, dict) else {}

    async def walk(self, root: str, max_entries: int = 5000,
                   should_visit: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
                   ) -> List[Dict[str, Any]]:
        """Depth-first listing of `root`, returning `{path, name, size, ...}`."""
        root = root or "/"
        found: List[Dict[str, Any]] = []
        queue: List[str] = [root]

        while queue and len(found) < max_entries:
            current = queue.pop(0)
            listing = await self.list_dir(current)
            for entry in listing.get("files") or []:
                name = entry.get("name") or ""
                if not name:
                    continue
                child = f"{current.rstrip('/')}/{name}"
                item = {
                    "path": child,
                    "name": name,
                    "isDir": bool(entry.get("isDir")),
                    "size": int(entry.get("size") or 0),
                    "lastModified": entry.get("lastModified") or "",
                    "mimeType": entry.get("mimeType") or "",
                }
                if item["isDir"]:
                    queue.append(child)
                else:
                    found.append(item)
                    if should_visit and not should_visit(child, item):
                        found.pop()
        return found

    async def read_file(self, path: str, max_bytes: Optional[int] = None) -> bytes:
        """Read a whole file through chunked `Open` + `Read` RPCs."""
        opened = await self.conn.call("Open", path=path)
        handle_id = opened.get("handleId")
        size = int(opened.get("size") or 0)
        limit = size if max_bytes is None else min(size, max_bytes)

        data = bytearray()
        offset = 0
        while offset < limit:
            params: Dict[str, Any] = {"offset": offset,
                                      "size": min(self._settings.chunk_size, limit - offset)}
            if handle_id:
                params["handleId"] = handle_id
            response = await self.conn.rpc("Read", path=path, **params)
            chunk_b64 = response.get("data") or ""
            if not chunk_b64:
                break
            chunk = base64.b64decode(chunk_b64)
            if not chunk:
                break
            data.extend(chunk)
            offset += len(chunk)
        return bytes(data)

    async def put_knowledge(self, card: Dict[str, Any]) -> Dict[str, Any]:
        """Hand one knowledge card to the device — it encrypts and stores it."""
        return await self.conn.call("KnowledgePut", card=card, timeout=self._settings.rpc_timeout)


# --------------------------------------------------------------------------- #
# Connection registry
# --------------------------------------------------------------------------- #

class P2PClient:
    """Keeps one live tunnel per client device so jobs reuse the ICE session."""

    def __init__(self, settings: Optional[P2PSettings] = None):
        self.settings = settings or P2PSettings()
        if not self.settings.sender_id:
            self.settings.sender_id = _load_or_create_sender_id()
        self._signaling = SignalingClient(self.settings)
        self._connections: Dict[str, P2PConnection] = {}
        self._lock = asyncio.Lock()

    async def device(self, link_id: str, consent_token: str, did_pub: str = "",
                     scope_path: str = "", writable: bool = False,
                     reuse: bool = True) -> DeviceClient:
        """Get an authenticated tunnel to `link_id`, opening one if needed."""
        if not link_id:
            raise P2PError("target link id is required")

        async with self._lock:
            connection = self._connections.get(link_id) if reuse else None
            if connection is not None and connection.state != "open":
                self._connections.pop(link_id, None)
                connection = None
            if connection is None:
                connection = P2PConnection(self.settings, self._signaling, link_id)
                await connection.open()
                self._connections[link_id] = connection

        client = DeviceClient(self.settings, connection)
        await client.authenticate(consent_token, did_pub=did_pub,
                                  scope_path=scope_path, writable=writable)
        return client

    async def drop(self, link_id: str) -> None:
        async with self._lock:
            connection = self._connections.pop(link_id, None)
        if connection is not None:
            await connection.close("dropped")

    async def close(self) -> None:
        async with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for connection in connections:
            await connection.close("client shutdown")
        await self._signaling.close()


_SENDER_ID_FILE = os.path.join("user_data", "p2p_sender_id")


def _load_or_create_sender_id() -> str:
    """Stable pseudo linkID so the gateway can route answers back to us."""
    try:
        if os.path.exists(_SENDER_ID_FILE):
            with open(_SENDER_ID_FILE, "r", encoding="utf-8") as handle:
                stored = handle.read().strip()
            if stored:
                return stored
        sender_id = _random_id("py", 10)
        os.makedirs(os.path.dirname(_SENDER_ID_FILE), exist_ok=True)
        with open(_SENDER_ID_FILE, "w", encoding="utf-8") as handle:
            handle.write(sender_id)
        return sender_id
    except OSError:
        return _random_id("py", 10)
