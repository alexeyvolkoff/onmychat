"""
agentic_tunnel.py — Agentic mode: remote inference over P2P raw-port tunnels.

In agentic mode the local OnMyChat node keeps the CPU-friendly work in-process
(SentenceTransformers embeddings, ChromaDB, TTS) and tunnels the heavy
inference calls — Ollama (intent / RAG / generation) and ComfyUI — to a remote
inference node. The calls themselves stay byte-identical to the local case:
all we do is expose local TCP listeners that proxy raw bytes through WebRTC
DataChannels to the remote node's local ports.

Topology:

    core_service (aiohttp in this process)
        -> TCP 127.0.0.1:<bind_port>          (InferenceForwarder listener)
            -> RawPortTunnel                  (WebRTC offerer, aiortc; offer
                                               carries port=<remote_port>,
                                               service=""; the remote OMD node
                                               answers and TCP-connects to
                                               127.0.0.1:<remote_port>)
                -> remoteOTS node local service   (Ollama :11434 / ComfyUI :8188)

This is the exact mirror of the browser's raw-mode transport
(`svar/src/lib/webrtc-drive.js::createWebRTCTransport`) reusing the signaling
machinery of `p2p_client.py`. No C++ changes are required: the node's
`handleTunnelOpenRequest` already opens raw TCP to its local port for
non-terminal tunnels and injects gateway-validated `X-OMD-Caller-*` headers,
so identity is trusted end-to-end (Ollama/ComfyUI ignore them).

Control frames
--------------
The node's P2PTunnel filters control frames itself; the offerer mirrors the
browser behaviour:
  - receives `__ready__` (text) on channel open (best-effort wait, 2s);
  - sends `__ping__` (text) every keepalive interval, counts misses;
  - answers node `__ping__` with `__pong__`.

Startup
-------
    await agentic_tunnel.start_from_settings(SETTINGS)   # api.py on_startup

- reads `AI_PROVIDER`; unless "agentic" this module stays dormant;
- binds local listeners and rewrites `core_service.OLLAMA_URL` /
  `core_service.COMFY_API_URL` to the listener addresses (every call site in
  core_service/api.py resolves these module globals at runtime, so the
  heavy-lifting call paths need no changes);
- tunnels open lazily per accepted TCP connection.

Per-request routing
-------------------
The SVAR frontend sends the chosen inference node in headers:
`X-OMD-Ai-Mode: agentic` / `X-OMD-Ai-Node: <linkID>`
(call `note_request()` from request handlers). The manager hot-switches its
active node; already-established connections keep serving the node they were
opened against. Precedence: request header > config.ini `INFERENCE_NODE`.

Signaling token: `AI_TOKEN` (this node owner's omd_key) when set, otherwise
the last omd_key seen on an incoming request (`note_token`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional

import aiohttp

import p2p_client
from p2p_client import P2PError, P2PSettings, SignalingClient, _random_id

log = logging.getLogger("agentic")

# send-side chunk cap: negotiated aiortc maxMessageSize defaults to 65536,
# stay safely below (the C++ node also slices at 64 KB).
_SEND_CHUNK = 60 * 1024
_SEND_BUDGET = 1 * 1024 * 1024      # max DataChannel buffered bytes before throttle
_POLL_INTERVAL = 0.3
_READY_TIMEOUT = 2.0                # node "__ready__" wait (old binaries may omit it)

FRAME_READY = "__ready__"
FRAME_PING = "__ping__"
FRAME_PONG = "__pong__"


class RawPortTunnel:
    """Offerer-side raw-port tunnel: one binary DataChannel ↔ one TCP stream.

    Lifecycle: `open()` → handshake (offer via gateway, poll answer, ICE,
    DataChannel open, optional `__ready__`) → external pump drives
    `send()` inbound and `on_data` outbound. Bytes are binary frames in both
    directions; text frames are control-only.
    """

    def __init__(self, settings: P2PSettings, signaling: SignalingClient,
                 target_link_id: str, remote_port: int):
        assert remote_port > 0, "raw-port tunnel requires port>0 in the offer"
        self._settings = settings
        self._signaling = signaling
        self.target_link_id = target_link_id
        self.remote_port = remote_port
        self.tunnel_id = _random_id("p2p-tunnel-")
        self.state = "new"              # new | connecting | open | closed | failed
        self.on_data: Optional[Callable[[bytes], Awaitable[None]]] = None
        self.last_error: str = ""

        self._pc = None
        self._dc = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._tasks: List[asyncio.Task] = []
        self._pending_candidates: List[str] = []
        self._last_alive = time.monotonic()
        self._closed_event = asyncio.Event()
        self._ready_event: asyncio.Event = asyncio.Event()
        self._got_remote_desc = False

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    async def open(self) -> "RawPortTunnel":
        if self.state in ("open", "connecting"):
            return self
        try:
            self.state = "connecting"
            await self._create_channel()
            await asyncio.wait_for(self._handshake(), timeout=self._settings.handshake_timeout)
            self.state = "open"
            self._last_alive = time.monotonic()
            self._tasks.append(asyncio.create_task(self._keepalive()))
            self._tasks.append(asyncio.create_task(self._dispatch()))
            self._tasks.append(asyncio.create_task(self._watch_pc()))
            return self
        except Exception as exc:
            self.last_error = str(exc)
            self.state = "failed"
            await self.close("open failed: " + self.last_error)
            raise

    async def _create_channel(self) -> None:
        try:
            from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
        except ImportError as exc:
            raise P2PError("aiortc is not installed — run: venv/bin/pip install aiortc") from exc

        config = RTCConfiguration(iceServers=[
            RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential"))
            for s in self._settings.ice_servers
        ])
        self._pc = RTCPeerConnection(configuration=config)
        self._pc.on("connectionstatechange", self._on_pc_state)
        self._dc = self._pc.createDataChannel("p2p-tunnel")
        self._dc.on("message", self._on_message)
        self._dc.on("close", lambda: asyncio.create_task(self._on_closed("data channel closed")))

    async def _handshake(self) -> None:
        from aiortc import RTCSessionDescription

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        sdp = self._pc.localDescription.sdp

        await self._signaling.send(
            self.target_link_id, self.tunnel_id, "offer",
            payload=await _b64encode(sdp),
            port=self.remote_port,
            service="",             # NOT "p2p-fs": node treats it as raw TCP proxy
        )
        log.info("agentic: raw-port offer %s ->%s:%d (tunnel=%s)",
                 self.target_link_id, self.target_link_id, self.remote_port, self.tunnel_id)

        deadline = time.monotonic() + self._settings.handshake_timeout
        got_answer = False
        while time.monotonic() < deadline:
            for sig in await self._signaling.poll(self.tunnel_id):
                kind = sig.get("kind")
                payload = sig.get("payload") or ""
                if kind == "answer" and not got_answer:
                    remote_sdp = await _b64decode(payload)
                    await self._pc.setRemoteDescription(
                        RTCSessionDescription(sdp=remote_sdp, type="answer"))
                    got_answer = True
                elif kind == "candidate" and payload:
                    self._on_remote_candidate(await _b64decode(payload))
                elif kind in ("tunnel-closed", "closed"):
                    raise P2PError(f"gateway reported tunnel {kind}")
            if got_answer:
                break
            await asyncio.sleep(self._settings.poll_interval)
        if not got_answer:
            raise P2PTimeoutError(f"no answer from {self.target_link_id}")

        # Trickle candidates may arrive after the answer; drain briefly.
        deadline2 = time.monotonic() + 1.0
        while time.monotonic() < deadline2:
            got = False
            for sig in await self._signaling.poll(self.tunnel_id):
                kind = sig.get("kind")
                payload = sig.get("payload") or ""
                if kind == "candidate" and payload:
                    self._on_remote_candidate(await _b64decode(payload))
                    got = True
                elif kind in ("tunnel-closed", "closed"):
                    raise P2PError(f"gateway reported tunnel {kind}")
            if not got:
                break

        dc_deadline = time.monotonic() + self._settings.handshake_timeout
        while self._dc.readyState != "open":
            if time.monotonic() > dc_deadline:
                raise P2PTimeoutError("data channel did not open in time")
            if self._pc.connectionState in ("failed", "closed"):
                raise P2PError(f"peer connection {self._pc.connectionState}")
            await asyncio.sleep(0.05)

        await self._flush_candidates()

        # Wait briefly for the node's __ready__ (race: first bytes may race the
        # answerer's onMessage wiring — the browser does the same for FS tunnels).
        if self._dc.readyState == "open":
            if not self._ready_event.is_set():
                try:
                    await asyncio.wait_for(self._ready_event.wait(), timeout=_READY_TIMEOUT)
                except asyncio.TimeoutError:
                    pass  # old binaries may not send it; channel is fine

    # -- inbound frames ---------------------------------------------------- #

    def _on_message(self, data) -> None:
        if isinstance(data, (bytes, bytearray)):
            self._last_alive = time.monotonic()
            try:
                self._queue.put_nowait(bytes(data))
            except asyncio.QueueFull:
                log.warning("agentic: tunnel %s inbound queue overflow, dropping", self.tunnel_id)
            return
        s = (data or "").strip()
        if s == FRAME_READY:
            if getattr(self, "_ready_event", None):
                self._ready_event.set()
            self._last_alive = time.monotonic()
        elif s == FRAME_PING:
            self._send_raw(FRAME_PONG)
            self._last_alive = time.monotonic()
        elif s == FRAME_PONG:
            self._last_alive = time.monotonic()

    def _send_raw(self, text: str) -> None:
        try:
            if self._dc is not None and self._dc.readyState == "open":
                self._dc.send(text)
        except Exception:
            pass

    # -- outbound bytes ---------------------------------------------------- #

    async def send(self, data: bytes) -> None:
        """Write bytes into the tunnel (chunked, with backpressure)."""
        if self.state != "open":
            raise P2PError(f"tunnel not open (state={self.state})")
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + _SEND_CHUNK]
            dc = self._dc
            if dc is None or dc.readyState != "open":
                raise P2PError("data channel closed during send")
            while getattr(dc, "bufferedAmount", 0) > _SEND_BUDGET:
                if self.state != "open":
                    raise P2PError("tunnel closed during send")
                await asyncio.sleep(0.02)
            dc.send(chunk)
            offset += len(chunk)

    # -- candidates -------------------------------------------------------- #

    def _on_remote_candidate(self, raw: str) -> None:
        s = (raw or "").strip()
        if s.startswith("a="):
            s = s[2:].strip()
        if not s:
            return
        if getattr(self, "_got_remote_desc", False) is False:
            self._pending_candidates.append(s)
            return
        if self._pc is not None:
            asyncio.create_task(self._add_candidate(s))

    async def _flush_candidates(self) -> None:
        self._got_remote_desc = True
        pending, self._pending_candidates = self._pending_candidates, []
        for c in pending:
            await self._add_candidate(c)

    async def _add_candidate(self, s: str) -> None:
        try:
            await self._pc.addIceCandidate({"candidate": s, "sdpMLineIndex": 0, "sdpMid": "0"})
        except Exception as exc:
            # Non-fatal: the answer SDP usually carries its own candidates.
            log.debug("agentic: candidate skipped for %s: %s", self.tunnel_id, exc)

    # -- lifecycle / keepalive --------------------------------------------- #

    def _on_pc_state(self) -> None:
        state = getattr(self._pc, "connectionState", "")
        if state in ("failed", "closed"):
            asyncio.create_task(self._on_closed(f"peer connection {state}"))

    async def _watch_pc(self) -> None:
        while self.state == "open":
            await asyncio.sleep(0.5)
            if getattr(self, "_dc", None) is None or self._dc.readyState != "open":
                await self._on_closed("data channel lost")
                return

    async def _dispatch(self) -> None:
        """Hand inbound tunnel bytes to `on_data`, preserving order."""
        while self.state == "open":
            try:
                data = await self._queue.get()
            except asyncio.CancelledError:
                return
            if self.on_data is None:
                continue
            try:
                await self.on_data(data)
            except Exception as exc:
                log.warning("agentic: on_data failed for %s: %s", self.tunnel_id, exc)
                await self._on_closed("on_data error")
                return

    async def _keepalive(self) -> None:
        interval = self._settings.keepalive_interval
        limit = interval * self._settings.keepalive_misses
        while self.state == "open":
            await asyncio.sleep(interval)
            if self.state != "open":
                return
            self._send_raw(FRAME_PING)
            if time.monotonic() - self._last_alive > limit:
                await self._on_closed("keepalive timeout")
                return

    async def _on_closed(self, reason: str) -> None:
        if self.state == "closed":
            return
        self.state = "closed"
        self.last_error = reason
        log.info("agentic: tunnel %s closed: %s", self.tunnel_id, reason)
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()
        for closer in (getattr(self._dc, "close", None), getattr(self._pc, "close", None)):
            if closer is None:
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
        self._dc = None
        self._pc = None
        self._closed_event.set()

    async def close(self, reason: str = "closed by local") -> None:
        await self._on_closed(reason)


class P2PTimeoutError(P2PError):
    pass


async def _b64encode(text: str) -> str:
    import base64
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


async def _b64decode(text: str) -> str:
    import base64
    return base64.b64decode((text or "").encode("ascii"))


# --------------------------------------------------------------------------- #
# InferenceForwarder: local TCP listener → raw-port tunnel per connection
# --------------------------------------------------------------------------- #

class InferenceForwarder:
    """Local TCP listener piping every connection through one raw tunnel."""

    def __init__(self, manager: "AgenticManager", name: str,
                 bind_host: str, bind_port: int, remote_port: int):
        self.manager = manager
        self.name = name
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.remote_port = remote_port
        self.listening = False
        self.connections = 0
        self.tunnels_open = 0
        self.last_error = ""
        self._server: Optional[asyncio.AbstractServer] = None
        self._conns: Dict[asyncio.Task, object] = {}

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_conn, self.bind_host, self.bind_port)
        self.listening = True
        log.info("agentic: %s forwarder listening on %s:%s (remote :%s)",
                 self.name, self.bind_host, self.bind_port, self.remote_port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._server = None
        self.listening = False

    @property
    def bind_url(self) -> str:
        return f"http://{self.bind_host}:{self.bind_port}"

    async def _handle_conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        tunnel = self.manager.make_tunnel(self.remote_port)
        tunnel.on_data = self._make_on_data(writer)
        peer = writer.get_extra_info("peername")
        try:
            await tunnel.open()
            self.tunnels_open += 1
            log.info("agentic: %s conn from %s tunneled to %s:%s (tunnel=%s)",
                     self.name, peer, tunnel.target_link_id, self.remote_port, tunnel.tunnel_id)
            t1 = asyncio.create_task(self._pump_tcp_to_dc(tunnel, reader))
            t2 = asyncio.create_task(self._pump_dc_to_tcp(tunnel, writer))
            done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, (asyncio.CancelledError, ConnectionError)):
                    log.warning("agentic: %s pump error: %s", self.name, exc)
        except Exception as exc:
            self.last_error = f"tunnel open failed: {exc}"
            log.warning("agentic: %s conn from %s failed: %s", self.name, peer, exc)
        finally:
            self.tunnels_open = max(0, self.tunnels_open - 1)
            try:
                tunnel.on_data = None
                await tunnel.close("forwarder cleanup")
            except Exception:
                pass
            if not writer.is_closing():
                writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _make_on_data(self, writer: asyncio.StreamWriter) -> Callable[[bytes], Awaitable[None]]:
        async def writechunk(data: bytes) -> None:
            if writer.is_closing():
                raise ConnectionError("client closed")
            writer.write(data)
            await writer.drain()
        return writechunk

    async def _pump_tcp_to_dc(self, tunnel: "RawPortTunnel", reader: asyncio.StreamReader) -> None:
        while tunnel.is_open:
            data = await reader.read(_SEND_CHUNK)
            if not data:
                break
            await tunnel.send(data)
        await tunnel.close("client connection ended")

    async def _pump_dc_to_tcp(self, tunnel: "RawPortTunnel", writer: asyncio.StreamWriter) -> None:
        while tunnel.is_open:
            try:
                data = await asyncio.wait_for(tunnel._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return
            if writer.is_closing():
                raise ConnectionError("client closed")
            writer.write(data)
            await writer.drain()


# --------------------------------------------------------------------------- #
# AgenticManager
# --------------------------------------------------------------------------- #

class AgenticManager:
    """Owns the listeners + tunnels, the active node hot-switch and URLs."""

    def __init__(self, settings):
        self._section = settings
        self.mode = str(_sget(settings, "AI_PROVIDER", "local")).strip().lower()
        self.active_node = str(_sget(settings, "INFERENCE_NODE", "") or "").strip()
        self.remote_ollama_port = int(_sget(settings, "INFERENCE_OLLAMA_PORT", 11434) or 11434)
        self.remote_comfy_port = int(_sget(settings, "INFERENCE_COMFY_PORT", 8188) or 8188)
        self.bind_ollama_port = int(_sget(settings, "INFERENCE_BIND_OLLAMA", self.remote_ollama_port) or self.remote_ollama_port)
        self.bind_comfy_port = int(_sget(settings, "INFERENCE_BIND_COMFY", self.remote_comfy_port) or self.remote_comfy_port)
        self.bind_host = str(_sget(settings, "INFERENCE_BIND_HOST", "127.0.0.1") or "127.0.0.1")
        self.ai_token = str(_sget(settings, "AI_TOKEN", "") or "").strip()
        self.last_seen_token = ""
        self.last_error = ""

        self.settings = P2PSettings.from_settings(settings)
        self.settings.token = self._signaling_token()
        self.signaling = SignalingClient(self.settings)
        self._forwarders: List[InferenceForwarder] = []
        self._started = False

    # -- token routing ------------------------------------------------------ #

    def _signaling_token(self) -> str:
        return self.ai_token or self.last_seen_token

    def note_token(self, omd_key: str) -> None:
        if omd_key and len(omd_key) >= 32:
            self.last_seen_token = omd_key
            self.settings.token = self._signaling_token()

    # -- lifecycle ---------------------------------------------------------- #

    async def start(self) -> None:
        if self._started:
            return
        if not self.active_node:
            raise RuntimeError("agentic mode enabled but INFERENCE_NODE is not set")
        forwarders = [
            InferenceForwarder(self, "ollama", self.bind_host, self.bind_ollama_port, self.remote_ollama_port),
            InferenceForwarder(self, "comfy", self.bind_host, self.bind_comfy_port, self.remote_comfy_port),
        ]
        for f in forwarders:
            await f.start()
        self._forwarders = forwarders
        self._started = True
        log.info("agentic: started (node=%s ollama %s->:%s comfy %s->:%s)",
                 self.active_node, f"{self.bind_host}:{self.bind_ollama_port}",
                 self.remote_ollama_port, f"{self.bind_host}:{self.bind_comfy_port}",
                 self.remote_comfy_port)

    async def stop(self) -> None:
        for f in self._forwarders:
            await f.stop()
        self._forwarders.clear()
        await self.signaling.close()
        self._started = False

    # -- request hooks ------------------------------------------------------ #

    def make_tunnel(self, remote_port: int) -> RawPortTunnel:
        """Tunnel factory (test hook: override to inject a fake tunnel)."""
        self.settings.token = self._signaling_token()
        return RawPortTunnel(self.settings, self.signaling, self.active_node, remote_port)

    def set_active_node(self, node_id: str) -> bool:
        node_id = (node_id or "").strip()
        if node_id and node_id != self.active_node:
            log.info("agentic: switching inference node %s -> %s", self.active_node, node_id)
            self.active_node = node_id
            return True
        return False

    def urls(self) -> Dict[str, str]:
        if not self._started:
            return {}
        return {f.name: f.bind_url for f in self._forwarders}

    def patch_core_urls(self) -> None:
        import core_service
        urls = self.urls()
        if "ollama" in urls:
            core_service.OLLAMA_URL = urls["ollama"]
            log.info("agentic: core_service.OLLAMA_URL -> %s", urls["ollama"])
        if "comfy" in urls:
            core_service.COMFY_API_URL = urls["comfy"]
            log.info("agentic: core_service.COMFY_API_URL -> %s", urls["comfy"])

    def status(self) -> Dict:
        return {
            "mode": self.mode,
            "agentic": self._started,
            "active_node": self.active_node,
            "has_token": bool(self._signaling_token()),
            "p2p_sender_id": self.settings.sender_id,
            "last_error": self.last_error,
            "services": [
                {
                    "name": f.name,
                    "bind": f.bind_url,
                    "remote_port": f.remote_port,
                    "listening": f.listening,
                    "connections": f.connections,
                    "tunnels_open": f.tunnels_open,
                    "last_error": f.last_error,
                }
                for f in self._forwarders
            ],
        }


def _sget(section, key, default=None):
    value = section.get(key, default)
    return default if value is None or value in ("", "") else value


# --------------------------------------------------------------------------- #
# Module-level API
# --------------------------------------------------------------------------- #

_manager: Optional[AgenticManager] = None


def get_manager() -> Optional[AgenticManager]:
    return _manager


async def start_from_settings(settings) -> Optional[AgenticManager]:
    """Bind listeners + patch core URLs when AI_PROVIDER=agentic; else no-op."""
    global _manager
    mode = str(_sget(settings, "AI_PROVIDER", "local") or "local").strip().lower()
    if mode != "agentic":
        log.info("agentic: disabled (AI_PROVIDER=%s)", mode)
        return None
    manager = AgenticManager(settings)
    await manager.start()
    manager.patch_core_urls()
    _manager = manager
    return manager


def note_request(request=None, omd_key: str = "", settings_payload: dict | None = None) -> None:
    """Feed per-request info to the manager (headers > settings payload)."""
    if _manager is None:
        return
    try:
        _manager.note_token(omd_key or "")
        if request is not None:
            mode = (request.headers.get("X-OMD-Ai-Mode") or "").strip().lower()
            node = (request.headers.get("X-OMD-Ai-Node") or "").strip()
        else:
            mode, node = "", ""
        if not mode and settings_payload:
            mode = str(settings_payload.get("ai_mode") or "").strip().lower()
        if not node and settings_payload:
            node = str(settings_payload.get("ai_node_infer") or "").strip()
        if mode == "agentic" and node:
            _manager.set_active_node(node)
    except Exception as exc:
        log.debug("agentic: note_request failed: %s", exc)
