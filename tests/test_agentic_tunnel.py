"""
Standalone tests for agentic_tunnel (run: venv/bin/python tests/test_agentic_tunnel.py).

No aiortc/gateway involved: the raw-port tunnel is replaced by a loopback
"fake tunnel" that speaks the same interface through a plain TCP connection
to a mock "remote inference service" (echo server). This exercises the whole
forwarder path: TCP listener accept -> tunnel open -> bidirectional pump ->
cleanup, plus RawPortTunnel control-frame filtering and send chunking.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentic_tunnel
from agentic_tunnel import RawPortTunnel, InferenceForwarder


class FakeTunnel:
    """Same interface as RawPortTunnel, but the 'DataChannel' is a plain TCP
    connection to the mock remote service."""

    def __init__(self, manager, remote_port):
        self.manager = manager
        self.remote_port = remote_port
        self.target_link_id = manager.active_node
        self.tunnel_id = "fake-tunnel"
        self.state = "new"
        self.on_data = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._reader = None
        self._writer = None
        self._pump_task = None

    @property
    def is_open(self):
        return self.state == "open"

    async def open(self):
        self._reader, self._writer = await asyncio.open_connection("127.0.0.1", self.remote_port)
        self.state = "open"
        self._pump_task = asyncio.create_task(self._pump_remote_to_queue())
        return self

    async def _pump_remote_to_queue(self):
        while self.state == "open":
            data = await self._reader.read(65536)
            if not data:
                break
            self._queue.put_nowait(data)
        await self.close("remote connection ended")

    async def send(self, data):
        self._writer.write(data)
        await self._writer.drain()

    async def close(self, reason="closed by local"):
        if self.state == "closed":
            return
        self.state = "closed"
        if self._pump_task:
            self._pump_task.cancel()
        if self._writer:
            self._writer.close()


class MockDC:
    def __init__(self):
        self.sent_frames = []
        self.readyState = "open"
        self.bufferedAmount = 0

    def send(self, data):
        self.sent_frames.append(data)


def make_manager(active_node="nodeA"):
    class M:
        pass
    m = M()
    m.active_node = active_node
    m.fake_remote_port_holder = None

    def make_tunnel(remote_port):
        return FakeTunnel(m, m.fake_remote_port_holder)
    m.make_tunnel = make_tunnel
    return m


async def test_forwarder_roundtrip():
    # Mock "remote node service" (Ollama/Comfy stand-in): echoes requests with prefix
    async def echo_handler(reader, writer):
        data = await reader.read(65536)
        writer.write(b"REMOTE:" + data)
        await writer.drain()
        writer.close()

    remote_server = await asyncio.start_server(echo_handler, "127.0.0.1", 0)
    remote_port = remote_server.sockets[0].getsockname()[1]

    manager = make_manager()
    manager.fake_remote_port_holder = remote_port
    manager.settings = None
    manager.signaling = None

    fwd_listener = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)  # placeholder
    fwd_listener.close()

    fwd = InferenceForwarder.__new__(InferenceForwarder)
    fwd.manager = manager
    fwd.name = "test"
    fwd.bind_host = "127.0.0.1"
    fwd.bind_port = 0
    fwd.remote_port = 99999  # invalid: must be overridden by fake tunnel anyway
    fwd.listening = False
    fwd.connections = 0
    fwd.tunnels_open = 0
    fwd.last_error = ""
    fwd._server = None
    fwd._conns = {}

    await fwd.start()
    listen_port = fwd._server.sockets[0].getsockname()[1]

    reader, writer = await asyncio.open_connection("127.0.0.1", listen_port)
    writer.write(b"hello-tunnel")
    await writer.drain()
    resp = await asyncio.wait_for(reader.read(1024), timeout=5)
    assert resp == b"REMOTE:hello-tunnel", resp

    fwd._server.close()
    remote_server.close()
    writer.close()
    print("PASS test_forwarder_roundtrip")


async def test_control_frame_filtering():
    t = RawPortTunnel.__new__(RawPortTunnel)
    t._queue = asyncio.Queue(maxsize=16)
    t._ready_event = asyncio.Event()
    t._got_remote_desc = True
    t._pending_candidates = []
    t.state = "open"
    t.tunnel_id = "t"
    t.last_error = ""
    t._tasks = []
    t._last_alive = 0.0

    dc = MockDC()
    t._dc = dc
    t._pc = None

    t._on_message("__ping__")
    assert "__pong__" in dc.sent_frames and "__pong__" in dc.sent_frames
    t._on_message("__ready__")
    assert t._ready_event.is_set()
    t._on_message(b"payload-bytes")
    assert t._queue.get_nowait() == b"payload-bytes"
    t._on_message("__pong__")  # no queue entry
    assert t._queue.empty()
    print("PASS test_control_frame_filtering")


async def test_send_chunking_and_backpressure():
    t = RawPortTunnel.__new__(RawPortTunnel)
    t.state = "open"
    dc = MockDC()
    t._dc = dc

    big = bytes(range(256)) * 600  # 153600 bytes -> 3 chunks of <=60KB
    asyncio.get_running_loop().create_task(t.send(big))
    await asyncio.sleep(0.1)
    frames = dc.sent_frames
    assert all(len(f) <= 60 * 1024 for f in frames if isinstance(f, bytes))
    assert sum(len(f) for f in frames if isinstance(f, bytes)) == len(big)
    print("PASS test_send_chunking_and_backpressure")


async def test_status_and_urls():
    import configparser
    cp = configparser.ConfigParser()
    cp["settings"] = {
        "AI_PROVIDER": "agentic",
        "INFERENCE_NODE": "n1",
        "INFERENCE_BIND_COMFY": "28188",
        "P2P_SIGNALING_URL": "http://127.0.0.1:1/api/signaling",
    }
    m = agentic_tunnel.AgenticManager(cp["settings"])
    assert m.mode == "agentic"
    assert m.active_node == "n1"
    assert m.remote_ollama_port == 11434
    assert m.bind_comfy_port == 28188
    # start with fake ports is not possible here (would bind 11434); just check urls empty
    assert m.urls() == {}
    print("PASS test_status_and_urls")


async def main():
    await test_forwarder_roundtrip()
    await test_control_frame_filtering()
    await test_send_chunking_and_backpressure()
    await test_status_and_urls()
    print("ALL PASSED")


if __name__ == "__main__":
    asyncio.run(main())
