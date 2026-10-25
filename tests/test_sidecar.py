"""
Unit and integration tests for Python Sidecar — Iteration 2 Skeleton

Tests:
  - Startup on loopback (127.0.0.1) with port output
  - Token authentication requirement (missing / wrong token rejected)
  - Origin header validation (reject untrusted browser origins)
  - Hello handshake -> ready + state messages
  - Ping / pong exchange
  - Unknown messages ignored gracefully
"""

import asyncio
from contextlib import asynccontextmanager
import json
import secrets
import pytest
import websockets
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus
from sidecar.server import SidecarServer


@asynccontextmanager
async def run_sidecar(token: str, load_transcriber: bool = False):
    server = SidecarServer(host="127.0.0.1", port=0, token=token, load_transcriber=load_transcriber)
    await server.start()
    try:
        yield server, server.port
    finally:
        await server.stop()


def test_server_binds_to_loopback_and_has_port():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (server, port):
            assert port > 0
            assert server.host == "127.0.0.1"
    asyncio.run(_test())


def test_handshake_with_valid_token_succeeds():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                hello_msg = {
                    "type": "hello",
                    "version": 1,
                    "token": token,
                    "client": "test-suite"
                }
                await ws.send(json.dumps(hello_msg))

                # First expected response: ready
                raw_ready = await asyncio.wait_for(ws.recv(), timeout=2.0)
                ready_msg = json.loads(raw_ready)
                assert ready_msg["type"] == "ready"
                assert ready_msg["version"] == 1
                assert "devices" in ready_msg

                # Second expected response: state (loading or idle depending on model load)
                raw_state = await asyncio.wait_for(ws.recv(), timeout=2.0)
                state_msg = json.loads(raw_state)
                assert state_msg["type"] == "state"
                assert state_msg["state"] in ("loading", "idle")
    asyncio.run(_test())


def test_handshake_with_invalid_token_is_rejected():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                bad_msg = {
                    "type": "hello",
                    "version": 1,
                    "token": "wrong-token-12345"
                }
                await ws.send(json.dumps(bad_msg))

                with pytest.raises(ConnectionClosed) as exc_info:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)
                assert exc_info.value.rcvd.code == 1008
    asyncio.run(_test())


def test_unauthenticated_first_message_is_rejected():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                non_hello_msg = {
                    "type": "ping",
                    "version": 1,
                    "id": 1
                }
                await ws.send(json.dumps(non_hello_msg))

                with pytest.raises(ConnectionClosed) as exc_info:
                    await asyncio.wait_for(ws.recv(), timeout=2.0)
                assert exc_info.value.rcvd.code == 1008
    asyncio.run(_test())


def test_external_origin_header_is_rejected():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            headers = {"Origin": "http://malicious.com"}
            with pytest.raises(InvalidStatus):
                async with connect(uri, additional_headers=headers):
                    pass
    asyncio.run(_test())


def test_ping_pong_exchange():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                await ws.send(json.dumps({"type": "hello", "version": 1, "token": token}))
                await ws.recv()  # ready
                await ws.recv()  # state

                ping_id = 42
                await ws.send(json.dumps({"type": "ping", "version": 1, "id": ping_id}))

                raw_pong = await asyncio.wait_for(ws.recv(), timeout=2.0)
                pong_msg = json.loads(raw_pong)
                assert pong_msg["type"] == "pong"
                assert pong_msg["id"] == ping_id
    asyncio.run(_test())


def test_unknown_message_is_ignored_safely():
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (_, port):
            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                await ws.send(json.dumps({"type": "hello", "version": 1, "token": token}))
                await ws.recv()  # ready
                await ws.recv()  # state

                # Send unknown message type
                await ws.send(json.dumps({"type": "unknown_future_command", "val": 123}))

                # Confirm connection is still healthy and responds to ping
                await ws.send(json.dumps({"type": "ping", "version": 1, "id": 99}))
                raw_pong = await asyncio.wait_for(ws.recv(), timeout=2.0)
                assert json.loads(raw_pong)["type"] == "pong"
    asyncio.run(_test())


def test_drain_does_not_block_ping_pong():
    """
    Fake slow transcriber (slow processing per chunk, 3 chunks queued),
    send 'stop', then send 'ping' mid-drain;
    assert a 'pong' arrives within 500 ms and the final state is 'idle'
    after all 3 transcripts were delivered.
    """
    async def _test():
        token = secrets.token_hex(32)
        async with run_sidecar(token) as (server, port):
            class FakeSlowTranscriber:
                def __init__(self, on_transcript):
                    self.on_transcript = on_transcript
                    self._queue = asyncio.Queue()
                    self._busy = False
                    self.is_ready = True
                    self.config = type("Config", (), {"model_name": "fake"})()
                    self._task = None

                @property
                def pending_count(self):
                    return self._queue.qsize() + (1 if self._busy else 0)

                def has_pending(self):
                    return self.pending_count > 0

                def enqueue(self, seq, audio_path, duration_s, session_dir, t_start=0.0, t_end=0.0, **kwargs):
                    self._queue.put_nowait((seq, audio_path, duration_s, t_start, t_end))
                    if not self._task or self._task.done():
                        self._task = asyncio.create_task(self._worker())

                async def _worker(self):
                    while not self._queue.empty():
                        self._busy = True
                        seq, path, dur, t_start, t_end = await self._queue.get()
                        await asyncio.sleep(0.4)  # Slow chunk processing
                        from sidecar.transcriber import TranscriptionResult
                        res = TranscriptionResult(
                            seq=seq,
                            text=f"Chunk {seq} text",
                            audio_s=dur,
                            transcribe_s=0.4,
                            rtf=0.2,
                            word_count=3,
                            queue_wait_ms=0.0,
                            latency_ms=400.0,
                            queue_depth=self._queue.qsize(),
                            audio_file=path,
                            t_start=t_start,
                            t_end=t_end,
                        )
                        self.on_transcript(res)
                        self._busy = False

                def stop(self):
                    if self._task and not self._task.done():
                        self._task.cancel()

            fake = FakeSlowTranscriber(server._on_transcript)
            server.transcriber = fake

            uri = f"ws://127.0.0.1:{port}"
            async with connect(uri) as ws:
                await ws.send(json.dumps({"type": "hello", "version": 1, "token": token}))
                await ws.recv()  # ready
                await ws.recv()  # state

                # Queue 3 chunks with timeline offsets
                for i in range(1, 4):
                    fake.enqueue(i, f"dummy_{i}.wav", 2.0, "", t_start=(i - 1) * 2.0, t_end=i * 2.0)

                # Send stop
                await ws.send(json.dumps({"type": "stop", "version": 1}))

                # Mid-drain: send ping immediately
                t0 = asyncio.get_event_loop().time()
                await ws.send(json.dumps({"type": "ping", "version": 1, "id": 777}))

                # Assert pong arrives within 500 ms while draining
                pong_received = False
                transcripts_received = []
                final_idle = False

                while not final_idle:
                    raw = await asyncio.wait_for(ws.recv(), timeout=6.0)
                    msg = json.loads(raw)
                    if msg.get("type") == "pong" and msg.get("id") == 777:
                        pong_latency = (asyncio.get_event_loop().time() - t0) * 1000.0
                        assert pong_latency < 500.0, f"Pong latency was {pong_latency}ms (expected < 500ms)"
                        pong_received = True
                    elif msg.get("type") == "transcript":
                        transcripts_received.append(msg)
                    elif msg.get("type") == "state" and msg.get("state") == "idle":
                        final_idle = True

                assert pong_received, "Pong was not received during drain"
                assert len(transcripts_received) == 3, f"Expected 3 transcripts, got {len(transcripts_received)}"
                assert final_idle, "Final state was not idle"
                # Verify timeline offsets in transcript message
                assert transcripts_received[0]["t_start"] == 0.0
                assert transcripts_received[0]["t_end"] == 2.0
                assert transcripts_received[1]["t_start"] == 2.0
                assert transcripts_received[1]["t_end"] == 4.0
                assert transcripts_received[2]["t_start"] == 4.0
                assert transcripts_received[2]["t_end"] == 6.0

    asyncio.run(_test())

