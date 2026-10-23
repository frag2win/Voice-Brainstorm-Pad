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
async def run_sidecar(token: str):
    server = SidecarServer(host="127.0.0.1", port=0, token=token)
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

                # Second expected response: state (idle)
                raw_state = await asyncio.wait_for(ws.recv(), timeout=2.0)
                state_msg = json.loads(raw_state)
                assert state_msg["type"] == "state"
                assert state_msg["state"] == "idle"
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
