"""
Voice Brainstorm Pad — Python Sidecar Server (Iteration 2 Skeleton)

Responsibilities:
 - Bind strictly to loopback (127.0.0.1) on an OS-assigned free port (port 0)
 - Print VBP_SIDECAR_PORT=<port> to stdout for the Electron supervisor
 - Check Origin header: reject external browser connections with HTTP 403
 - Require per-launch secret token in the initial `hello` handshake message within 3s
 - Close connection with code 1008 (Policy Violation) if token is invalid or missing
 - Emit initial `ready` and `state: idle` messages upon successful handshake
 - Handle `ping` with matching `pong`
 - Safely ignore unknown message types without crashing (TRD 2.5)
"""

import argparse
import asyncio
import hmac
import json
import logging
import os
import signal
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure project root is on sys.path so `sidecar.*` imports work regardless of CWD or execution method
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from sidecar.chunker import AudioChunker
    from sidecar.recorder import AudioRecorder
except ModuleNotFoundError:
    from chunker import AudioChunker  # type: ignore
    from recorder import AudioRecorder  # type: ignore

try:
    from websockets.asyncio.server import serve, Server
    from websockets.exceptions import ConnectionClosed
except ImportError:
    # Fallback if older websockets version is present
    from websockets.server import serve, WebSocketServer as Server  # type: ignore
    from websockets.exceptions import ConnectionClosed  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[sidecar %(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("sidecar")

PROTOCOL_VERSION = 1


def list_audio_devices() -> List[Dict[str, Any]]:
    """Query available microphone devices using sounddevice if available."""
    devices = []
    try:
        import sounddevice as sd  # type: ignore
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({
                    "id": str(i),
                    "name": dev.get("name", f"Microphone {i}"),
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 16000),
                })
    except Exception as e:
        logger.warning(f"Could not query audio devices: {e}")
    return devices


class SidecarServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, token: Optional[str] = None):
        self.host = host
        self.requested_port = port
        self.token = token or os.environ.get("VBP_SIDECAR_TOKEN", "")
        self.server: Optional[Server] = None
        self.port: int = 0
        self.state: str = "idle"
        self._stop_event = asyncio.Event()

        # Recorder & Chunker components (Iteration 3)
        self.recorder: Optional[AudioRecorder] = None
        self.chunker: Optional[AudioChunker] = None
        self.active_device: Optional[str] = None

    async def start(self) -> None:
        """Start the WebSocket server on 127.0.0.1 and record the assigned port."""
        # origins=[None] rejects browser requests that send an Origin header (CORS attack prevention)
        self.server = await serve(
            self.handle_connection,
            self.host,
            self.requested_port,
            origins=[None],
            ping_interval=10,
            ping_timeout=10,
        )
        # Extract actual bound port
        for sock in self.server.sockets:
            self.port = sock.getsockname()[1]
            break

        logger.info(f"Sidecar listening on ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._stop_event.set()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        logger.info("Sidecar server stopped")

    async def handle_connection(self, websocket: Any) -> None:
        """Handle an incoming client connection with strict token authentication."""
        peer = websocket.remote_address
        logger.info(f"Client connected from {peer}")

        # ── Step 1: Authentication Handshake ──────────────────────────────────
        authenticated = False
        try:
            # Client has 3 seconds to send the hello message with the valid token
            raw_msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            msg = json.loads(raw_msg)

            if msg.get("type") == "hello":
                provided_token = str(msg.get("token", ""))
                if self.token and hmac.compare_digest(provided_token, self.token):
                    authenticated = True
                    logger.info("Client authentication successful")
                else:
                    logger.warning("Client authentication failed: invalid token")
            else:
                logger.warning(f"Client sent unexpected first message type: {msg.get('type')}")
        except asyncio.TimeoutError:
            logger.warning("Client timed out waiting for hello handshake")
        except json.JSONDecodeError:
            logger.warning("Client sent invalid JSON during handshake")
        except ConnectionClosed:
            logger.info("Client disconnected during handshake")
            return

        if not authenticated:
            # Code 1008: Policy Violation
            await websocket.close(code=1008, reason="Authentication failed")
            return

        # ── Step 2: Send `ready` and initial `state` ──────────────────────────
        devices = list_audio_devices()
        ready_payload = {
            "type": "ready",
            "version": PROTOCOL_VERSION,
            "engine": "faster-whisper",
            "model": "small.en",
            "devices": devices,
        }
        await websocket.send(json.dumps(ready_payload))

        state_payload = {
            "type": "state",
            "version": PROTOCOL_VERSION,
            "state": self.state,
        }
        await websocket.send(json.dumps(state_payload))

        # ── Step 3: Message Loop ──────────────────────────────────────────────
        try:
            async for raw_message in websocket:
                try:
                    data = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning(f"Malformed JSON received: {raw_message[:100]}")
                    continue

                msg_type = data.get("type")

                if msg_type == "ping":
                    ping_id = data.get("id")
                    pong_msg = {
                        "type": "pong",
                        "version": PROTOCOL_VERSION,
                        "id": ping_id,
                    }
                    await websocket.send(json.dumps(pong_msg))

                elif msg_type == "start":
                    session_folder = data.get("session_folder", "")
                    device_id = data.get("device_id") or self.active_device
                    if not session_folder:
                        logger.warning("Start requested without session_folder")
                        continue

                    loop = asyncio.get_running_loop()

                    def handle_level(rms: float):
                        try:
                            msg = json.dumps({"type": "level", "version": PROTOCOL_VERSION, "rms": rms})
                            asyncio.run_coroutine_threadsafe(websocket.send(msg), loop)
                        except Exception:
                            pass

                    self.chunker = AudioChunker(session_dir=session_folder)

                    def handle_frame(frame: np.ndarray):
                        if not self.chunker:
                            return
                        chunk = self.chunker.process_frame(frame)
                        if chunk:
                            try:
                                chunk_msg = json.dumps({
                                    "type": "chunk_ready",
                                    "version": PROTOCOL_VERSION,
                                    "seq": chunk.seq,
                                    "t_start": chunk.t_start,
                                    "t_end": chunk.t_end,
                                    "duration_s": chunk.duration_s,
                                    "audio_file": chunk.file_path,
                                })
                                asyncio.run_coroutine_threadsafe(websocket.send(chunk_msg), loop)
                            except Exception:
                                pass

                    if self.recorder:
                        self.recorder.stop()

                    self.recorder = AudioRecorder(
                        on_level=handle_level,
                        on_frame=handle_frame,
                    )
                    self.recorder.start(session_dir=session_folder, device=device_id)
                    self.state = "listening"
                    await websocket.send(json.dumps({
                        "type": "state",
                        "version": PROTOCOL_VERSION,
                        "state": "listening",
                    }))
                    logger.info(f"Dictation started for session: {session_folder}")

                elif msg_type == "stop":
                    if self.recorder:
                        self.recorder.stop()
                        self.recorder = None

                    if self.chunker:
                        final_chunk = self.chunker.flush()
                        if final_chunk:
                            await websocket.send(json.dumps({
                                "type": "chunk_ready",
                                "version": PROTOCOL_VERSION,
                                "seq": final_chunk.seq,
                                "t_start": final_chunk.t_start,
                                "t_end": final_chunk.t_end,
                                "duration_s": final_chunk.duration_s,
                                "audio_file": final_chunk.file_path,
                            }))
                        self.chunker = None

                    self.state = "idle"
                    await websocket.send(json.dumps({
                        "type": "state",
                        "version": PROTOCOL_VERSION,
                        "state": "idle",
                    }))
                    logger.info("Dictation stopped cleanly")

                elif msg_type == "set_device":
                    self.active_device = data.get("device_id")
                    logger.info(f"Active input device set to: {self.active_device}")

                elif msg_type == "list_devices":
                    devices = list_audio_devices()
                    await websocket.send(json.dumps({
                        "type": "devices",
                        "version": PROTOCOL_VERSION,
                        "devices": devices,
                    }))

                elif msg_type == "hello":
                    # Re-hello after reconnect or refresh
                    await websocket.send(json.dumps({
                        "type": "state",
                        "version": PROTOCOL_VERSION,
                        "state": self.state,
                    }))

                else:
                    # TRD 2.5: Unknown message types are logged and ignored, not fatal
                    logger.info(f"Ignoring unknown message type: {msg_type}")

        except ConnectionClosed:
            logger.info(f"Client {peer} disconnected")
        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            if self.recorder:
                self.recorder.stop()
                self.recorder = None
            if self.chunker:
                self.chunker.flush()
                self.chunker = None
            self.state = "idle"


async def run_server() -> None:
    parser = argparse.ArgumentParser(description="Voice Brainstorm Pad — Speech Sidecar")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on (0 for dynamic)")
    parser.add_argument("--token", type=str, default="", help="Auth token (or env VBP_SIDECAR_TOKEN)")
    args = parser.parse_args()

    token = args.token or os.environ.get("VBP_SIDECAR_TOKEN", "")
    if not token:
        logger.error("No token provided via --token or VBP_SIDECAR_TOKEN")
        sys.exit(1)

    server = SidecarServer(host="127.0.0.1", port=args.port, token=token)
    await server.start()

    # CRITICAL: Print the port line to stdout and flush so Electron supervisor receives it
    print(f"VBP_SIDECAR_PORT={server.port}", flush=True)

    # Setup signal handlers for clean exit
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_exit(*_):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_exit)
        except (NotImplementedError, AttributeError):
            # Windows does not support add_signal_handler for all signals
            pass

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
