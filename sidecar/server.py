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
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
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
    from sidecar.transcriber import Transcriber, TranscriberConfig, TranscriptionResult
except ModuleNotFoundError:
    from chunker import AudioChunker  # type: ignore
    from recorder import AudioRecorder  # type: ignore
    from transcriber import Transcriber, TranscriberConfig, TranscriptionResult  # type: ignore

try:
    from websockets.asyncio.server import serve, Server
    from websockets.exceptions import ConnectionClosed
    from websockets.protocol import State
except ImportError:
    # Fallback if older websockets version is present
    from websockets.server import serve, WebSocketServer as Server  # type: ignore
    from websockets.exceptions import ConnectionClosed  # type: ignore
    try:
        from websockets.protocol import State  # type: ignore
    except ImportError:
        State = None

logging.basicConfig(
    level=logging.INFO,
    format="[sidecar %(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("sidecar")

PROTOCOL_VERSION = 1


def list_audio_devices() -> List[Dict[str, Any]]:
    """Query available microphone devices using sounddevice, validating each against 16kHz mono."""
    devices = []
    try:
        import sounddevice as sd  # type: ignore
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                hostapi_name = hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else ""
                # Skip WDM-KS as PortAudio throws -9996 on Windows WDM-KS shared streams
                if "WDM-KS" in hostapi_name:
                    continue
                # Verify that the device can actually be opened for 16kHz mono float32 input
                try:
                    sd.check_input_settings(device=i, samplerate=16000, channels=1)
                except Exception:
                    continue

                dev_name = dev.get("name", f"Microphone {i}")
                devices.append({
                    "id": str(i),
                    "name": f"{dev_name} ({hostapi_name})",
                    "channels": dev.get("max_input_channels", 1),
                    "default_samplerate": dev.get("default_samplerate", 16000),
                })
    except (ImportError, OSError) as e:
        logger.warning(f"PortAudio or sounddevice unavailable: {e}")
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
        self.state: str = "loading"
        self._stop_event = asyncio.Event()

        # Recorder, Chunker & Transcriber components (Iteration 3 & 4)
        self.recorder: Optional[AudioRecorder] = None
        self.chunker: Optional[AudioChunker] = None
        self.transcriber: Optional[Transcriber] = None
        self.active_device: Optional[str] = None
        self.current_session_folder: Optional[str] = None
        self._active_ws: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._drain_task: Optional[asyncio.Task] = None

    def _on_transcript(self, result: TranscriptionResult) -> None:
        """Called by Transcriber worker thread when transcription of a chunk completes."""
        logger.info(
            f"[Transcriber] chunk #{result.seq} ({result.audio_s:.1f}s, RTF {result.rtf:.2f}x, {result.word_count} words, {result.latency_ms:.0f}ms latency) -> \"{result.text}\""
        )
        ws = self._active_ws
        loop = self._loop
        if ws and loop:
            is_open = True
            if hasattr(ws, "state"):
                is_open = (ws.state == State.OPEN) if State else (getattr(ws.state, "name", "") == "OPEN")
            elif hasattr(ws, "closed"):
                is_open = not ws.closed

            if is_open:
                try:
                    msg = json.dumps({
                        "type": "transcript",
                        "version": PROTOCOL_VERSION,
                        "seq": result.seq,
                        "text": result.text,
                        "duration_s": result.audio_s,
                        "transcribe_s": round(result.transcribe_s, 3),
                        "rtf": round(result.rtf, 3),
                        "words": result.word_count,
                        "queue_wait_ms": round(result.queue_wait_ms, 1),
                        "latency_ms": round(result.latency_ms, 1),
                        "queue_depth": result.queue_depth,
                        "audio_file": result.audio_file,
                        "t_start": getattr(result, "t_start", 0.0),
                        "t_end": getattr(result, "t_end", 0.0),
                    })
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except Exception as e:
                    logger.error(f"Error sending transcript message: {e}")

    def _on_chunk_failed(self, data: dict) -> None:
        """Called by Transcriber worker thread when transcription of a chunk fails."""
        logger.warning(
            f"[Transcriber] chunk #{data.get('seq')} failed transcription: {data.get('error')}"
        )
        ws = self._active_ws
        loop = self._loop
        if ws and loop:
            is_open = True
            if hasattr(ws, "state"):
                is_open = (ws.state == State.OPEN) if State else (getattr(ws.state, "name", "") == "OPEN")
            elif hasattr(ws, "closed"):
                is_open = not ws.closed

            if is_open:
                try:
                    msg = json.dumps({
                        "type": "chunk_failed",
                        "version": PROTOCOL_VERSION,
                        "seq": data["seq"],
                        "audio_file": data["audio_file"],
                        "error": data["error"],
                    })
                    asyncio.run_coroutine_threadsafe(ws.send(msg), loop)
                except Exception as e:
                    logger.error(f"Error sending chunk_failed message: {e}")

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

        # Initialize Whisper Transcriber worker asynchronously so port announcement is instant (Iteration 4)
        loop = asyncio.get_running_loop()
        def _load():
            try:
                self.transcriber = Transcriber(
                    config=TranscriberConfig(model_name="small.en", cpu_threads=4),
                    on_transcript=self._on_transcript,
                    on_chunk_failed=self._on_chunk_failed,
                    auto_warmup=True,
                )
                self.state = "idle"
            except Exception as e:
                logger.warning(f"Could not load small.en model ({e}), falling back to base.en")
                try:
                    self.transcriber = Transcriber(
                        config=TranscriberConfig(model_name="base.en", cpu_threads=4),
                        on_transcript=self._on_transcript,
                        on_chunk_failed=self._on_chunk_failed,
                        auto_warmup=True,
                    )
                    self.state = "idle"
                except Exception as e2:
                    logger.error(f"Failed to load Whisper model: {e2}")
                    self.state = "error"

            # If client is already connected during model load, push updated state & ready info
            if self._active_ws and self._loop:
                devices = list_audio_devices()
                ready_payload = json.dumps({
                    "type": "ready",
                    "version": PROTOCOL_VERSION,
                    "engine": "faster-whisper",
                    "model": self.transcriber.config.model_name if self.transcriber else "",
                    "devices": devices,
                })
                state_payload = json.dumps({
                    "type": "state",
                    "version": PROTOCOL_VERSION,
                    "state": self.state,
                })
                try:
                    asyncio.run_coroutine_threadsafe(self._active_ws.send(ready_payload), self._loop)
                    asyncio.run_coroutine_threadsafe(self._active_ws.send(state_payload), self._loop)
                except Exception:
                    pass

        self._load_task = loop.run_in_executor(None, _load)

    async def stop(self) -> None:
        """Stop the WebSocket server and background workers."""
        self._stop_event.set()
        if self.transcriber:
            self.transcriber.stop()
            self.transcriber = None
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
        self._active_ws = websocket
        self._loop = asyncio.get_running_loop()

        devices = list_audio_devices()
        ready_payload = {
            "type": "ready",
            "version": PROTOCOL_VERSION,
            "engine": "faster-whisper",
            "model": self.transcriber.config.model_name if (self.transcriber and self.transcriber.is_ready) else "",
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
                    if self._drain_task and not self._drain_task.done():
                        self._drain_task.cancel()

                    if self._load_task and not self._load_task.done():
                        logger.info("Waiting for background transcriber warmup to finish before starting dictation...")
                        await self._load_task
                        self._load_task = None

                    if not self.transcriber or not self.transcriber.is_ready:
                        logger.error("Refusing to enter listening: transcriber is not ready or failed to load")
                        await websocket.send(json.dumps({
                            "type": "error",
                            "version": PROTOCOL_VERSION,
                            "message": "Whisper transcriber is not ready or failed to load.",
                        }))
                        await websocket.send(json.dumps({
                            "type": "state",
                            "version": PROTOCOL_VERSION,
                            "state": self.state,
                        }))
                        continue

                    session_folder = data.get("session_folder", "")
                    dev_arg = data.get("device_id")
                    if dev_arg is not None and str(dev_arg).strip() != "":
                        device_id = str(dev_arg).strip()
                    else:
                        device_id = self.active_device

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

                    self.current_session_folder = session_folder
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
                                    "device_name": getattr(self.recorder, "active_device_name", "Default Microphone"),
                                })
                                asyncio.run_coroutine_threadsafe(websocket.send(chunk_msg), loop)
                            except Exception:
                                pass

                            # Enqueue into Transcriber worker (Iteration 4)
                            if self.transcriber:
                                self.transcriber.enqueue(
                                    seq=chunk.seq,
                                    audio_path=chunk.file_path,
                                    duration_s=chunk.duration_s,
                                    session_dir=session_folder,
                                    t_start=chunk.t_start,
                                    t_end=chunk.t_end,
                                )

                    if self.recorder:
                        self.recorder.stop()

                    if self.transcriber:
                        self.transcriber.reset_context()

                    self.recorder = AudioRecorder(
                        on_level=handle_level,
                        on_frame=handle_frame,
                    )
                    try:
                        self.recorder.start(session_dir=session_folder, device=device_id)
                        self.state = "listening"
                        dev_name = getattr(self.recorder, "active_device_name", "Default Microphone")
                        dev_id = getattr(self.recorder, "active_device_id", None)
                        logger.info(f"Dictation started for session: {session_folder} on device [{dev_id}]: {dev_name}")
                        await websocket.send(json.dumps({
                            "type": "state",
                            "version": PROTOCOL_VERSION,
                            "state": "listening",
                        }))
                        await websocket.send(json.dumps({
                            "type": "device_info",
                            "version": PROTOCOL_VERSION,
                            "device_id": str(dev_id) if dev_id is not None else "",
                            "device_name": dev_name,
                        }))
                    except Exception as e:
                        logger.error(f"Failed to start recording: {e}")
                        self.recorder = None
                        self.state = "idle"
                        await websocket.send(json.dumps({
                            "type": "error",
                            "version": PROTOCOL_VERSION,
                            "message": f"Microphone error: {e}",
                        }))
                        await websocket.send(json.dumps({
                            "type": "state",
                            "version": PROTOCOL_VERSION,
                            "state": "idle",
                        }))

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
                            if self.transcriber:
                                self.transcriber.enqueue(
                                    seq=final_chunk.seq,
                                    audio_path=final_chunk.file_path,
                                    duration_s=final_chunk.duration_s,
                                    session_dir=self.current_session_folder or "",
                                    t_start=final_chunk.t_start,
                                    t_end=final_chunk.t_end,
                                )
                        self.chunker = None

                    # Drain transcriber queue asynchronously via background task (Catch-up blocker fix)
                    if self.transcriber and self.transcriber.has_pending():
                        if self._drain_task and not self._drain_task.done():
                            self._drain_task.cancel()
                        self._drain_task = asyncio.create_task(self._drain_then_idle())
                    else:
                        self.state = "idle"
                        await websocket.send(json.dumps({
                            "type": "state",
                            "version": PROTOCOL_VERSION,
                            "state": "idle",
                        }))
                    logger.info("Dictation stopped cleanly")

                elif msg_type == "set_device":
                    raw_dev = data.get("device_id")
                    if raw_dev is not None and str(raw_dev).strip() != "":
                        self.active_device = str(raw_dev).strip()
                    else:
                        self.active_device = None

                    logger.info(f"Active input device set to: {self.active_device or 'Default Microphone'}")
                    if self.recorder and self.state == "listening":
                        self.recorder.switch_device(self.active_device)
                        dev_name = getattr(self.recorder, "active_device_name", "Default Microphone")
                        dev_id = getattr(self.recorder, "active_device_id", None)
                        await websocket.send(json.dumps({
                            "type": "device_info",
                            "version": PROTOCOL_VERSION,
                            "device_id": str(dev_id) if dev_id is not None else "",
                            "device_name": dev_name,
                        }))

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
            if self._drain_task and not self._drain_task.done():
                self._drain_task.cancel()
            if self._active_ws == websocket:
                self._active_ws = None
            if self.recorder:
                self.recorder.stop()
                self.recorder = None
            if self.chunker:
                final_chunk = self.chunker.flush()
                if final_chunk and self.transcriber:
                    self.transcriber.enqueue(
                        seq=final_chunk.seq,
                        audio_path=final_chunk.file_path,
                        duration_s=final_chunk.duration_s,
                        session_dir=self.current_session_folder or "",
                        t_start=final_chunk.t_start,
                        t_end=final_chunk.t_end,
                    )
                self.chunker = None
            self.state = "idle"

    async def _drain_then_idle(self) -> None:
        """Drain transcriber queue in background without blocking WebSocket loop."""
        try:
            ws = self._active_ws
            if not ws or not self.transcriber:
                self.state = "idle"
                return

            self.state = "catching_up"
            last_n = self.transcriber.pending_count
            await ws.send(json.dumps({
                "type": "state",
                "version": PROTOCOL_VERSION,
                "state": "catching_up",
                "pending_chunks": last_n,
            }))

            while self.transcriber and self.transcriber.has_pending():
                await asyncio.sleep(0.05)
                cur_n = self.transcriber.pending_count
                if cur_n != last_n:
                    last_n = cur_n
                    if last_n > 0:
                        await ws.send(json.dumps({
                            "type": "state",
                            "version": PROTOCOL_VERSION,
                            "state": "catching_up",
                            "pending_chunks": last_n,
                        }))

            self.state = "idle"
            await ws.send(json.dumps({
                "type": "state",
                "version": PROTOCOL_VERSION,
                "state": "idle",
            }))
            logger.info("Transcriber queue fully drained; sidecar transitioned to idle")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error during drain_then_idle: {e}")



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
