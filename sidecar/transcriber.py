"""
Transcriber Worker — Voice Brainstorm Pad Iteration 4

Responsibilities:
  - Run faster-whisper local model (default: small.en, int8, CPU)
  - Process audio chunks from a FIFO queue strictly in sequence
  - Hallucination suppression (clean_transcript, beam_size=1, condition_on_previous_text=False)
  - Calculate latency and Real-Time Factor (RTF) metrics per chunk
  - Append transcribed text and metrics to session chunks.jsonl
  - Non-blocking worker thread communicating results via callbacks
"""

from dataclasses import dataclass
import json
import logging
import os
import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

logger = logging.getLogger("transcriber")

# Known hallucination phrases emitted by Whisper on silence / quiet audio
HALLUCINATION_PATTERNS = [
    re.compile(r"^thank you( for watching)?\.?$", re.IGNORECASE),
    re.compile(r"^please subscribe\.?$", re.IGNORECASE),
    re.compile(r"^subtitles by.*$", re.IGNORECASE),
    re.compile(r"^\[.*?\]$"),
    re.compile(r"^\(.*?\)$"),
    re.compile(r"^you$", re.IGNORECASE),
    re.compile(r"^\.+$"),
]


def clean_transcript(text: str) -> str:
    """Filter out Whisper silence hallucinations and format whitespace."""
    if not text:
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    # Check known single-phrase hallucinations
    for pattern in HALLUCINATION_PATTERNS:
        if pattern.match(cleaned):
            return ""

    # Check for excessive word repetition (e.g. "so so so so so")
    words = cleaned.split()
    if len(words) >= 6:
        unique_words = set(w.lower() for w in words)
        if len(unique_words) == 1:
            return ""

    return cleaned


@dataclass
class TranscriberConfig:
    model_name: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 4
    num_workers: int = 1
    language: str = "en"
    beam_size: int = 1
    condition_on_previous_text: bool = False
    temperature: float = 0.0


@dataclass
class TranscriptionResult:
    seq: int
    text: str
    audio_s: float
    transcribe_s: float
    rtf: float
    word_count: int
    latency_ms: float
    queue_depth: int
    audio_file: str


class Transcriber:
    def __init__(
        self,
        config: Optional[TranscriberConfig] = None,
        on_transcript: Optional[Callable[[TranscriptionResult], None]] = None,
        auto_warmup: bool = True,
    ):
        self.config = config or TranscriberConfig()
        self.on_transcript = on_transcript
        self.model: Optional[WhisperModel] = None
        self.is_ready = False

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._load_model(auto_warmup=auto_warmup)
        self._start_worker()

    def _load_model(self, auto_warmup: bool = True) -> None:
        """Load the faster-whisper model and optionally run warmup."""
        if WhisperModel is None:
            logger.error("faster-whisper is not installed")
            return

        t0 = time.time()
        logger.info(
            f"Loading Whisper model '{self.config.model_name}' ({self.config.compute_type}, {self.config.cpu_threads} threads)..."
        )
        self.model = WhisperModel(
            self.config.model_name,
            device=self.config.device,
            compute_type=self.config.compute_type,
            cpu_threads=self.config.cpu_threads,
            num_workers=self.config.num_workers,
        )
        load_time = time.time() - t0
        logger.info(f"Whisper model loaded in {load_time:.2f}s")
        self.is_ready = True

        if auto_warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Warm up engine with 1 second of synthetic audio so first chunk is fast."""
        if not self.model:
            return
        try:
            logger.info("Running 1s warmup inference...")
            t0 = time.time()
            dummy_audio = np.zeros(16000, dtype=np.float32)
            list(
                self.model.transcribe(
                    dummy_audio,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    condition_on_previous_text=False,
                )[0]
            )
            logger.info(f"Warmup complete in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def enqueue(self, seq: int, audio_path: str, duration_s: float, session_dir: str) -> None:
        """Enqueue an audio chunk for transcription."""
        item = {
            "seq": seq,
            "audio_path": audio_path,
            "duration_s": duration_s,
            "session_dir": session_dir,
            "enqueued_at": time.time(),
        }
        self._queue.put(item)

    def _worker_loop(self) -> None:
        """Sequential worker loop."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._transcribe_item(item)
            except Exception as e:
                logger.error(f"Error transcribing chunk #{item.get('seq')}: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def _transcribe_item(self, item: dict) -> None:
        seq = item["seq"]
        audio_path = item["audio_path"]
        duration_s = item["duration_s"]
        session_dir = item["session_dir"]
        queue_depth = self._queue.qsize()

        if not self.model or not os.path.exists(audio_path):
            return

        t0 = time.time()
        segments, _ = self.model.transcribe(
            audio_path,
            language=self.config.language,
            beam_size=self.config.beam_size,
            condition_on_previous_text=self.config.condition_on_previous_text,
            temperature=self.config.temperature,
            vad_filter=False,  # We already did VAD in chunker
        )

        texts = []
        for s in segments:
            # If segment is high probability non-speech, skip it
            if getattr(s, "no_speech_prob", 0.0) > 0.70:
                continue
            txt = s.text.strip()
            if txt:
                texts.append(txt)

        raw_text = " ".join(texts)
        clean_text = clean_transcript(raw_text)

        transcribe_s = time.time() - t0
        rtf = transcribe_s / max(0.01, duration_s)
        word_count = len(clean_text.split()) if clean_text else 0
        latency_ms = (time.time() - item["enqueued_at"]) * 1000

        # Log metrics to terminal (Iteration 4 core requirement)
        logger.info(
            f"[Transcriber] chunk #{seq} ({duration_s:.1f}s, RTF {rtf:.2f}x, {word_count} words) -> \"{clean_text}\""
        )

        # Update chunks.jsonl with transcription details
        self._update_jsonl(session_dir, seq, clean_text, rtf, transcribe_s)

        res = TranscriptionResult(
            seq=seq,
            text=clean_text,
            audio_s=duration_s,
            transcribe_s=transcribe_s,
            rtf=rtf,
            word_count=word_count,
            latency_ms=latency_ms,
            queue_depth=queue_depth,
            audio_file=audio_path,
        )

        if self.on_transcript:
            try:
                self.on_transcript(res)
            except Exception as e:
                logger.error(f"Error in on_transcript callback: {e}")

    def _update_jsonl(self, session_dir: str, seq: int, text: str, rtf: float, transcribe_s: float) -> None:
        """Update audit entry in chunks.jsonl with text and metrics."""
        jsonl_path = os.path.join(session_dir, "chunks.jsonl")
        if not os.path.exists(jsonl_path):
            return

        try:
            lines = []
            updated = False
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    data = json.loads(line_str)
                    if data.get("seq") == seq:
                        data["text"] = text
                        data["rtf"] = round(rtf, 3)
                        data["transcribe_ms"] = round(transcribe_s * 1000, 1)
                        data["model"] = self.config.model_name
                        updated = True
                    lines.append(json.dumps(data) + "\n")

            if updated:
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
        except Exception as e:
            logger.warning(f"Could not update chunks.jsonl for chunk #{seq}: {e}")

    def wait_drained(self, timeout: float = 15.0) -> None:
        """Wait until all queued chunks have been transcribed."""
        t_start = time.time()
        while not self._queue.empty():
            if time.time() - t_start > timeout:
                logger.warning("Timed out waiting for transcriber queue to drain")
                break
            time.sleep(0.05)

    def stop(self) -> None:
        """Stop transcriber worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
