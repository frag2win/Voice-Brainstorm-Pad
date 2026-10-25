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
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import queue
import re
import threading
import time
from typing import Any, Callable, Optional

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


def strip_prompt_echo(text: str, prompt: str) -> str:
    """If output text starts with or equals the tail of the prompt, strip the repeated part."""
    if not text or not prompt:
        return text

    p_words = prompt.strip().split()
    t_words = text.strip().split()
    if not p_words or not t_words:
        return text

    def normalize_word(w: str) -> str:
        return re.sub(r"[^\w]", "", w).lower()

    norm_p = [normalize_word(w) for w in p_words]
    norm_t = [normalize_word(w) for w in t_words]

    # Look for longest matching tail of prompt at the start of text
    max_k = min(len(norm_p), len(norm_t))
    best_match_k = 0
    for k in range(max_k, 0, -1):
        if norm_p[-k:] == norm_t[:k]:
            best_match_k = k
            break

    if best_match_k > 0:
        remaining_words = t_words[best_match_k:]
        return " ".join(remaining_words).strip()

    return text


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
    temperature: tuple = (0.0, 0.2, 0.4)
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6


@dataclass
class TranscriptionResult:
    seq: int
    text: str
    audio_s: float
    transcribe_s: float
    rtf: float
    word_count: int
    queue_wait_ms: float
    latency_ms: float
    queue_depth: int
    audio_file: str
    t_start: float = 0.0
    t_end: float = 0.0


class Transcriber:
    def __init__(
        self,
        config: Optional[TranscriberConfig] = None,
        on_transcript: Optional[Callable[[TranscriptionResult], None]] = None,
        on_chunk_failed: Optional[Callable[[dict], None]] = None,
        auto_warmup: bool = True,
        model: Optional[Any] = None,
    ):
        self.config = config or TranscriberConfig()
        self.on_transcript = on_transcript
        self.on_chunk_failed = on_chunk_failed
        self.model: Optional[WhisperModel] = model
        self.is_ready = model is not None
        self._last_prompt_text = ""

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.model is None:
            self._load_model(auto_warmup=auto_warmup)
        self._start_worker()

    @property
    def pending_count(self) -> int:
        return self._queue.unfinished_tasks

    def has_pending(self) -> bool:
        return self._queue.unfinished_tasks > 0

    def reset_context(self) -> None:
        """Reset prompt context when a new recording starts."""
        self._last_prompt_text = ""

    def _load_model(self, auto_warmup: bool = True) -> None:
        """Load the faster-whisper model, attempting local offline cache first (G4, NFR-6)."""
        if WhisperModel is None:
            logger.error("faster-whisper is not installed")
            return

        t0 = time.time()
        logger.info(
            f"Loading Whisper model '{self.config.model_name}' ({self.config.compute_type}, {self.config.cpu_threads} threads)..."
        )
        try:
            # Try offline local cache first
            self.model = WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                cpu_threads=self.config.cpu_threads,
                num_workers=self.config.num_workers,
                local_files_only=True,
            )
        except Exception as e:
            err_msg = str(e).lower()
            is_cache_miss = (
                "not found in cache" in err_msg
                or "local_files_only" in err_msg
                or "entry not found" in err_msg
                or "file not found" in err_msg
                or isinstance(e, (FileNotFoundError, OSError))
            )
            if not is_cache_miss:
                logger.error(f"Failed to load Whisper model from cache: {e}")
                raise

            logger.warning(
                f"Whisper model '{self.config.model_name}' not found in local cache. Falling back to downloading model..."
            )
            self.model = WhisperModel(
                self.config.model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
                cpu_threads=self.config.cpu_threads,
                num_workers=self.config.num_workers,
                local_files_only=False,
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

    def enqueue(
        self,
        seq: int,
        audio_path: str,
        duration_s: float,
        session_dir: str,
        t_start: float = 0.0,
        t_end: float = 0.0,
    ) -> None:
        """Enqueue an audio chunk for transcription."""
        item = {
            "seq": seq,
            "audio_path": audio_path,
            "duration_s": duration_s,
            "session_dir": session_dir,
            "t_start": t_start,
            "t_end": t_end,
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

        # Collect per-chunk diagnostics at start of chunk
        rss_mb = 0.0
        avail_ram_mb = 0.0
        try:
            import psutil
            proc = psutil.Process()
            rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)
            avail_ram_mb = round(psutil.virtual_memory().available / (1024 * 1024), 2)
        except Exception as e:
            logger.debug(f"Could not retrieve memory diagnostics: {e}")

        if not self.model or not os.path.exists(audio_path):
            return

        t0 = time.time()
        queue_wait_ms = (t0 - item["enqueued_at"]) * 1000

        # TRD D3: Initial prompt from previous chunk for context continuity
        prompt = self._last_prompt_text if self._last_prompt_text else None

        texts = []
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=self.config.language,
                beam_size=self.config.beam_size,
                condition_on_previous_text=self.config.condition_on_previous_text,
                temperature=self.config.temperature,
                compression_ratio_threshold=self.config.compression_ratio_threshold,
                log_prob_threshold=self.config.log_prob_threshold,
                no_speech_threshold=self.config.no_speech_threshold,
                initial_prompt=prompt,
                vad_filter=False,  # VAD already handled by Silero in chunker
            )

            # Check if temperature fallback happened
            used_temp = getattr(info, "temperature", None)
            initial_temp = (
                self.config.temperature[0]
                if isinstance(self.config.temperature, (tuple, list))
                else self.config.temperature
            )
            if used_temp is not None and used_temp > initial_temp:
                logger.warning(
                    f"Chunk #{seq}: temperature fallback occurred (used temperature {used_temp} > initial {initial_temp})"
                )

            try:
                for s in segments:
                    seg_temp = getattr(s, "temperature", None)
                    if seg_temp is not None and seg_temp > initial_temp and used_temp == initial_temp:
                        logger.warning(
                            f"Chunk #{seq}: segment temperature fallback occurred (temperature {seg_temp} > initial {initial_temp})"
                        )
                    # Keep segments unless confidence is overwhelmingly non-speech
                    if getattr(s, "no_speech_prob", 0.0) > 0.90 and getattr(s, "avg_logprob", 0.0) < -1.5:
                        continue
                    txt = s.text.strip()
                    if txt:
                        texts.append(txt)
            except RuntimeError as e:
                if "mkl_malloc" in str(e).lower() or "memory" in str(e).lower():
                    logger.warning(
                        f"MKL memory pressure during chunk #{seq} ({e}), running garbage collection and retrying with greedy decoding..."
                    )
                    import gc
                    gc.collect()
                    # Protect the second pass with its own try/except
                    try:
                        retry_segments, _ = self.model.transcribe(
                            audio_path,
                            language=self.config.language,
                            beam_size=1,
                            condition_on_previous_text=False,
                            temperature=0.0,
                            vad_filter=False,
                        )
                        texts = []
                        for s in retry_segments:
                            txt = s.text.strip()
                            if txt:
                                texts.append(txt)
                    except Exception as retry_err:
                        self._handle_chunk_failure(session_dir, seq, audio_path, str(retry_err), rss_mb, avail_ram_mb)
                        return
                else:
                    self._handle_chunk_failure(session_dir, seq, audio_path, str(e), rss_mb, avail_ram_mb)
                    return
        except Exception as main_err:
            self._handle_chunk_failure(session_dir, seq, audio_path, str(main_err), rss_mb, avail_ram_mb)
            return

        raw_text = " ".join(texts)
        clean_text = clean_transcript(raw_text)

        # Apply prompt echo guard
        if clean_text and prompt:
            clean_text = strip_prompt_echo(clean_text, prompt)

        transcribe_s = time.time() - t0
        rtf = transcribe_s / max(0.01, duration_s)
        word_count = len(clean_text.split()) if clean_text else 0
        latency_ms = (duration_s * 1000.0) + ((time.time() - item["enqueued_at"]) * 1000.0)

        if clean_text:
            # Retain last 25 words as initial_prompt for the next chunk
            words = clean_text.split()
            self._last_prompt_text = " ".join(words[-25:])

        logger.info(
            f"[Transcriber] chunk #{seq} ({duration_s:.1f}s, RTF {rtf:.2f}x, {word_count} words, {latency_ms:.0f}ms end-to-end latency) -> \"{clean_text}\""
        )

        res = TranscriptionResult(
            seq=seq,
            text=clean_text,
            audio_s=duration_s,
            transcribe_s=transcribe_s,
            rtf=rtf,
            word_count=word_count,
            queue_wait_ms=queue_wait_ms,
            latency_ms=latency_ms,
            queue_depth=queue_depth,
            audio_file=audio_path,
            t_start=item.get("t_start", 0.0),
            t_end=item.get("t_end", 0.0),
        )

        # Append to append-only transcripts.jsonl with memory diagnostics
        self._append_transcript(session_dir, res, rss_mb=rss_mb, avail_ram_mb=avail_ram_mb)

        if self.on_transcript:
            try:
                self.on_transcript(res)
            except Exception as e:
                logger.error(f"Error in on_transcript callback: {e}")

    def _handle_chunk_failure(
        self,
        session_dir: str,
        seq: int,
        audio_file: str,
        error_msg: str,
        rss_mb: float,
        avail_ram_mb: float,
    ) -> None:
        """Append failure record to transcripts.jsonl and notify chunk_failed callback."""
        logger.error(f"Chunk #{seq} failed transcription: {error_msg}")

        if session_dir and os.path.exists(session_dir):
            transcripts_path = os.path.join(session_dir, "transcripts.jsonl")
            entry = {
                "seq": seq,
                "failed": True,
                "error": error_msg,
                "audio_file": audio_file,
                "rss_mb": rss_mb,
                "avail_ram_mb": avail_ram_mb,
                "timestamp": time.time(),
            }
            try:
                with open(transcripts_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()
            except Exception as e:
                logger.error(f"Could not append failure record to transcripts.jsonl: {e}")

        if self.on_chunk_failed:
            try:
                self.on_chunk_failed({
                    "type": "chunk_failed",
                    "seq": seq,
                    "audio_file": audio_file,
                    "error": error_msg,
                })
            except Exception as e:
                logger.error(f"Error in on_chunk_failed callback: {e}")

    def _append_transcript(
        self,
        session_dir: str,
        res: TranscriptionResult,
        rss_mb: float = 0.0,
        avail_ram_mb: float = 0.0,
    ) -> None:
        """Append transcription result to append-only transcripts.jsonl."""
        if not session_dir or not os.path.exists(session_dir):
            return

        transcripts_path = os.path.join(session_dir, "transcripts.jsonl")
        entry = {
            "seq": res.seq,
            "text": res.text,
            "words": res.word_count,
            "duration_s": round(res.audio_s, 3),
            "transcribe_s": round(res.transcribe_s, 3),
            "rtf": round(res.rtf, 3),
            "queue_wait_ms": round(res.queue_wait_ms, 1),
            "latency_ms": round(res.latency_ms, 1),
            "t_start": round(res.t_start, 3),
            "t_end": round(res.t_end, 3),
            "rss_mb": rss_mb,
            "avail_ram_mb": avail_ram_mb,
            "model": self.config.model_name,
            "audio_file": res.audio_file,
            "timestamp": time.time(),
        }
        try:
            with open(transcripts_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
        except Exception as e:
            logger.warning(f"Could not append to transcripts.jsonl for chunk #{res.seq}: {e}")

    def wait_drained(self, timeout: float = 15.0) -> bool:
        """Wait until all queued and in-flight chunks have finished transcribing."""
        t_start = time.time()
        while self._queue.unfinished_tasks > 0:
            if time.time() - t_start > timeout:
                logger.warning("Timed out waiting for transcriber queue to drain")
                return False
            time.sleep(0.05)
        return True

    def stop(self) -> None:
        """Stop transcriber worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
