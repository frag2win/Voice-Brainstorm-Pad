"""
VAD & Audio Chunker — Voice Brainstorm Pad Iteration 3

Responsibilities:
  - Ingest 16 kHz mono float32 audio frames (512 samples = 32 ms)
  - Run Silero VAD (ONNX) to distinguish speech and pause frames
  - Enforce TRD 2.3 chunking parameters:
      min_chunk_s: 4.0 s
      target_chunk_s: 8.0 - 12.0 s
      max_chunk_s: 20.0 s
      min_silence_s: 0.3 s (300 ms pause cut threshold)
  - Cut chunks at natural speech pauses once target duration is reached
  - Cut at max duration (20s) if speaker speaks without pausing
  - Discard pure non-speech chunks (never emit empty/silence chunks)
  - Save chunk audio to <session_dir>/chunks/chunk_XXXX.wav (PCM16)
  - Append audit entry to <session_dir>/chunks.jsonl
"""

from dataclasses import dataclass
import json
import os
import struct
from typing import Any, List, Optional
import numpy as np

try:
    from faster_whisper.vad import get_vad_model
except ImportError:
    get_vad_model = None


@dataclass
class ChunkerConfig:
    min_chunk_s: float = 4.0
    target_chunk_s: float = 8.0
    max_chunk_s: float = 20.0
    min_silence_s: float = 0.3
    speech_padding_s: float = 0.2
    vad_threshold: float = 0.5
    energy_threshold: float = 0.01
    sample_rate: int = 16000
    frame_size: int = 512


@dataclass
class AudioChunk:
    seq: int
    t_start: float
    t_end: float
    duration_s: float
    samples: np.ndarray
    file_path: str


class AudioChunker:
    def __init__(self, session_dir: str, config: Optional[ChunkerConfig] = None):
        self.session_dir = session_dir
        self.config = config or ChunkerConfig()
        self.sequence_num = 0

        self.chunks_dir = os.path.join(session_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.jsonl_path = os.path.join(session_dir, "chunks.jsonl")

        # Load Silero VAD model if available
        self._vad = None
        if get_vad_model is not None:
            try:
                self._vad = get_vad_model()
            except Exception:
                pass

        # Internal state
        self._buffer: List[np.ndarray] = []
        self._current_chunk_samples = 0
        self._total_samples_emitted = 0
        self._silence_samples = 0
        self._has_speech = False

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Evaluate whether a 512-sample frame contains speech."""
        # Energy check: pure zeros or near-zero noise is silence
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < self.config.energy_threshold:
            return False

        # If Silero VAD is loaded, run neural model
        if self._vad is not None:
            try:
                prob = float(self._vad(frame)[0])
                return prob >= self.config.vad_threshold or rms >= 0.08
            except Exception:
                pass

        # Fallback to energy threshold
        return rms >= self.config.energy_threshold

    def process_frame(self, frame: np.ndarray) -> Optional[AudioChunk]:
        """Ingest a 512-sample frame, evaluate pause boundaries, and return AudioChunk if cut."""
        if len(frame) != self.config.frame_size:
            # Pad or trim if needed
            if len(frame) < self.config.frame_size:
                frame = np.pad(frame, (0, self.config.frame_size - len(frame)))
            else:
                frame = frame[: self.config.frame_size]

        is_speech = self._is_speech(frame)
        self._buffer.append(frame)
        self._current_chunk_samples += len(frame)

        if is_speech:
            self._has_speech = True
            self._silence_samples = 0
        else:
            self._silence_samples += len(frame)

        cur_duration = self._current_chunk_samples / self.config.sample_rate
        silence_duration = self._silence_samples / self.config.sample_rate

        # ── Cut Decision Rules ────────────────────────────────────────────────
        should_cut = False

        # Rule 1: We hit target duration (e.g. >= 8s) AND found a pause >= 300ms
        if cur_duration >= self.config.target_chunk_s and silence_duration >= self.config.min_silence_s:
            should_cut = True

        # Rule 2: Hard maximum duration exceeded (e.g. >= 20s continuous speech)
        elif cur_duration >= self.config.max_chunk_s:
            should_cut = True

        if should_cut:
            return self._emit_chunk()

        return None

    def flush(self) -> Optional[AudioChunk]:
        """Flush remaining buffered audio on stop."""
        if self._current_chunk_samples > 0:
            return self._emit_chunk()
        return None

    def _emit_chunk(self) -> Optional[AudioChunk]:
        """Finalize and save current chunk if it contains speech, or discard silence."""
        if not self._buffer:
            return None

        samples = np.concatenate(self._buffer)
        duration_s = len(samples) / self.config.sample_rate
        t_start = self._total_samples_emitted / self.config.sample_rate
        t_end = t_start + duration_s

        # Reset buffer state
        self._buffer = []
        self._current_chunk_samples = 0
        self._silence_samples = 0

        # Discard pure non-speech chunks (TRD 2.3)
        if not self._has_speech:
            self._total_samples_emitted += len(samples)
            return None

        self._has_speech = False
        self.sequence_num += 1
        seq = self.sequence_num

        # Save chunk WAV
        chunk_filename = f"chunk_{seq:04d}.wav"
        chunk_path = os.path.join(self.chunks_dir, chunk_filename)
        self._write_chunk_wav(chunk_path, samples)

        # Log audit entry to chunks.jsonl
        rel_audio_path = os.path.join("chunks", chunk_filename).replace("\\", "/")
        audit_entry = {
            "seq": seq,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "duration_s": round(duration_s, 3),
            "samples": len(samples),
            "audio_file": rel_audio_path,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        self._total_samples_emitted += len(samples)

        return AudioChunk(
            seq=seq,
            t_start=t_start,
            t_end=t_end,
            duration_s=duration_s,
            samples=samples,
            file_path=chunk_path,
        )

    def _write_chunk_wav(self, file_path: str, samples: np.ndarray) -> None:
        """Write float32 samples as standard 16-bit PCM WAV."""
        pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
        raw_bytes = pcm16.tobytes()
        byte_rate = self.config.sample_rate * 2
        block_align = 2
        data_size = len(raw_bytes)
        riff_size = 36 + data_size

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            riff_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            1,  # mono
            self.config.sample_rate,
            byte_rate,
            block_align,
            16,
            b"data",
            data_size,
        )
        with open(file_path, "wb") as f:
            f.write(header)
            f.write(raw_bytes)
