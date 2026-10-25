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
    max_chunk_s: float = 15.0       # Progressive cut limit (reduced from 20s to meet 10s latency target)
    min_silence_s: float = 0.3      # 300ms pause target
    pause_cut_s: float = 2.5        # Long pause cut threshold (matches paragraph break threshold)
    speech_padding_s: float = 0.2
    vad_threshold: float = 0.5
    energy_threshold: float = 0.003  # Preserves quiet speech from laptop microphones
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
    def __init__(
        self,
        session_dir: str,
        config: Optional[ChunkerConfig] = None,
        vad_fn: Optional[Any] = None,
    ):
        self.session_dir = session_dir
        self.config = config or ChunkerConfig()
        self.chunks_dir = os.path.join(session_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)
        self.jsonl_path = os.path.join(session_dir, "chunks.jsonl")

        # Resume sequence numbering from existing chunks.jsonl
        self.sequence_num = 0
        if os.path.exists(self.jsonl_path):
            try:
                with open(self.jsonl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.sequence_num = max(self.sequence_num, data.get("seq", 0))
            except Exception:
                pass

        # Resume sample offset from audio.wav: always byte size calculation, never wave.getnframes()
        self._total_samples_emitted = 0
        audio_wav_path = os.path.join(session_dir, "audio.wav")
        if os.path.exists(audio_wav_path):
            size = os.path.getsize(audio_wav_path)
            self._total_samples_emitted = max(0, (size - 44) // 2)

        # Load Silero VAD model or use injected vad_fn
        self._vad = vad_fn
        if self._vad is None and get_vad_model is not None:
            try:
                self._vad = get_vad_model()
            except Exception:
                pass

        # Internal state
        self._buffer: List[np.ndarray] = []
        self._buffer_is_speech: List[bool] = []
        self._current_chunk_samples = 0
        self._silence_samples = 0
        self._has_speech = False

    def _is_speech(self, frame: np.ndarray) -> bool:
        """Evaluate whether a 512-sample frame contains speech."""
        if self._vad is not None:
            try:
                res = self._vad(frame)
                prob = float(res[0]) if hasattr(res, "__getitem__") else float(res)
                return prob >= self.config.vad_threshold
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms >= self.config.energy_threshold

    def process_frame(self, frame: np.ndarray) -> Optional[AudioChunk]:
        """Ingest a 512-sample frame, evaluate progressive pauses, and return AudioChunk if cut."""
        if len(frame) != self.config.frame_size:
            if len(frame) < self.config.frame_size:
                frame = np.pad(frame, (0, self.config.frame_size - len(frame)))
            else:
                frame = frame[: self.config.frame_size]

        is_speech = self._is_speech(frame)
        self._buffer.append(frame)
        self._buffer_is_speech.append(is_speech)
        self._current_chunk_samples += len(frame)

        if is_speech:
            self._has_speech = True
            self._silence_samples = 0
        else:
            self._silence_samples += len(frame)

        cur_duration = self._current_chunk_samples / self.config.sample_rate
        silence_duration = self._silence_samples / self.config.sample_rate

        # ── Cut Decision Rules ──
        # Long-pause rule: if trailing silence >= pause_cut_s (default 2.5s) and speech >= 1.0s, cut immediately
        if silence_duration >= self.config.pause_cut_s:
            speech_frames = sum(1 for flag in self._buffer_is_speech if flag)
            speech_s = speech_frames * self.config.frame_size / self.config.sample_rate
            if speech_s >= 1.0:
                return self._emit_chunk()
            else:
                # Discard as noise: clear buffer and advance timeline
                self._total_samples_emitted += self._current_chunk_samples
                self._buffer = []
                self._buffer_is_speech = []
                self._current_chunk_samples = 0
                self._silence_samples = 0
                self._has_speech = False
                return None

        # Progressive pauses: shorter pauses are accepted as chunk grows longer
        if cur_duration >= 12.0 and silence_duration >= 0.08:
            return self._emit_chunk()
        elif cur_duration >= 10.0 and silence_duration >= 0.15:
            return self._emit_chunk()
        elif cur_duration >= self.config.target_chunk_s and silence_duration >= self.config.min_silence_s:
            return self._emit_chunk()

        # Hard limit reached (~15s continuous speech): force cut at quietest frame in last 3s
        elif cur_duration >= self.config.max_chunk_s:
            return self._emit_forced_cut(search_window_s=3.0)

        return None

    def flush(self) -> Optional[AudioChunk]:
        """Flush remaining buffered audio on stop."""
        if self._current_chunk_samples > 0:
            return self._emit_chunk()
        return None

    def _emit_forced_cut(self, search_window_s: float = 3.0) -> Optional[AudioChunk]:
        """Find the quietest frame in the search window, split buffer, and carry remainder forward."""
        if not self._buffer:
            return None

        frames_in_search = int(search_window_s * self.config.sample_rate / self.config.frame_size)
        window_size = min(len(self._buffer), max(1, frames_in_search))
        search_window = self._buffer[-window_size:]

        # Find frame with minimal RMS energy
        min_idx = min(range(len(search_window)), key=lambda i: float(np.mean(search_window[i] ** 2)))
        cut_point = len(self._buffer) - window_size + min_idx + 1

        chunk_frames = self._buffer[:cut_point]
        chunk_flags = self._buffer_is_speech[:cut_point]
        remainder_frames = self._buffer[cut_point:]
        remainder_flags = self._buffer_is_speech[cut_point:]

        return self._emit_frames(chunk_frames, remainder_frames, chunk_flags, remainder_flags)

    def _emit_chunk(self) -> Optional[AudioChunk]:
        """Finalize and save all currently buffered audio as a chunk."""
        frames = self._buffer
        flags = self._buffer_is_speech
        self._buffer = []
        self._buffer_is_speech = []
        return self._emit_frames(frames, remainder_frames=[], chunk_flags=flags, remainder_flags=[])

    def _emit_frames(
        self,
        chunk_frames: List[np.ndarray],
        remainder_frames: List[np.ndarray],
        chunk_flags: List[bool],
        remainder_flags: List[bool],
    ) -> Optional[AudioChunk]:
        if not chunk_frames:
            self._buffer = remainder_frames
            self._buffer_is_speech = remainder_flags
            return None

        samples = np.concatenate(chunk_frames)
        duration_s = len(samples) / self.config.sample_rate
        t_start = self._total_samples_emitted / self.config.sample_rate
        t_end = t_start + duration_s

        # Reset buffer to the remainder
        self._buffer = remainder_frames
        self._buffer_is_speech = remainder_flags
        self._current_chunk_samples = sum(len(f) for f in remainder_frames)

        # Compute silence_samples for the carried-over remainder from flags
        trailing_silence_frames = 0
        for flag in reversed(remainder_flags):
            if not flag:
                trailing_silence_frames += 1
            else:
                break
        self._silence_samples = trailing_silence_frames * self.config.frame_size

        chunk_has_speech = any(chunk_flags) if chunk_flags else self._has_speech
        # Compute _has_speech directly from the remainder frames' cached flags
        self._has_speech = any(remainder_flags) if remainder_flags else False

        # Discard pure non-speech chunks (TRD 2.3)
        if not chunk_has_speech:
            self._total_samples_emitted += len(samples)
            return None
        self.sequence_num += 1
        seq = self.sequence_num

        # Save chunk WAV
        chunk_filename = f"chunk_{seq:04d}.wav"
        chunk_path = os.path.join(self.chunks_dir, chunk_filename)
        self._write_chunk_wav(chunk_path, samples)

        # Log audit entry to strictly append-only chunks.jsonl (Defect 1 fix)
        rel_audio_path = os.path.join("chunks", chunk_filename).replace("\\", "/")
        import time as _t
        audit_entry = {
            "seq": seq,
            "t_start": round(t_start, 3),
            "t_end": round(t_end, 3),
            "duration_s": round(duration_s, 3),
            "samples": len(samples),
            "audio_file": rel_audio_path,
            "timestamp": _t.time(),
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
            f.flush()

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
