"""
Audio Recorder & Live WAV Writer — Voice Brainstorm Pad Iteration 3

Responsibilities:
  - Stream microphone input at 16 kHz mono float32 using sounddevice
  - Live WAV writer that streams PCM16 audio and periodically patches the 44-byte WAV header
  - Lock-free / non-blocking audio callback pushing into a thread-safe Queue
  - Live RMS level meter computation at ~10 Hz
"""

import logging
import math
import os
import queue
import struct
import threading
import time
from typing import Callable, Optional
import numpy as np

logger = logging.getLogger("recorder")

try:
    import sounddevice as sd  # type: ignore
except ImportError:
    sd = None


class LiveWavWriter:
    """
    Writes raw PCM 16-bit audio to disk while periodically patching the
    44-byte WAV header so the file remains valid and playable even during
    recording or after an unexpected process termination (TRD 2.3).
    """

    def __init__(self, file_path: str, sample_rate: int = 16000, channels: int = 1):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_per_sample = 2  # 16-bit PCM
        self.total_samples = 0
        self.samples_since_last_patch = 0

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self.file = open(file_path, "wb+")
        self._write_initial_header()

    def _write_initial_header(self) -> None:
        """Write standard 44-byte RIFF/WAVE header with 0 initial data length."""
        byte_rate = self.sample_rate * self.channels * self.bytes_per_sample
        block_align = self.channels * self.bytes_per_sample

        # 44-byte WAV Header
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36,                  # ChunkSize = 36 + SubChunk2Size
            b"WAVE",
            b"fmt ",
            16,                  # Subchunk1Size (16 for PCM)
            1,                   # AudioFormat (1 = PCM)
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            16,                  # BitsPerSample
            b"data",
            0,                   # Subchunk2Size (data bytes)
        )
        self.file.write(header)
        self.file.flush()

    def patch_header(self) -> None:
        """Update RIFF and data chunk size fields in the 44-byte header."""
        if not self.file or self.file.closed:
            return

        data_size = self.total_samples * self.channels * self.bytes_per_sample
        riff_size = 36 + data_size

        cur_pos = self.file.tell()
        try:
            # Patch RIFF ChunkSize at offset 4
            self.file.seek(4)
            self.file.write(struct.pack("<I", riff_size))

            # Patch data Subchunk2Size at offset 40
            self.file.seek(40)
            self.file.write(struct.pack("<I", data_size))

            self.file.flush()
        finally:
            self.file.seek(cur_pos)

    def write_samples(self, samples: np.ndarray) -> None:
        """Convert float32 [-1.0, 1.0] samples to PCM16 and append to file."""
        if not self.file or self.file.closed:
            return

        # Clip and convert to 16-bit signed integers
        pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
        raw_bytes = pcm16.tobytes()

        self.file.write(raw_bytes)
        num_samples = len(samples)
        self.total_samples += num_samples
        self.samples_since_last_patch += num_samples

        # Patch header every 1 second (16000 samples)
        if self.samples_since_last_patch >= self.sample_rate:
            self.patch_header()
            self.samples_since_last_patch = 0

    def close(self) -> None:
        """Finalize header and close file."""
        if self.file and not self.file.closed:
            self.patch_header()
            self.file.flush()
            self.file.close()
            self.file = None


class AudioRecorder:
    """
    Captures live microphone audio using sounddevice, computes RMS levels at ~10 Hz,
    and feeds samples to both the LiveWavWriter and the AudioChunker.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 512,
        on_level: Optional[Callable[[float], None]] = None,
        on_frame: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.on_level = on_level
        self.on_frame = on_frame

        self.stream: Optional[Any] = None
        self.wav_writer: Optional[LiveWavWriter] = None
        self.is_recording = False
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_level_time = 0.0

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice stream callback — non-blocking, pushes frame to queue."""
        if not self.is_recording:
            return

        # Flatten to 1D float32
        data = indata[:, 0].copy()
        self._queue.put(data)

        # Compute level at ~10 Hz (every 100ms)
        now = time.time()
        if self.on_level and (now - self._last_level_time) >= 0.1:
            self._last_level_time = now
            rms = float(np.sqrt(np.mean(data ** 2)))
            self.on_level(rms)

    def _worker_loop(self):
        """Worker thread drains audio queue, writes to audio.wav and dispatches frames."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self.wav_writer:
                self.wav_writer.write_samples(frame)

            if self.on_frame:
                try:
                    self.on_frame(frame)
                except Exception as e:
                    pass

    def start(self, session_dir: str, device: Optional[int | str] = None) -> str:
        """Start recording into session_dir/audio.wav."""
        if self.is_recording:
            return ""

        wav_path = os.path.join(session_dir, "audio.wav")
        self.wav_writer = LiveWavWriter(wav_path, sample_rate=self.sample_rate, channels=1)

        self.is_recording = True
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if sd is not None:
            dev_idx = None
            if device is not None:
                try:
                    dev_idx = int(device)
                except ValueError:
                    pass

            try:
                self.stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.frame_size,
                    device=dev_idx,
                    callback=self._audio_callback,
                )
                self.stream.start()
            except Exception as e:
                if dev_idx is not None:
                    logger.warning(
                        f"Failed to open audio device {dev_idx} ({e}), falling back to system default"
                    )
                    self.stream = sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                        blocksize=self.frame_size,
                        device=None,
                        callback=self._audio_callback,
                    )
                    self.stream.start()
                else:
                    self.stop()
                    raise

        return wav_path

    def stop(self) -> None:
        """Stop recording and finalize files."""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        if self.wav_writer:
            self.wav_writer.close()
            self.wav_writer = None
