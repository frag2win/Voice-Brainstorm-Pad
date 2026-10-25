"""
Unit tests for Chunker & VAD — Voice Brainstorm Pad Iteration 4.2

Tests:
  - Pause cut at 8s with 300ms pause (using injected fake VAD)
  - Progressive cuts at 10s (150ms pause) and 12s (80ms pause)
  - Forced cut at the quietest frame with remainder carried into next chunk
  - Authoritative resume offset from byte size after crash-truncated WAV header
  - VAD called exactly once per frame (using call counter)
  - Pure silence never emits chunks
  - Chunks written to disk with valid WAV headers and logged to chunks.jsonl
"""

import json
import os
import shutil
import struct
import tempfile
import wave
import numpy as np
import pytest

from sidecar.chunker import AudioChunker, ChunkerConfig


@pytest.fixture
def temp_session_dir():
    dirpath = tempfile.mkdtemp(prefix="vbp_test_session_")
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


class ScriptedVAD:
    """Fake VAD returning scripted speech probabilities per frame with a call counter."""
    def __init__(self, default_prob: float = 0.0):
        self.default_prob = default_prob
        self.call_count = 0
        self._ranges: list[tuple[int, int, float]] = []

    def add_range(self, start_frame: int, end_frame: int, prob: float):
        self._ranges.append((start_frame, end_frame, prob))

    def __call__(self, frame: np.ndarray) -> list[float]:
        idx = self.call_count
        self.call_count += 1
        for start, end, prob in self._ranges:
            if start <= idx < end:
                return [prob]
        return [self.default_prob]


# Alias for test compatibility
MockVad = ScriptedVAD


def make_frame(amplitude: float = 0.5, frame_size: int = 512) -> np.ndarray:
    """Create a constant or tone frame for testing."""
    return np.full(frame_size, amplitude, dtype=np.float32)


def test_chunker_initialization(temp_session_dir):
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=20.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)
    assert chunker.config.min_chunk_s == 4.0
    assert chunker.config.target_chunk_s == 8.0
    assert chunker.config.max_chunk_s == 20.0
    assert chunker.sequence_num == 0


def test_silence_produces_no_chunks(temp_session_dir):
    vad = ScriptedVAD(default_prob=0.0)
    chunker = AudioChunker(session_dir=temp_session_dir, vad_fn=vad)

    frame_size = 512
    silence_frame = np.zeros(frame_size, dtype=np.float32)
    total_frames = int(15.0 * 16000 / frame_size)

    emitted = []
    for _ in range(total_frames):
        chunk = chunker.process_frame(silence_frame)
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 0, "Silence should never produce chunks"


def test_pause_cut_at_8s_with_300ms_pause(temp_session_dir):
    """
    Pause cut at 8s with a 300ms pause:
    8.0s speech (250 frames) followed by 10 frames (320ms) of pause.
    Must emit chunk at ~8.32s with seq 1.
    """
    vad = ScriptedVAD(default_prob=0.9)  # Default speech
    # 250 frames speech (0 to 8.0s), then 10 frames silence (8.0 to 8.32s)
    vad.add_range(250, 260, 0.1)

    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    emitted = []
    # Feed 300 frames total (9.6s)
    for _ in range(300):
        chunk = chunker.process_frame(make_frame(amplitude=0.3))
        if chunk:
            emitted.append(chunk)

    assert len(emitted) >= 1
    first = emitted[0]
    assert first.seq == 1
    assert 8.0 <= first.duration_s <= 8.5
    assert first.t_start == 0.0


def test_progressive_cuts_at_10s_and_12s(temp_session_dir):
    """
    Progressive cuts:
    1. At 10s, a shorter pause (160ms >= 150ms) triggers cut even though < 300ms.
    2. At 12s, an even shorter pause (96ms >= 80ms) triggers cut even though < 150ms.
    """
    # Test 10s progressive cut
    vad_10 = ScriptedVAD(default_prob=0.9)
    # 313 frames = 10.016s of speech, then 5 frames = 160ms pause
    vad_10.add_range(313, 318, 0.1)

    chunker_10 = AudioChunker(
        session_dir=temp_session_dir,
        config=ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3),
        vad_fn=vad_10,
    )

    emitted_10 = []
    for _ in range(340):
        chunk = chunker_10.process_frame(make_frame(amplitude=0.3))
        if chunk:
            emitted_10.append(chunk)

    assert len(emitted_10) >= 1
    assert 10.0 <= emitted_10[0].duration_s <= 10.5

    # Test 12s progressive cut in a new dir
    dir_12 = tempfile.mkdtemp(prefix="vbp_test_12_")
    try:
        vad_12 = ScriptedVAD(default_prob=0.9)
        # 375 frames = 12.0s of speech, then 3 frames = 96ms pause
        vad_12.add_range(375, 378, 0.1)

        chunker_12 = AudioChunker(
            session_dir=dir_12,
            config=ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3),
            vad_fn=vad_12,
        )

        emitted_12 = []
        for _ in range(400):
            chunk = chunker_12.process_frame(make_frame(amplitude=0.3))
            if chunk:
                emitted_12.append(chunk)

        assert len(emitted_12) >= 1
        assert 12.0 <= emitted_12[0].duration_s <= 12.5
    finally:
        shutil.rmtree(dir_12, ignore_errors=True)


def test_forced_cut_at_quietest_frame_with_remainder(temp_session_dir):
    """
    Continuous speech past max duration (15.0s):
    Forced cut occurs at the quietest frame in the last 3.0s window.
    The remainder frames must be carried into the next chunk.
    """
    vad = ScriptedVAD(default_prob=0.9)  # Always speech, no pause cut
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    # Total 500 frames = 16.0s
    # Frame 422 is at ~13.5s (within the 12s-15s window). Make it very quiet.
    quiet_frame_idx = 422

    emitted = []
    for i in range(500):
        amp = 0.0001 if i == quiet_frame_idx else 0.4
        frame = make_frame(amplitude=amp)
        chunk = chunker.process_frame(frame)
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 2, f"Expected 2 chunks, got {len(emitted)}"
    chunk1, chunk2 = emitted[0], emitted[1]

    # Chunk 1 cut at quietest frame (~13.5s)
    expected_cut_s = (quiet_frame_idx + 1) * 512 / 16000
    assert abs(chunk1.duration_s - expected_cut_s) < 0.1
    assert chunk1.seq == 1

    # Chunk 2 has remainder + extra frames (~2.5s)
    assert abs(chunk2.t_start - chunk1.t_end) < 1e-4
    assert chunk2.seq == 2
    assert chunk2.duration_s > 2.0


def test_resume_offset_after_crash_truncated_header(temp_session_dir):
    """
    Crash recovery:
    audio.wav exists with a 44-byte unpatched header (wave.getnframes() returns 0),
    but 64,000 raw PCM bytes (32,000 samples = 2.0s) were written before crash.
    Chunker resume offset must ALWAYS be (os.path.getsize - 44) // 2 = 32000 samples (2.0s).
    """
    audio_wav_path = os.path.join(temp_session_dir, "audio.wav")
    raw_pcm = np.zeros(32000, dtype=np.int16).tobytes()

    # Create unfinalized WAV header with 0 data length
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36,  # Stale/unpatched riff size
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        16000,
        32000,
        2,
        16,
        b"data",
        0,   # Stale data size
    )
    with open(audio_wav_path, "wb") as f:
        f.write(header)
        f.write(raw_pcm)

    # Verify wave.open sees 0 frames due to truncated header
    with wave.open(audio_wav_path, "rb") as wf:
        assert wf.getnframes() == 0

    # AudioChunker must use byte-level offset, recovering all 32,000 samples
    chunker = AudioChunker(session_dir=temp_session_dir)
    assert chunker._total_samples_emitted == 32000


def test_vad_called_exactly_once_per_frame(temp_session_dir):
    """
    Verify Silero VAD is never called twice on any frame.
    Across forced cuts, remainders, and flushes, call count must equal frame count.
    """
    vad = ScriptedVAD(default_prob=0.9)
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    total_frames = 600  # 19.2s -> triggers forced cut at 15s and carries remainder
    for i in range(total_frames):
        amp = 0.0001 if i == 420 else 0.3
        chunker.process_frame(make_frame(amplitude=amp))

    chunker.flush()

    assert vad.call_count == total_frames, (
        f"Expected VAD called exactly {total_frames} times (once per frame), "
        f"but was called {vad.call_count} times."
    )


def test_chunk_files_and_jsonl_written_to_disk(temp_session_dir):
    vad = ScriptedVAD(default_prob=0.9)
    vad.add_range(125, 140, 0.1)  # 4.0s speech, then pause

    config = ChunkerConfig(min_chunk_s=2.0, target_chunk_s=3.0, max_chunk_s=10.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    for _ in range(150):
        chunker.process_frame(make_frame(amplitude=0.3))
    chunker.flush()

    chunks_dir = os.path.join(temp_session_dir, "chunks")
    assert os.path.isdir(chunks_dir)

    wav_path = os.path.join(chunks_dir, "chunk_0001.wav")
    assert os.path.isfile(wav_path)
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0

    jsonl_path = os.path.join(temp_session_dir, "chunks.jsonl")
    assert os.path.isfile(jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 1
    assert lines[0]["seq"] == 1
    assert "t_start" in lines[0]
    assert "t_end" in lines[0]
    assert "audio_file" in lines[0]


def test_long_pause_cut_with_sufficient_speech(temp_session_dir):
    """
    5 s speech + 3 s silence -> exactly one chunk cut at the pause.
    Frame size = 512 samples (32ms).
    5.0s speech = 157 frames.
    3.0s silence = 94 frames.
    """
    vad = MockVad(default_prob=0.0)
    # Frames 0..157 are speech (0.9 probability)
    vad.add_range(0, 157, 0.9)

    config = ChunkerConfig(pause_cut_s=2.5)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    emitted = []
    # Feed 157 speech frames (5.024s) + 94 silence frames (3.008s) = 251 frames
    for _ in range(251):
        chunk = chunker.process_frame(make_frame(amplitude=0.3))
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 1, f"Expected exactly one chunk cut at pause, got {len(emitted)}"
    assert emitted[0].seq == 1
    # Chunk should contain speech + pause cut (around 7.5s)
    assert 7.0 <= emitted[0].duration_s <= 8.0


def test_long_pause_with_insufficient_speech_discards_as_noise(temp_session_dir):
    """
    0.5 s speech + 3 s silence -> no cut (discard as noise).
    Frame size = 512 samples (32ms).
    0.5s speech = 16 frames.
    3.0s silence = 94 frames.
    """
    vad = MockVad(default_prob=0.0)
    # Frames 0..16 are speech (0.9 probability)
    vad.add_range(0, 16, 0.9)

    config = ChunkerConfig(pause_cut_s=2.5)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config, vad_fn=vad)

    emitted = []
    # Feed 16 speech frames (0.512s) + 94 silence frames (3.008s) = 110 frames
    for _ in range(110):
        chunk = chunker.process_frame(make_frame(amplitude=0.3))
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 0, f"Expected 0 chunks (discarded as noise), got {len(emitted)}"

