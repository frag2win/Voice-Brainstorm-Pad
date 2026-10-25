"""
Unit tests for Chunker & VAD — Voice Brainstorm Pad Iteration 3

Tests:
  - Chunker initialization with TRD parameters (min 4s, target 8-12s, max 20s, min silence 0.3s)
  - Synthetic audio with known pauses: cuts land at silence boundaries
  - Continuous speech without pause: forces a cut at max duration (20s)
  - Short audio followed by pause: accumulates until min duration (4s) before cutting
  - Pure silence: never emits chunks
  - Chunks written to disk with valid WAV headers and logged to chunks.jsonl
"""

import json
import math
import os
import shutil
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


def make_sine_wave(freq_hz: float, duration_s: float, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure tone representing speech energy."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def make_silence(duration_s: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def test_chunker_initialization(temp_session_dir):
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=20.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)
    assert chunker.config.min_chunk_s == 4.0
    assert chunker.config.target_chunk_s == 8.0
    assert chunker.config.max_chunk_s == 20.0
    assert chunker.sequence_num == 0


def test_silence_produces_no_chunks(temp_session_dir):
    chunker = AudioChunker(session_dir=temp_session_dir)
    # Feed 15 seconds of silence in 32ms frames (512 samples at 16kHz)
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


def test_cuts_land_at_pause_after_target_duration(temp_session_dir):
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=20.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)

    sample_rate = 16000
    frame_size = 512

    # Construct audio:
    # 0s - 9s: Speech (9s total, passes target of 8s)
    # 9s - 10s: Silence pause (1s, exceeds 0.3s silence threshold)
    # 10s - 19s: Speech (9s)
    speech1 = make_sine_wave(300, 9.0, sample_rate)
    silence = make_silence(1.0, sample_rate)
    speech2 = make_sine_wave(300, 9.0, sample_rate)

    full_audio = np.concatenate([speech1, silence, speech2])
    emitted = []

    # Stream in 512-sample frames
    for i in range(0, len(full_audio) - frame_size + 1, frame_size):
        frame = full_audio[i : i + frame_size]
        chunk = chunker.process_frame(frame)
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) >= 2, f"Expected at least 2 chunks, got {len(emitted)}"
    # First chunk should cut at or right after the pause (approx 9.0s - 9.5s)
    first = emitted[0]
    assert 8.0 <= first.duration_s <= 10.5, f"First chunk duration {first.duration_s}s not in expected range"
    assert first.seq == 1


def test_continuous_speech_forced_cut_at_max_duration(temp_session_dir):
    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)

    sample_rate = 16000
    frame_size = 512

    # Construct 18 seconds of continuous speech with NO pauses
    speech = make_sine_wave(300, 18.0, sample_rate)
    emitted = []

    for i in range(0, len(speech) - frame_size + 1, frame_size):
        frame = speech[i : i + frame_size]
        chunk = chunker.process_frame(frame)
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) >= 2, "Continuous speech past max duration must be cut"
    first = emitted[0]
    assert first.duration_s <= config.max_chunk_s + 0.5


def test_chunk_files_and_jsonl_written_to_disk(temp_session_dir):
    config = ChunkerConfig(min_chunk_s=2.0, target_chunk_s=3.0, max_chunk_s=10.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)

    sample_rate = 16000
    frame_size = 512

    speech = make_sine_wave(300, 4.0, sample_rate)
    silence = make_silence(0.5, sample_rate)
    full = np.concatenate([speech, silence])

    for i in range(0, len(full) - frame_size + 1, frame_size):
        chunker.process_frame(full[i : i + frame_size])
    chunker.flush()

    # Verify chunks directory exists
    chunks_dir = os.path.join(temp_session_dir, "chunks")
    assert os.path.isdir(chunks_dir)

    # Check WAV file
    wav_path = os.path.join(chunks_dir, "chunk_0001.wav")
    assert os.path.isfile(wav_path)
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit PCM
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0

    # Check chunks.jsonl
    jsonl_path = os.path.join(temp_session_dir, "chunks.jsonl")
    assert os.path.isfile(jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) >= 1
    assert lines[0]["seq"] == 1
    assert "t_start" in lines[0]
    assert "t_end" in lines[0]
    assert "audio_file" in lines[0]


def test_chunker_resumed_session_preserves_audio_wav_offset(temp_session_dir):
    """
    Defect 2 fix validation:
    When a session already has audio.wav (e.g. 5.0 seconds from previous dictation),
    resuming must initialize t_start from audio.wav (5.0s), never reset to 0.0.
    """
    sample_rate = 16000
    frame_size = 512

    # 1. Create pre-existing audio.wav with exactly 5.0s (80,000 samples)
    audio_wav_path = os.path.join(temp_session_dir, "audio.wav")
    existing_samples = 5 * sample_rate
    pcm16 = np.zeros(existing_samples, dtype=np.int16)
    with wave.open(audio_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    # Pre-create chunks.jsonl with a chunk from the previous run
    jsonl_path = os.path.join(temp_session_dir, "chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "seq": 1,
            "t_start": 0.0,
            "t_end": 4.5,
            "duration_s": 4.5,
            "samples": int(4.5 * sample_rate),
            "audio_file": "chunks/chunk_0001.wav",
        }) + "\n")

    # 2. Start new chunker on resumed session
    config = ChunkerConfig(min_chunk_s=3.0, target_chunk_s=4.0, max_chunk_s=10.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)

    # Verify sequence starts at 2 and sample offset starts at 5.0s (80,000 samples)
    assert chunker.sequence_num == 1
    assert chunker._total_samples_emitted == existing_samples

    # 3. Stream 5s of speech and 0.5s pause
    speech = make_sine_wave(300, 5.0, sample_rate)
    silence = make_silence(0.5, sample_rate)
    audio = np.concatenate([speech, silence])

    emitted = []
    for i in range(0, len(audio) - frame_size + 1, frame_size):
        chunk = chunker.process_frame(audio[i : i + frame_size])
        if chunk:
            emitted.append(chunk)

    assert len(emitted) >= 1
    first_resumed = emitted[0]
    assert first_resumed.seq == 2
    # CRITICAL: t_start must be 5.0s (from audio.wav), NOT 0.0!
    assert round(first_resumed.t_start, 2) == 5.0
    assert first_resumed.t_end > 5.0


def test_progressive_pauses_and_quietest_frame_cut(temp_session_dir):
    """
    Defect 4 fix validation:
    At 15.0s continuous speech, chunker must force-cut at the quietest frame
    in the last 3s and carry remainder forward into buffer.
    """
    sample_rate = 16000
    frame_size = 512

    config = ChunkerConfig(min_chunk_s=4.0, target_chunk_s=8.0, max_chunk_s=15.0, min_silence_s=0.3)
    chunker = AudioChunker(session_dir=temp_session_dir, config=config)
    # Override with deterministic speech detector so continuous synthetic audio isn't flagged by neural VAD hum-adaptation
    chunker._is_speech = lambda frame: float(np.sqrt(np.mean(frame ** 2))) > 0.02

    # Construct 16 seconds of continuous speech
    # Insert an intentional quiet dip (energy 0.005) at 13.5s
    full_audio = make_sine_wave(300, 16.0, sample_rate)
    dip_start = int(13.5 * sample_rate)
    dip_end = int(13.6 * sample_rate)
    full_audio[dip_start:dip_end] *= 0.01

    emitted = []
    for i in range(0, len(full_audio) - frame_size + 1, frame_size):
        chunk = chunker.process_frame(full_audio[i : i + frame_size])
        if chunk:
            emitted.append(chunk)

    final = chunker.flush()
    if final:
        emitted.append(final)

    assert len(emitted) == 2, f"Expected 2 chunks, got {len(emitted)}"
    # First chunk cut near 13.5s (quietest frame in last 3s)
    assert 13.0 <= emitted[0].duration_s <= 14.5
    # Second chunk contains remaining audio (~2.5s)
    assert 1.5 <= emitted[1].duration_s <= 3.0

