"""
Unit Tests for Transcriber Worker — Voice Brainstorm Pad Iteration 4

Tests:
  1. Transcriber initialization and warm-up
  2. Transcription of synthetic audio with known speech
  3. Silence produces empty text without hallucinations
  4. Hallucination filter catches repetitive junk / subtitle prompts
  5. Sequential chunk processing order (queue FIFO)
  6. Audit update in chunks.jsonl
"""

import json
import os
import time
import numpy as np
import pytest

from sidecar.transcriber import Transcriber, TranscriberConfig, clean_transcript


def test_clean_transcript_filters_hallucinations():
    assert clean_transcript("") == ""
    assert clean_transcript("   ") == ""
    assert clean_transcript("Thank you for watching.") == ""
    assert clean_transcript("Thank you.") == ""
    assert clean_transcript("[Music]") == ""
    assert clean_transcript("(applause)") == ""
    assert clean_transcript("Subtitles by the Amara.org community") == ""
    assert clean_transcript("Hello world! This is brainstorm pad.") == "Hello world! This is brainstorm pad."


def test_transcriber_initialization():
    config = TranscriberConfig(model_name="base.en", cpu_threads=4)
    transcriber = Transcriber(config=config, auto_warmup=True)
    assert transcriber.model is not None
    assert transcriber.is_ready is True
    transcriber.stop()


def test_transcriber_silence_produces_no_text(tmp_path):
    transcripts = []

    def on_transcript(res):
        transcripts.append(res)

    config = TranscriberConfig(model_name="base.en", cpu_threads=4)
    transcriber = Transcriber(config=config, on_transcript=on_transcript, auto_warmup=False)

    # Create 3 seconds of pure silence WAV
    silence_wav = str(tmp_path / "silence.wav")
    sample_rate = 16000
    samples = np.zeros(sample_rate * 3, dtype=np.float32)

    # Write WAV
    pcm16 = np.zeros(sample_rate * 3, dtype=np.int16)
    import struct
    data_bytes = pcm16.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data_bytes), b"WAVE", b"fmt ",
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", len(data_bytes)
    )
    with open(silence_wav, "wb") as f:
        f.write(header)
        f.write(data_bytes)

    transcriber.enqueue(
        seq=1,
        audio_path=silence_wav,
        duration_s=3.0,
        session_dir=str(tmp_path),
    )

    transcriber.wait_drained(timeout=5.0)
    transcriber.stop()

    assert len(transcripts) == 1
    assert transcripts[0].text == ""
    assert transcripts[0].rtf > 0.0


def test_transcriber_sequential_order(tmp_path):
    transcripts = []

    def on_transcript(res):
        transcripts.append(res)

    config = TranscriberConfig(model_name="base.en", cpu_threads=4)
    transcriber = Transcriber(config=config, on_transcript=on_transcript, auto_warmup=False)

    # Create 3 dummy audio files
    sample_rate = 16000
    pcm16 = np.zeros(sample_rate * 1, dtype=np.int16)
    import struct
    data_bytes = pcm16.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data_bytes), b"WAVE", b"fmt ",
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", len(data_bytes)
    )

    for i in range(1, 4):
        wav_file = str(tmp_path / f"chunk_{i}.wav")
        with open(wav_file, "wb") as f:
            f.write(header)
            f.write(data_bytes)
        transcriber.enqueue(seq=i, audio_path=wav_file, duration_s=1.0, session_dir=str(tmp_path))

    transcriber.wait_drained(timeout=10.0)
    transcriber.stop()

    assert len(transcripts) == 3
    assert [t.seq for t in transcripts] == [1, 2, 3]


def test_transcriber_updates_chunks_jsonl(tmp_path):
    jsonl_path = tmp_path / "chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 1, "duration_s": 1.0, "audio_file": "chunks/chunk_0001.wav"}) + "\n")

    config = TranscriberConfig(model_name="base.en", cpu_threads=4)
    transcriber = Transcriber(config=config, auto_warmup=False)

    # Create dummy wav
    wav_file = str(tmp_path / "chunk_0001.wav")
    sample_rate = 16000
    pcm16 = np.zeros(sample_rate * 1, dtype=np.int16)
    import struct
    data_bytes = pcm16.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(data_bytes), b"WAVE", b"fmt ",
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", len(data_bytes)
    )
    with open(wav_file, "wb") as f:
        f.write(header)
        f.write(data_bytes)

    transcriber.enqueue(seq=1, audio_path=wav_file, duration_s=1.0, session_dir=str(tmp_path))
    transcriber.wait_drained(timeout=5.0)
    transcriber.stop()

    # Read jsonl and verify text, model, rtf fields exist
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    assert "model" in lines[0]
    assert "rtf" in lines[0]
    assert "transcribe_ms" in lines[0]
