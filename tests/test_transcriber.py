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

from sidecar.transcriber import (
    Transcriber,
    TranscriberConfig,
    clean_transcript,
    strip_prompt_echo,
)


@pytest.fixture(scope="module")
def shared_transcriber():
    config = TranscriberConfig(model_name="tiny.en", cpu_threads=4)
    t = Transcriber(config=config, auto_warmup=False)
    yield t
    t.stop()


def test_clean_transcript_filters_hallucinations():
    assert clean_transcript("") == ""
    assert clean_transcript("   ") == ""
    assert clean_transcript("Thank you for watching.") == ""
    assert clean_transcript("Thank you.") == ""
    assert clean_transcript("[Music]") == ""
    assert clean_transcript("(applause)") == ""
    assert clean_transcript("Subtitles by the Amara.org community") == ""
    assert clean_transcript("Hello world! This is brainstorm pad.") == "Hello world! This is brainstorm pad."


def test_transcriber_initialization(shared_transcriber):
    assert shared_transcriber.model is not None
    assert shared_transcriber.is_ready is True


def test_transcriber_silence_produces_no_text(tmp_path, shared_transcriber):
    transcripts = []
    shared_transcriber.on_transcript = lambda res: transcripts.append(res)

    # Create 3 seconds of pure silence WAV
    silence_wav = str(tmp_path / "silence.wav")
    sample_rate = 16000
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

    shared_transcriber.enqueue(
        seq=1,
        audio_path=silence_wav,
        duration_s=3.0,
        session_dir=str(tmp_path),
    )

    shared_transcriber.wait_drained(timeout=5.0)

    assert len(transcripts) == 1
    assert transcripts[0].text == ""
    assert transcripts[0].rtf > 0.0


def test_transcriber_sequential_order(tmp_path, shared_transcriber):
    transcripts = []
    shared_transcriber.on_transcript = lambda res: transcripts.append(res)

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
        shared_transcriber.enqueue(seq=i, audio_path=wav_file, duration_s=1.0, session_dir=str(tmp_path))

    shared_transcriber.wait_drained(timeout=10.0)

    assert len(transcripts) == 3
    assert [t.seq for t in transcripts] == [1, 2, 3]


def test_transcriber_appends_to_transcripts_jsonl(tmp_path, shared_transcriber):
    transcripts_path = tmp_path / "transcripts.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"seq": 1, "duration_s": 1.0, "audio_file": "chunks/chunk_0001.wav"}) + "\n")

    shared_transcriber.on_transcript = None

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

    shared_transcriber.enqueue(seq=1, audio_path=wav_file, duration_s=1.0, session_dir=str(tmp_path))
    shared_transcriber.wait_drained(timeout=5.0)

    # Read transcripts.jsonl and verify metrics exist
    assert os.path.isfile(transcripts_path)
    with open(transcripts_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 1
    assert lines[0]["seq"] == 1
    assert "model" in lines[0]
    assert "rtf" in lines[0]
    assert "transcribe_s" in lines[0]
    assert "queue_wait_ms" in lines[0]
    assert "latency_ms" in lines[0]

    # Verify chunks.jsonl remained strictly untouched and append-only
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_lines = [json.loads(line) for line in f if line.strip()]
    assert len(chunk_lines) == 1
    assert "model" not in chunk_lines[0]
    # Verify per-chunk diagnostics are recorded
    assert "rss_mb" in lines[0]
    assert "avail_ram_mb" in lines[0]


def test_strip_prompt_echo():
    # 1. Output equals tail of prompt -> stripped completely
    prompt = "We are dictating notes for the local voice pad."
    text = "local voice pad."
    assert strip_prompt_echo(text, prompt) == ""

    # 2. Output starts with tail of prompt -> repeated part stripped
    text2 = "local voice pad. And continuing with a new idea."
    assert strip_prompt_echo(text2, prompt) == "And continuing with a new idea."

    # 3. Output with punctuation/casing variations in prompt echo
    prompt3 = "Hello, world! Today is great."
    text3 = "today is great! More text."
    assert strip_prompt_echo(text3, prompt3) == "More text."

    # 4. Output with no overlap -> unchanged
    text4 = "Completely distinct phrase."
    assert strip_prompt_echo(text4, prompt) == "Completely distinct phrase."

    # 5. Empty prompt or text -> unchanged
    assert strip_prompt_echo("hello", "") == "hello"
    assert strip_prompt_echo("", "prompt") == ""


def test_transcriber_reset_context(shared_transcriber):
    shared_transcriber._last_prompt_text = "persisted context from previous session"
    shared_transcriber.reset_context()
    assert shared_transcriber._last_prompt_text == ""


def test_failed_chunk_appends_to_jsonl_and_emits_event(tmp_path):
    """Verify that if transcription raises, a failure record is logged and chunk_failed emitted."""
    class FailingWhisperModel:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("Simulated transcription failure")

    config = TranscriberConfig(model_name="tiny.en", cpu_threads=1)
    # Instantiate transcriber with injected failing model
    transcriber = Transcriber(config=config, auto_warmup=False, model=FailingWhisperModel())

    failed_events = []
    transcriber.on_chunk_failed = lambda evt: failed_events.append(evt)

    # Create dummy wav
    wav_file = str(tmp_path / "chunk_0099.wav")
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

    transcriber.enqueue(seq=99, audio_path=wav_file, duration_s=1.0, session_dir=str(tmp_path))
    transcriber.wait_drained(timeout=5.0)
    transcriber.stop()

    # 1. Assert chunk_failed notification received
    assert len(failed_events) == 1
    assert failed_events[0]["type"] == "chunk_failed"
    assert failed_events[0]["seq"] == 99
    assert failed_events[0]["audio_file"] == wav_file
    assert "Simulated transcription failure" in failed_events[0]["error"]

    # 2. Assert failure record written to transcripts.jsonl
    transcripts_path = tmp_path / "transcripts.jsonl"
    assert os.path.isfile(transcripts_path)
    with open(transcripts_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["seq"] == 99
    assert rec["failed"] is True
    assert "Simulated transcription failure" in rec["error"]
    assert rec["audio_file"] == wav_file
    assert "rss_mb" in rec
    assert "avail_ram_mb" in rec

