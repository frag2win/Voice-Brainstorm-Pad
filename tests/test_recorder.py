"""
Unit tests for Live WAV Writer — Voice Brainstorm Pad Iteration 3

Tests:
  - Live writing of PCM16 audio frames to audio.wav
  - Dynamic 44-byte header patching so WAV is playable at all times
  - Data integrity and frame count matching written samples
"""

import os
import shutil
import tempfile
import wave
import numpy as np
import pytest

from sidecar.recorder import LiveWavWriter


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="vbp_wav_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_live_wav_writer_creates_valid_wav(temp_dir):
    wav_path = os.path.join(temp_dir, "audio.wav")
    writer = LiveWavWriter(wav_path, sample_rate=16000, channels=1)

    # Write 1 second of float32 audio
    t = np.linspace(0, 1.0, 16000, endpoint=False, dtype=np.float32)
    samples = 0.5 * np.sin(2 * np.pi * 440 * t)

    writer.write_samples(samples)
    writer.close()

    assert os.path.isfile(wav_path)
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


def test_wav_header_is_patched_live(temp_dir):
    wav_path = os.path.join(temp_dir, "audio.wav")
    writer = LiveWavWriter(wav_path, sample_rate=16000, channels=1)

    # Write 8000 samples (0.5s)
    samples = np.zeros(8000, dtype=np.float32)
    writer.write_samples(samples)
    writer.patch_header()

    # Without calling close(), check that the file is already a valid WAV
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnframes() == 8000
        assert wf.getframerate() == 16000

    writer.close()
