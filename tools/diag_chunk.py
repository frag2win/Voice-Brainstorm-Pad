#!/usr/bin/env python3
"""
diag_chunk.py - Diagnostic inspection for failed or problematic audio chunks.
Analyzes WAV properties, reproduces and documents failure modes (such as MKL memory allocation errors),
and verifies clean transcription under current memory-safe transcriber pipeline.
"""

import os
import sys
import time
import wave
import psutil
import platform
import traceback
import numpy as np

# Set environment UTF-8 for Windows console
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure sidecar module path is importable
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sidecar.transcriber import Transcriber, TranscriberConfig


def get_system_telemetry():
    vmem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor(),
        "cores": f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical",
        "total_ram_gb": vmem.total / (1024 ** 3),
        "avail_ram_gb": vmem.available / (1024 ** 3),
        "used_ram_pct": vmem.percent,
        "process_rss_mb": proc.memory_info().rss / (1024 ** 2),
    }


def analyze_wav(wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_frames = wf.readframes(n_frames)

    duration_s = n_frames / float(framerate)
    samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0

    return {
        "path": wav_path,
        "channels": n_channels,
        "sample_rate": framerate,
        "sample_width_bytes": sampwidth,
        "n_frames": n_frames,
        "duration_s": duration_s,
        "rms_energy": rms,
        "peak_amplitude": peak,
    }


def main():
    default_failed_chunk = r"C:\Users\sunanda.AMFIIND\Documents\VoiceBrainstormPad\sessions\2026-10-23T15-08-08_test-3\chunks\chunk_0009.wav"
    wav_path = sys.argv[1] if len(sys.argv) > 1 else default_failed_chunk

    print("=" * 80)
    print("DIAGNOSTIC REPORT: AUDIO CHUNK FAILURE ANALYSIS")
    print("=" * 80)
    print(f"Target WAV: {wav_path}")

    if not os.path.exists(wav_path):
        print(f"ERROR: Target file does not exist: {wav_path}")
        sys.exit(1)

    # 1. System Telemetry
    sys_info = get_system_telemetry()
    print("\n--- System & Memory Telemetry ---")
    print(f"OS:                 {sys_info['os']}")
    print(f"CPU:                {sys_info['cpu']}")
    print(f"Topology:           {sys_info['cores']}")
    print(f"Total RAM:          {sys_info['total_ram_gb']:.2f} GB")
    print(f"Available RAM:      {sys_info['avail_ram_gb']:.2f} GB ({sys_info['used_ram_pct']:.1f}% used)")
    print(f"Process RSS (Init): {sys_info['process_rss_mb']:.2f} MB")

    # 2. Audio Properties
    audio_info = analyze_wav(wav_path)
    print("\n--- Audio File Properties ---")
    print(f"Duration:           {audio_info['duration_s']:.3f} s ({audio_info['n_frames']} frames)")
    print(f"Sample Rate:        {audio_info['sample_rate']} Hz (Mono: {audio_info['channels'] == 1})")
    print(f"Sample Width:       {audio_info['sample_width_bytes']} bytes (PCM16)")
    print(f"RMS Energy:         {audio_info['rms_energy']:.4f}")
    print(f"Peak Amplitude:     {audio_info['peak_amplitude']:.4f}")

    # 3. Documenting the Failure that Occurred in Session 2026-10-23T15-08-08_test-3
    print("\n--- Documented Failure Mode: Chunk #9 in Live Run ---")
    print("Context: In session 2026-10-23T15-08-08_test-3, chunks #4-#8 and #10-#13 transcribed successfully,")
    print("         but chunk #9 was skipped and missing from transcripts.jsonl.")
    print("Root Cause: High thread concurrency and unbounded temperature fallback list (0.0 .. 1.0)")
    print("            triggered Intel MKL heap exhaustion under Windows CPython.")
    print("\nError Text Logged during Failure:")
    print("-" * 60)
    print("RuntimeError: mkl_malloc: failed to allocate memory")
    print("  File \".venv\\Lib\\site-packages\\faster_whisper\\transcribe.py\", line 689, in generate_with_fallback")
    print("    for (result, ... in decode():")
    print("RuntimeError: mkl_malloc: failed to allocate memory")
    print("-" * 60)

    # 4. Verifying with Current Transcriber Pipeline
    print("\n--- Active Transcriber Pipeline Validation ---")
    print("Active Safeguards:")
    print("  - MKL_NUM_THREADS clamped to 4 (prevents thread multiplication)")
    print("  - OMP_NUM_THREADS clamped to 4")
    print("  - Temperature fallback bounded to (0.0, 0.2, 0.4)")
    print("  - Exception handler with gc.collect() and greedy temperature=0.0 retry")

    results = []
    errors = []

    transcriber = Transcriber(
        config=TranscriberConfig(
            model_name="small.en",
            compute_type="int8",
            cpu_threads=4,
        ),
        on_transcript=lambda res: results.append(res),
        on_chunk_failed=lambda err: errors.append(err),
    )

    session_dir = os.path.dirname(os.path.dirname(os.path.abspath(wav_path)))

    t0 = time.perf_counter()
    transcriber.enqueue(
        seq=9,
        audio_path=wav_path,
        duration_s=audio_info["duration_s"],
        session_dir=session_dir,
        t_start=57.12,
        t_end=65.152,
    )
    # Drain queue
    while transcriber.pending_count > 0:
        time.sleep(0.05)
    transcriber.stop()
    elapsed = time.perf_counter() - t0

    proc = psutil.Process(os.getpid())
    print(f"Process RSS (Post-transcribe): {proc.memory_info().rss / (1024 ** 2):.2f} MB")

    if errors:
        print("\n[FAIL] Pipeline reported errors:")
        for err in errors:
            print(f"  Seq {err.get('seq')}: {err.get('error')}")
    elif results:
        res = results[0]
        print("\n[SUCCESS] Chunk #9 transcribed cleanly:")
        print(f"  Seq:              {res.seq}")
        print(f"  Text:             \"{res.text}\"")
        print(f"  Word Count:       {res.word_count}")
        print(f"  Duration:         {res.audio_s:.3f} s")
        print(f"  Transcribe Time:  {res.transcribe_s:.3f} s")
        print(f"  Real-Time Factor: {res.rtf:.3f}x (Speedup: {1.0 / res.rtf:.2f}x real-time)")
        print(f"  End-to-End Lat:   {res.latency_ms:.1f} ms")
        print(f"  Queue Wait:       {res.queue_wait_ms:.1f} ms")
    else:
        print("\n[WARN] No transcript emitted (silence or non-speech filtered)")

    print("=" * 80)


if __name__ == "__main__":
    main()
