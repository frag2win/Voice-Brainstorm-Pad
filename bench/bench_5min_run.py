#!/usr/bin/env python3
"""
bench/bench_5min_run.py - Real-Time Streaming Speech Benchmark
Measures:
  - Real-Time Factor (RTF) per-chunk and overall
  - p95 End-to-End Latency (ms)
  - Max Queue Depth
  - Word Error Rate (WER) against Ground Truth Reference Text
"""

import os
import sys
import time
import wave
import shutil
import argparse
import psutil
import platform
import numpy as np
from typing import List, Dict, Any

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

bench_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(bench_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sidecar.chunker import AudioChunker, ChunkerConfig
from sidecar.transcriber import Transcriber, TranscriberConfig
from sidecar.recorder import LiveWavWriter


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, Any]:
    """Computes Word Error Rate using standard Levenshtein distance on normalized words."""
    import re

    def normalize(text: str) -> List[str]:
        # Lowercase, remove punctuation, split on whitespace
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [w for w in cleaned.split() if w]

    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    r_len = len(ref_words)
    h_len = len(hyp_words)

    if r_len == 0:
        return {"wer": 0.0 if h_len == 0 else 1.0, "sub": 0, "ins": h_len, "del": 0, "ref_words": 0, "hyp_words": h_len}

    # Dynamic programming matrix (r_len + 1) x (h_len + 1)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    distance = dp[r_len][h_len]
    wer = distance / float(r_len)

    return {
        "wer": wer,
        "distance": distance,
        "ref_words": r_len,
        "hyp_words": h_len,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Streaming Speech Benchmark")
    parser.add_argument(
        "--audio",
        type=str,
        default=os.path.join(bench_dir, "speech_5min.wav"),
        help="Path to input audio WAV file (default: bench/speech_5min.wav)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default=os.path.join(bench_dir, "speech_5min_ref.txt"),
        help="Path to ground truth reference text (default: bench/speech_5min_ref.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(bench_dir, "results_5min.md"),
        help="Path to markdown output report (default: bench/results_5min.md)",
    )
    return parser.parse_args()


def run_benchmark():
    args = parse_args()
    wav_path = os.path.abspath(args.audio)
    ref_path = os.path.abspath(args.ref)
    report_path = os.path.abspath(args.output)

    if not os.path.exists(wav_path):
        print(f"ERROR: Audio file not found: {wav_path}")
        sys.exit(1)
    if not os.path.exists(ref_path):
        print(f"ERROR: Reference text not found: {ref_path}")
        sys.exit(1)

    with open(ref_path, "r", encoding="utf-8") as f:
        reference_text = f.read().strip()

    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    total_audio_s = n_frames / float(framerate)
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print("=" * 80)
    print("REAL-TIME STREAMING SPEECH BENCHMARK")
    print("=" * 80)
    print(f"Audio Sample:     {wav_path}")
    print(f"Duration:         {total_audio_s:.2f} s ({total_audio_s/60.0:.2f} minutes)")
    print(f"Sampling Rate:    {framerate} Hz (Mono: {n_channels == 1})")
    print(f"Reference Words:  {len(reference_text.split())} words")

    vmem = psutil.virtual_memory()
    print(f"Host CPU / Cores: {platform.processor()} ({psutil.cpu_count(logical=False)} cores / {psutil.cpu_count(logical=True)} threads)")
    print(f"Host RAM:         {vmem.total / (1024**3):.2f} GB total, {vmem.available / (1024**3):.2f} GB available")
    print("-" * 80)

    # Temporary session directory
    session_dir = os.path.join(bench_dir, "session_benchmark_run")
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(os.path.join(session_dir, "chunks"), exist_ok=True)

    # Initialize LiveWavWriter
    audio_wav_path = os.path.join(session_dir, "audio.wav")
    wav_writer = LiveWavWriter(audio_wav_path, sample_rate=framerate, channels=1)

    # Metrics collectors
    queue_depth_samples: List[int] = []
    chunk_results: List[Any] = []

    # Initialize Transcriber
    print("[*] Initializing Transcriber (small.en, int8, cpu_threads=4)...")
    transcriber = Transcriber(
        config=TranscriberConfig(
            model_name="small.en",
            compute_type="int8",
            cpu_threads=4,
        ),
        on_transcript=lambda res: chunk_results.append(res),
        on_chunk_failed=lambda err: print(f"  [ERROR] Chunk #{err.get('seq')}: {err.get('error')}"),
        auto_warmup=True,
    )

    # Initialize Chunker
    chunker = AudioChunker(
        session_dir=session_dir,
        config=ChunkerConfig(
            target_chunk_s=8.0,
            pause_cut_s=2.5,
        ),
    )

    def enqueue_chunk(chunk):
        cur_depth = transcriber.pending_count + 1
        queue_depth_samples.append(cur_depth)
        transcriber.enqueue(
            seq=chunk.seq,
            audio_path=chunk.file_path,
            duration_s=chunk.duration_s,
            session_dir=session_dir,
            t_start=chunk.t_start,
            t_end=chunk.t_end,
        )

    # Simulate real-time streaming: feed 512 samples (32 ms) at a time
    frame_size = 512
    total_samples = len(samples)
    print(f"[*] Streaming {total_samples} samples through Chunker & Transcriber...")

    for i in range(0, total_samples, frame_size):
        frame = samples[i : i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        wav_writer.write_samples(frame)
        chunk = chunker.process_frame(frame)
        if chunk is not None:
            enqueue_chunk(chunk)

        if (i // frame_size) % 10 == 0:
            queue_depth_samples.append(transcriber.pending_count)

    print("\n[*] Audio stream complete. Flushing trailing chunker buffer...")
    final_chunk = chunker.flush()
    if final_chunk is not None:
        enqueue_chunk(final_chunk)
    wav_writer.close()

    queue_depth_samples.append(transcriber.pending_count)

    print(f"[*] Draining transcriber queue (pending: {transcriber.pending_count})...")
    drain_t0 = time.perf_counter()
    while transcriber.pending_count > 0:
        time.sleep(0.05)
        queue_depth_samples.append(transcriber.pending_count)

    drain_time = time.perf_counter() - drain_t0
    print(f"[*] Transcriber queue fully drained in {drain_time:.2f}s.")
    transcriber.stop()

    # Calculate Metrics
    max_queue_depth = max(queue_depth_samples) if queue_depth_samples else 0
    total_transcribe_time = sum(res.transcribe_s for res in chunk_results)
    overall_rtf = total_transcribe_time / total_audio_s if total_audio_s > 0 else 0.0

    all_chunk_rtfs = [res.rtf for res in chunk_results]
    mean_chunk_rtf = float(np.mean(all_chunk_rtfs)) if all_chunk_rtfs else 0.0
    p95_chunk_rtf = float(np.percentile(all_chunk_rtfs, 95)) if all_chunk_rtfs else 0.0

    latencies = [res.latency_ms for res in chunk_results]
    p95_latency_ms = float(np.percentile(latencies, 95)) if latencies else 0.0
    median_latency_ms = float(np.median(latencies)) if latencies else 0.0
    max_latency_ms = float(np.max(latencies)) if latencies else 0.0

    # Assemble full transcribed text
    hyp_text = " ".join(res.text.strip() for res in chunk_results if res.text.strip())
    wer_stats = calculate_wer(reference_text, hyp_text)
    wer_pct = wer_stats["wer"] * 100.0

    # Formula-based verdict calculations
    # 1. Overall RTF: NFR-1 target is < 1.0x (comfortably < 0.5x for iteration-0 gate)
    if overall_rtf < 0.5:
        overall_rtf_status = "PASS (comfortably below 0.5x)"
    elif overall_rtf <= 1.0:
        overall_rtf_status = "PASS (below 1.0x, but above 0.5x iteration-0 target)"
    else:
        overall_rtf_status = f"FAIL (exceeds real-time: {overall_rtf:.3f}x > 1.0x)"

    # 2. Mean Chunk RTF
    mean_chunk_rtf_status = "PASS" if mean_chunk_rtf <= 1.0 else f"FAIL ({mean_chunk_rtf:.3f}x > 1.0x)"

    # 3. p95 End-to-End Latency: NFR-2 target is <= 10.0s (10000ms)
    p95_latency_status = "PASS" if p95_latency_ms <= 10000.0 else f"FAIL ({p95_latency_ms/1000.0:.2f}s > 10.0s)"

    # 4. Max Queue Depth: bounded queue (<= 2 chunks)
    queue_depth_status = "PASS" if max_queue_depth <= 2 else f"FAIL (unbounded backlog: {max_queue_depth} chunks)"

    # 5. Word Error Rate: NFR-3 target is <= 8.0%
    wer_status = "PASS" if wer_pct <= 8.0 else f"FAIL ({wer_pct:.2f}% > 8.0%)"

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Audio Duration:    {total_audio_s:.2f} s ({total_audio_s/60.0:.2f} min)")
    print(f"Total Transcribe Time:   {total_transcribe_time:.2f} s")
    print(f"Overall Real-Time Factor:{overall_rtf:.3f}x -> {overall_rtf_status}")
    print(f"Mean Chunk RTF:          {mean_chunk_rtf:.3f}x -> {mean_chunk_rtf_status}")
    print(f"p95 Chunk RTF:           {p95_chunk_rtf:.3f}x")
    print(f"Median Latency:          {median_latency_ms:.1f} ms")
    print(f"p95 Latency:             {p95_latency_ms:.1f} ms ({p95_latency_ms/1000.0:.2f}s) -> {p95_latency_status}")
    print(f"Max Latency:             {max_latency_ms:.1f} ms ({max_latency_ms/1000.0:.2f}s)")
    print(f"Max Queue Depth:         {max_queue_depth} chunks -> {queue_depth_status}")
    print(f"Reference Word Count:    {wer_stats['ref_words']}")
    print(f"Hypothesis Word Count:   {wer_stats['hyp_words']}")
    print(f"Word Error Rate (WER):   {wer_pct:.2f}% -> {wer_status}")
    print("=" * 80)

    # Save summary report to output markdown
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("# Real-Time Streaming Speech Benchmark Results\n\n")
        rf.write(f"- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write(f"- **OS**: {platform.system()} {platform.release()}\n")
        rf.write(f"- **CPU**: {platform.processor()} ({psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical)\n")
        rf.write(f"- **RAM**: {vmem.total / (1024**3):.2f} GB total, {vmem.available / (1024**3):.2f} GB available\n")
        rf.write(f"- **Model**: `small.en` (compute_type=int8, cpu_threads=4)\n")
        rf.write(f"- **Audio File**: `{wav_path}`\n")
        rf.write(f"- **Audio Duration**: {total_audio_s:.2f}s ({total_audio_s/60.0:.2f} min)\n\n")
        rf.write("## Empirical Measurements & Verdicts\n\n")
        rf.write("| Metric | Measured Value | Spec Target (PRD / TRD) | Status |\n")
        rf.write("| :--- | :---: | :---: | :---: |\n")
        rf.write(f"| **Overall RTF** | **{overall_rtf:.3f}x** | < 1.0x (Real-Time) | {overall_rtf_status} |\n")
        rf.write(f"| **Mean Chunk RTF** | **{mean_chunk_rtf:.3f}x** | < 1.0x | {mean_chunk_rtf_status} |\n")
        rf.write(f"| **p95 End-to-End Latency** | **{p95_latency_ms:.1f} ms** ({p95_latency_ms/1000.0:.2f}s) | <= 10.0s (NFR-2) | {p95_latency_status} |\n")
        rf.write(f"| **Median Latency** | **{median_latency_ms:.1f} ms** ({median_latency_ms/1000.0:.2f}s) | N/A | Informational |\n")
        rf.write(f"| **Max Queue Depth** | **{max_queue_depth} chunks** | Bounded (NFR-1) | {queue_depth_status} |\n")
        rf.write(f"| **Word Error Rate (WER)** | **{wer_pct:.2f}%** | <= 8.0% (NFR-3) | {wer_status} |\n\n")
        rf.write("## Per-Chunk Telemetry Breakdown\n\n")
        rf.write("| Seq | Duration (s) | Transcribe (s) | RTF | Latency (ms) | Words | Clean Transcript Text |\n")
        rf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for res in chunk_results:
            rf.write(f"| #{res.seq} | {res.audio_s:.2f}s | {res.transcribe_s:.2f}s | {res.rtf:.3f}x | {res.latency_ms:.1f}ms | {res.word_count} | {res.text} |\n")
        rf.write("\n## Full Output Transcript\n\n")
        rf.write(f"> {hyp_text}\n")

    print(f"\n[+] Wrote comprehensive report to: {report_path}")

    # Clean up temporary session audio
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)


if __name__ == "__main__":
    run_benchmark()
