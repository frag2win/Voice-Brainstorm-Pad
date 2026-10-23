#!/usr/bin/env python3
"""
Standalone Whisper CPU Benchmark Tool
Measures whether local faster-whisper models can keep up with fast continuous speech
on Windows CPU (4 cores / 8 threads, no NVIDIA GPU).

Usage:
    python bench.py [--seconds 60] [--file path/to/sample.wav] [--models base.en distil-small.en small.en]
"""

import os
import sys
import time
import wave
import argparse
import platform
import threading
from typing import List, Dict, Any, Optional

# Configure UTF-8 stdout/stderr on Windows so special characters and progress indicators never crash cp1252 consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress Hugging Face symlinks warning on Windows before importing faster_whisper
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import sounddevice as sd
import psutil
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio


class CPUMonitor:
    """Monitors CPU frequency (MHz) and utilization (%) every interval_sec in a background thread."""

    def __init__(self, interval_sec: float = 2.0):
        self.interval = interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.mhz_samples: List[float] = []
        self.util_samples: List[float] = []

    def start(self):
        self._stop_event.clear()
        self.mhz_samples = []
        self.util_samples = []

        # Prime psutil utilization tracker
        try:
            psutil.cpu_percent(interval=None)
            freq = psutil.cpu_freq()
            if freq and freq.current:
                self.mhz_samples.append(freq.current)
        except Exception:
            pass

        def _worker():
            while not self._stop_event.is_set():
                if self._stop_event.wait(self.interval):
                    break
                try:
                    pct = psutil.cpu_percent(interval=None)
                    freq = psutil.cpu_freq()
                    if freq and freq.current:
                        self.mhz_samples.append(freq.current)
                    if pct is not None:
                        self.util_samples.append(pct)
                except Exception:
                    pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=1.0)

        # Sample once more at the end
        try:
            freq = psutil.cpu_freq()
            pct = psutil.cpu_percent(interval=None)
            if freq and freq.current:
                self.mhz_samples.append(freq.current)
            if pct is not None:
                self.util_samples.append(pct)
        except Exception:
            pass

        min_mhz = min(self.mhz_samples) if self.mhz_samples else 0.0
        avg_mhz = sum(self.mhz_samples) / len(self.mhz_samples) if self.mhz_samples else 0.0
        max_mhz = max(self.mhz_samples) if self.mhz_samples else 0.0
        avg_util = sum(self.util_samples) / len(self.util_samples) if self.util_samples else 0.0

        return {
            "min_mhz": min_mhz,
            "avg_mhz": avg_mhz,
            "max_mhz": max_mhz,
            "avg_util": avg_util,
            "sample_count": len(self.mhz_samples),
        }


def record_audio(output_wav_path: str, seconds: float = 60.0, samplerate: int = 16000):
    """Records audio from the default microphone at 16 kHz mono 16-bit PCM and saves to WAV."""
    print(f"\n[*] Preparing to record {seconds:.0f}s of continuous speech at {samplerate} Hz (mono)...")
    print("    Tip: Speak continuously at your normal brainstorming pace.")
    print("    Recording starts in 3 seconds:")
    for count in range(3, 0, -1):
        print(f"    {count}...", flush=True)
        time.sleep(1)

    total_frames = int(seconds * samplerate)
    print(f"\n[REC] RECORDING NOW ({seconds:.0f} seconds)... Speak!")

    audio_data = sd.rec(total_frames, samplerate=samplerate, channels=1, dtype="int16")

    # Countdown timer in console
    start_rec = time.time()
    while True:
        elapsed = time.time() - start_rec
        remaining = max(0.0, seconds - elapsed)
        bar_len = 30
        progress = min(1.0, elapsed / seconds)
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\r    [{bar}] {elapsed:4.1f}s / {seconds:4.1f}s (Remaining: {remaining:4.1f}s)", end="", flush=True)
        if elapsed >= seconds:
            break
        time.sleep(0.5)

    sd.wait()
    print("\n[OK] Recording finished! Saving audio to WAV file...")

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

    print(f"[+] Saved audio sample: {output_wav_path}\n")


def get_system_summary() -> Dict[str, str]:
    """Gathers system specs for reporting."""
    cpu_freq = psutil.cpu_freq()
    base_mhz = f"{cpu_freq.max:.0f} MHz" if cpu_freq and cpu_freq.max else "N/A"
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or "CPU",
        "cores": f"{psutil.cpu_count(logical=False)} physical cores / {psutil.cpu_count(logical=True)} logical threads",
        "max_freq": base_mhz,
        "ram": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "python": sys.version.split()[0],
    }


def load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads audio from WAV file (using standard library wave) or falls back to decode_audio."""
    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)

            if sampwidth == 2:  # 16-bit PCM
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:  # 8-bit PCM
                samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:  # 32-bit float or int32
                samples = np.frombuffer(data, dtype=np.float32)
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth}")

            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)

            if framerate != target_sr:
                duration = len(samples) / float(framerate)
                target_length = int(round(duration * target_sr))
                indices = np.linspace(0, len(samples) - 1, target_length)
                samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

            return samples
    except Exception:
        # Fallback to faster_whisper decode_audio
        return decode_audio(file_path, sampling_rate=target_sr)


def run_benchmark(
    audio_path: str,
    models: List[str],
    cpu_threads: int = 4,
    compute_type: str = "int8",
    num_workers: int = 1,
    log_cpu: bool = True,
    results_path: str = "results.md",
):
    """Runs the benchmark across all specified models and writes/prints results."""
    if not os.path.isfile(audio_path):
        print(f"[!] Error: Audio file '{audio_path}' not found!")
        sys.exit(1)

    print(f"[*] Decoding audio file: {audio_path}")
    audio_float32 = load_audio_file(audio_path, target_sr=16000)
    audio_seconds = len(audio_float32) / 16000.0
    audio_minutes = audio_seconds / 60.0
    print(f"[i] Audio length: {audio_seconds:.2f} seconds ({audio_minutes:.2f} minutes)\n")

    sys_info = get_system_summary()
    print("=" * 80)
    print(" SYSTEM ENVIRONMENT")
    print(f" OS:         {sys_info['os']}")
    print(f" CPU:        {sys_info['cpu']}")
    print(f" Topology:   {sys_info['cores']} (Benchmark uses cpu_threads={cpu_threads})")
    print(f" Max Freq:   {sys_info['max_freq']}")
    print(f" Memory:     {sys_info['ram']}")
    print(f" Python:     {sys_info['python']}")
    print(f" Backend:    faster-whisper (compute_type={compute_type}, num_workers={num_workers})")
    print("=" * 80)

    results = []

    for idx, model_name in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Benchmarking Model: {model_name}")
        print("-" * 60)

        # 1. Load Model (measure load time)
        print(f"[*] Loading '{model_name}' on CPU (int8, {cpu_threads} threads)...")
        t0_load = time.perf_counter()
        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        load_time = time.perf_counter() - t0_load
        print(f"    Model loaded in: {load_time:.2f}s")

        # 2. Warm-up transcription (5 seconds, excluded from timing)
        warmup_duration = min(5.0, audio_seconds)
        warmup_samples = int(warmup_duration * 16000)
        warmup_slice = audio_float32[:warmup_samples]
        print(f"[*] Running 5-second warm-up transcription (excluded from timing)...", end="", flush=True)
        warmup_segs, _ = model.transcribe(
            warmup_slice,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        _ = list(warmup_segs)  # Force full generator evaluation
        print(" Done.")

        # 3. Full Transcription with optional CPU monitoring
        cpu_monitor = CPUMonitor(interval_sec=2.0) if log_cpu else None
        if cpu_monitor:
            cpu_monitor.start()

        print(f"[*] Transcribing full sample ({audio_seconds:.2f}s)...", flush=True)
        t0_transcribe = time.perf_counter()

        segments, info = model.transcribe(
            audio_float32,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        segments_list = list(segments)  # Fully evaluate transcription
        transcribe_time = time.perf_counter() - t0_transcribe

        cpu_stats = cpu_monitor.stop() if cpu_monitor else {}

        # 4. Metrics calculation
        rtf = transcribe_time / audio_seconds if audio_seconds > 0 else 0.0
        transcript_text = " ".join(seg.text.strip() for seg in segments_list).strip()
        word_count = len(transcript_text.split())
        wpm = (word_count / audio_minutes) if audio_minutes > 0 else 0.0
        keeps_up = rtf < 1.0

        min_mhz = cpu_stats.get("min_mhz", 0.0)
        avg_mhz = cpu_stats.get("avg_mhz", 0.0)
        avg_util = cpu_stats.get("avg_util", 0.0)

        result_entry = {
            "model": model_name,
            "load_time": load_time,
            "audio_sec": audio_seconds,
            "transcribe_sec": transcribe_time,
            "rtf": rtf,
            "keeps_up": keeps_up,
            "word_count": word_count,
            "wpm": wpm,
            "min_mhz": min_mhz,
            "avg_mhz": avg_mhz,
            "avg_util": avg_util,
            "transcript": transcript_text,
        }
        results.append(result_entry)

        # Print run summary
        status_str = "YES (Real-Time)" if keeps_up else "NO (Lagging)"
        print(f"   Transcribe Time:   {transcribe_time:.2f}s")
        print(f"   Real-Time Factor:  {rtf:.2f}x  [{status_str}]")
        print(f"   Word Count:        {word_count} words ({wpm:.1f} WPM)")
        if log_cpu and avg_mhz > 0:
            print(f"   CPU Clock (MHz):   Min: {min_mhz:.0f} MHz | Avg: {avg_mhz:.0f} MHz")
            print(f"   CPU Utilisation:   Avg: {avg_util:.1f}%")
        print(f"   Transcript Preview: \"{(transcript_text[:120] + '...') if len(transcript_text) > 120 else transcript_text}\"")

    # Format Markdown Table and Print
    md_content = generate_markdown_report(sys_info, results, cpu_threads, compute_type, audio_path)

    # Save to results.md
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[+] Full results table saved to: {results_path}")

    # Print summary table to stdout
    print_cli_table(results)


def print_cli_table(results: List[Dict[str, Any]]):
    """Prints a clean CLI table of the benchmark results."""
    print("\n" + "=" * 105)
    print(" BENCHMARK RESULTS SUMMARY")
    print("=" * 105)
    header = f"{'Model':<18} | {'Load (s)':<8} | {'Transcribe':<10} | {'RTF':<7} | {'Keeps Up?':<12} | {'Words':<6} | {'WPM':<6} | {'Avg MHz':<8} | {'Avg CPU %':<9}"
    print(header)
    print("-" * 105)
    for r in results:
        status = "YES (<=1.0)" if r["keeps_up"] else "NO (>1.0)"
        line = (
            f"{r['model']:<18} | "
            f"{r['load_time']:>7.2f}s | "
            f"{r['transcribe_sec']:>9.2f}s | "
            f"{r['rtf']:>6.2f}x | "
            f"{status:<12} | "
            f"{r['word_count']:>5} | "
            f"{r['wpm']:>5.1f} | "
            f"{r['avg_mhz']:>7.0f} | "
            f"{r['avg_util']:>8.1f}%"
        )
        print(line)
    print("=" * 105)
    print(" * Real-Time Factor (RTF) = Transcribe Seconds / Audio Seconds. Lower is faster.")
    print(" * RTF < 1.0 means transcription finishes faster than the speech duration (keeps up in real time).\n")


def generate_markdown_report(
    sys_info: Dict[str, str],
    results: List[Dict[str, Any]],
    cpu_threads: int,
    compute_type: str,
    audio_path: str,
) -> str:
    """Constructs the markdown results document."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append("# Local Whisper Benchmark Results\n")
    md.append(f"**Date**: {timestamp}  ")
    md.append(f"**Audio File**: `{os.path.basename(audio_path)}` ({results[0]['audio_sec']:.2f}s)  ")
    md.append(f"**Compute**: CPU only (`{compute_type}`, `cpu_threads={cpu_threads}`, `num_workers=1`, `beam_size=1`)\n")

    md.append("## System Environment\n")
    md.append(f"- **OS**: {sys_info['os']}")
    md.append(f"- **CPU**: {sys_info['cpu']}")
    md.append(f"- **Topology**: {sys_info['cores']}")
    md.append(f"- **Memory**: {sys_info['ram']}")
    md.append(f"- **Python**: {sys_info['python']}\n")

    md.append("## Performance Comparison Table\n")
    md.append(
        "| Model | Load Time (s) | Audio (s) | Transcribe (s) | RTF (Transcribe/Audio) | Keeps Up? | Word Count | Spoken WPM | Min CPU MHz | Avg CPU MHz | Avg CPU % |"
    )
    md.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for r in results:
        status_badge = "**YES** (Real-Time)" if r["keeps_up"] else "**NO** (Lagging)"
        md.append(
            f"| `{r['model']}` | {r['load_time']:.2f} | {r['audio_sec']:.2f} | {r['transcribe_sec']:.2f} | **{r['rtf']:.2f}x** | {status_badge} | {r['word_count']} | {r['wpm']:.1f} | {r['min_mhz']:.0f} | {r['avg_mhz']:.0f} | {r['avg_util']:.1f}% |"
        )

    md.append("\n> **Note**: **Real-Time Factor (RTF)** = Transcribe Time / Audio Duration. An RTF of 0.25x means the model processes audio 4x faster than real-time.\n")

    md.append("## Full Transcripts\n")
    for r in results:
        md.append(f"### `{r['model']}`\n")
        if r["transcript"]:
            md.append(f"> {r['transcript']}\n")
        else:
            md.append("> *(No speech detected or empty transcript)*\n")

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark local faster-whisper models on CPU for real-time speech transcription."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Duration of live microphone recording in seconds (default: 60).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to an existing WAV/audio file to reuse (bypasses mic recording).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["base.en", "distil-small.en", "small.en"],
        help="Whisper models to benchmark (default: base.en distil-small.en small.en).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Number of CPU threads to allocate to faster-whisper (default: 4).",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Quantization type for faster-whisper (default: int8).",
    )
    parser.add_argument(
        "--no-cpu-log",
        action="store_true",
        help="Disable CPU MHz and utilization logging during transcription.",
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.wav"),
        help="Target path where microphone recording is saved (default: bench/sample.wav).",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.md"),
        help="Path where markdown results table is saved (default: bench/results.md).",
    )

    args = parser.parse_args()

    # Determine audio source
    if args.file:
        audio_file = os.path.abspath(args.file)
        if not os.path.exists(audio_file):
            print(f"[!] Error: Specified --file '{audio_file}' does not exist.")
            sys.exit(1)
        print(f"[+] Reusing existing audio file: {audio_file}")
    else:
        audio_file = os.path.abspath(args.sample_path)
        record_audio(audio_file, seconds=args.seconds)

    # Run benchmark
    run_benchmark(
        audio_path=audio_file,
        models=args.models,
        cpu_threads=args.cpu_threads,
        compute_type=args.compute_type,
        num_workers=1,
        log_cpu=not args.no_cpu_log,
        results_path=os.path.abspath(args.results_path),
    )


if __name__ == "__main__":
    main()
