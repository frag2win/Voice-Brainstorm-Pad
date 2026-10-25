"""
Microphone Verification & Diagnostic Tool — Voice Brainstorm Pad

Run this script to:
 1. List every functional microphone on your system.
 2. Verify which microphone is the default.
 3. Test recording from each microphone and see live volume levels.
 4. Save short test WAV files to confirm audio capture works on each device.

Usage:
  & ".\\.venv\\Scripts\\python.exe" test_microphones.py
"""

import sys
import time
import struct
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice is not installed. Run from the virtual environment:")
    print('  & ".\\.venv\\Scripts\\python.exe" test_microphones.py')
    sys.exit(1)


def get_functional_devices():
    hostapis = sd.query_hostapis()
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            hostapi_name = hostapis[dev["hostapi"]]["name"] if dev.get("hostapi") is not None else ""
            if "WDM-KS" in hostapi_name:
                continue
            try:
                sd.check_input_settings(device=i, samplerate=16000, channels=1)
                devices.append({
                    "id": i,
                    "name": dev.get("name", f"Microphone {i}"),
                    "hostapi": hostapi_name,
                    "default_samplerate": dev.get("default_samplerate", 16000),
                })
            except Exception:
                pass
    return devices


def save_wav(file_path: str, samples: np.ndarray, sample_rate: int = 16000):
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype(np.int16)
    data_bytes = pcm16.tobytes()
    data_size = len(data_bytes)
    riff_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE", b"fmt ",
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size
    )
    with open(file_path, "wb") as f:
        f.write(header)
        f.write(data_bytes)


def test_device(dev_id: int, dev_name: str, duration_s: float = 3.0):
    print(f"\n=======================================================")
    print(f" Testing: [ID {dev_id}] {dev_name}")
    print(f" -> Speak now for {duration_s:.0f} seconds...")
    print(f"=======================================================")

    recorded_chunks = []
    sample_rate = 16000

    def callback(indata, frames, time_info, status):
        recorded_chunks.append(indata[:, 0].copy())
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        # Visual meter bar (30 chars)
        bar_len = min(30, int(rms * 150))
        meter = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
        sys.stdout.write(f"\r Level: [{meter}] RMS: {rms:.4f} ")
        sys.stdout.flush()

    try:
        with sd.InputStream(device=dev_id, samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
            time.sleep(duration_s)
    except Exception as e:
        print(f"\n [!] Failed to open device: {e}")
        return

    print("\n -> Recording complete!")
    if recorded_chunks:
        full_audio = np.concatenate(recorded_chunks)
        peak = float(np.max(np.abs(full_audio)))
        avg_rms = float(np.sqrt(np.mean(full_audio ** 2)))
        out_file = f"test_mic_{dev_id}.wav"
        save_wav(out_file, full_audio, sample_rate)
        print(f" -> Peak Amplitude: {peak:.4f} | Avg RMS: {avg_rms:.4f}")
        print(f" -> Saved sample to: {out_file} ({len(full_audio)/sample_rate:.2f}s)")
        if peak > 0.02:
            print(" -> Status: \u2705 SOUND DETECTED! Microphone is active and working.")
        else:
            print(" -> Status: \u26a0\ufe0f Very low sound / silence detected. Check volume or speak louder.")


def main():
    print("=" * 60)
    print(" Voice Brainstorm Pad — Microphone Device Inspector")
    print("=" * 60)

    try:
        default_in = sd.default.device[0]
        default_name = sd.query_devices(default_in)["name"] if default_in >= 0 else "Unknown"
    except Exception:
        default_in = -1
        default_name = "Unknown"

    print(f"\nSystem Default Recording Device: [ID {default_in}] {default_name}")

    devices = get_functional_devices()
    print(f"\nAvailable Valid Microphones ({len(devices)} found):")
    for d in devices:
        is_default = " (DEFAULT)" if d["id"] == default_in else ""
        print(f"  [{d['id']}] {d['name']} [{d['hostapi']}]{is_default}")

    if not devices:
        print("\nNo functional microphones detected!")
        return

    print("\nRunning automated 2-second capture test on all detected inputs:")
    for d in devices:
        test_device(d["id"], f"{d['name']} [{d['hostapi']}]", duration_s=2.0)

    print("\n" + "=" * 60)
    print(" Diagnostic Complete. Check the generated test_mic_*.wav files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
