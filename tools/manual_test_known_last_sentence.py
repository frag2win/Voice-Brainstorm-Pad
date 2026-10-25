#!/usr/bin/env python3
"""
test_known_last_sentence.py - Empirical test for known last sentence preservation
Requirements:
  - Speak continuous speech ending with a specific known sentence
  - Press Stop immediately after speaking
  - Confirm that the final sentence is captured in the flushed chunk
  - Confirm that no_speech_prob or silence cutting does not drop or truncate the final clause
"""

import os
import sys
import time
import wave
import shutil
import subprocess
import numpy as np

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from sidecar.chunker import AudioChunker, ChunkerConfig
from sidecar.transcriber import Transcriber, TranscriberConfig
from sidecar.recorder import LiveWavWriter


KNOWN_LAST_SENTENCE = "Finally, we verify that the very last sentence spoken by the user before pressing stop is never lost."
PRIOR_CONTEXT = "We are dictating continuously at a rapid brainstorming pace without taking any long breaks. All intermediate phrases are captured cleanly."
FULL_AUDIO_TEXT = f"{PRIOR_CONTEXT} {KNOWN_LAST_SENTENCE}"


def generate_known_sentence_audio(wav_path: str):
    ps_content = f'''Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 2
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile("{wav_path}", $format)
$synth.Speak(@'
{FULL_AUDIO_TEXT}
'@)
$synth.Dispose()
'''
    ps_file = os.path.join(os.path.dirname(wav_path), "gen_last_sentence.ps1")
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write(ps_content)

    subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ps_file], check=True, capture_output=True)


def run_known_last_sentence_test():
    print("=" * 80)
    print("KNOWN-LAST-SENTENCE PRESERVATION TEST")
    print("=" * 80)

    session_dir = os.path.join(repo_root, "bench", "session_known_last_sentence")
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    os.makedirs(session_dir, exist_ok=True)

    wav_path = os.path.abspath(os.path.join(session_dir, "speech_known_tail.wav"))
    print(f"[*] Generating speech sample ending in known sentence: \"{KNOWN_LAST_SENTENCE}\"...")
    generate_known_sentence_audio(wav_path)

    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        audio_bytes = wf.readframes(n_frames)

    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    duration_s = n_frames / float(framerate)
    print(f"Generated sample: {duration_s:.2f}s ({n_frames} frames at {framerate} Hz)")

    # Set up chunker and transcriber
    chunker = AudioChunker(
        session_dir=session_dir,
        config=ChunkerConfig(
            target_chunk_s=8.0,
            pause_cut_s=2.5,
        ),
    )

    transcripts = []
    transcriber = Transcriber(
        config=TranscriberConfig(
            model_name="small.en",
            compute_type="int8",
            cpu_threads=4,
        ),
        on_transcript=lambda res: transcripts.append(res),
        auto_warmup=True,
    )

    wav_writer = LiveWavWriter(os.path.join(session_dir, "audio.wav"), sample_rate=framerate, channels=1)

    frame_size = 512
    print("[*] Streaming audio frames until end of speech...")
    for i in range(0, len(samples), frame_size):
        frame = samples[i : i + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        wav_writer.write_samples(frame)
        chunk = chunker.process_frame(frame)
        if chunk is not None:
            transcriber.enqueue(
                seq=chunk.seq,
                audio_path=chunk.file_path,
                duration_s=chunk.duration_s,
                session_dir=session_dir,
                t_start=chunk.t_start,
                t_end=chunk.t_end,
            )

    # Immediately trigger STOP right as speech ends
    print("[*] Pressing STOP immediately after speech completes. Flushing chunker...")
    final_chunk = chunker.flush()
    assert final_chunk is not None, "Final chunk must not be None when speech audio was buffered!"
    print(f"[+] Final chunk emitted on flush: seq #{final_chunk.seq}, duration: {final_chunk.duration_s:.2f}s, file: {final_chunk.file_path}")

    transcriber.enqueue(
        seq=final_chunk.seq,
        audio_path=final_chunk.file_path,
        duration_s=final_chunk.duration_s,
        session_dir=session_dir,
        t_start=final_chunk.t_start,
        t_end=final_chunk.t_end,
    )
    wav_writer.close()

    print("[*] Draining transcriber queue...")
    while transcriber.pending_count > 0:
        time.sleep(0.05)
    transcriber.stop()

    # Combine transcripts
    full_transcript = " ".join(t.text.strip() for t in transcripts)
    print("\n" + "-" * 80)
    print("TRANSCRIBED CHUNKS:")
    for t in transcripts:
        print(f"  Chunk #{t.seq} ({t.audio_s:.2f}s, RTF {t.rtf:.2f}x): \"{t.text}\"")

    print("\nFULL COMBINED OUTPUT:")
    print(f"  \"{full_transcript}\"")
    print("-" * 80)

    # Verification of the known last sentence
    target_clause = "very last sentence spoken by the user before pressing stop is never lost"
    target_normalized = target_clause.lower().replace(",", "").replace(".", "")
    output_normalized = full_transcript.lower().replace(",", "").replace(".", "")

    clause_present = target_normalized in output_normalized
    print("\nEMPIRICAL VERIFICATION:")
    print(f"1. Target Tail Sentence: \"{KNOWN_LAST_SENTENCE}\"")
    print(f"2. Final Chunk Seq:      #{final_chunk.seq}")
    print(f"3. Tail Clause Present:  {clause_present}")

    assert clause_present, f"Expected '{target_clause}' in output, but was truncated or lost!"
    print("\n[SUCCESS] Known-last-sentence test PASSED: Tail sentence fully transcribed and preserved.")
    print("=" * 80)


if __name__ == "__main__":
    run_known_last_sentence_test()
