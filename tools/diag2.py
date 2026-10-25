import sys, time, wave, numpy as np, psutil
from faster_whisper import WhisperModel

CHUNK, AUDIO = sys.argv[1], sys.argv[2]
avail = lambda: round(psutil.virtual_memory().available / 2**30, 2)
m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)

print("== A: failing chunk, OLD settings (full fallback ladder), 10 repeats")
for i in range(10):
    t = time.time()
    try:
        segs, _ = m.transcribe(CHUNK, language="en", beam_size=1,
                               condition_on_previous_text=False,
                               temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), vad_filter=False)
        segs = list(segs)
        print(i, f"{time.time()-t:.2f}s", "avail GB", avail(),
              "max seg temp", max((s.temperature for s in segs), default=None))
    except Exception as e:
        print(i, "FAILED:", repr(e), "avail GB", avail())

print("== B: transcribe time vs audio length (warm)")
w = wave.open(AUDIO); sr = w.getframerate()
raw = np.frombuffer(w.readframes(sr * 40), dtype=np.int16).astype(np.float32) / 32768
list(m.transcribe(raw[: sr * 5], language="en", beam_size=1, vad_filter=False)[0])  # warm-up
for sec in (4, 8, 15, 25):
    t = time.time()
    segs, _ = m.transcribe(raw[: sr * sec], language="en", beam_size=1,
                           condition_on_previous_text=False, vad_filter=False)
    words = len(" ".join(s.text for s in segs).split())
    d = time.time() - t
    print(f"{sec:>2}s audio -> {d:.2f}s  RTF {d/sec:.2f}  words {words}")
