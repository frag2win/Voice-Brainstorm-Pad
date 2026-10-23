# Local Whisper Benchmark Results

**Date**: 2026-10-23 18:16:47  
**Audio File**: `sample.wav` (60.00s)  
**Compute**: CPU only (`int8`, `cpu_threads=4`, `num_workers=1`, `beam_size=1`)

## System Environment

- **OS**: Windows 11 (10.0.26300)
- **CPU**: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- **Topology**: 4 physical cores / 8 logical threads
- **Memory**: 7.8 GB
- **Python**: 3.13.5

## Performance Comparison Table

| Model | Load Time (s) | Audio (s) | Transcribe (s) | RTF (Transcribe/Audio) | Keeps Up? | Word Count | Spoken WPM | Min CPU MHz | Avg CPU MHz | Avg CPU % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base.en` | 1.55 | 60.00 | 3.25 | **0.05x** | **YES** (Real-Time) | 138 | 138.0 | 2419 | 2419 | 32.7% |
| `distil-small.en` | 1.65 | 60.00 | 7.65 | **0.13x** | **YES** (Real-Time) | 142 | 142.0 | 2419 | 2419 | 47.1% |
| `small.en` | 1.64 | 60.00 | 9.62 | **0.16x** | **YES** (Real-Time) | 131 | 131.0 | 2419 | 2419 | 47.1% |

> **Note**: **Real-Time Factor (RTF)** = Transcribe Time / Audio Duration. An RTF of 0.25x means the model processes audio 4x faster than real-time.

## Full Transcripts

### `base.en`

> four years into a service it doesn't always seem to be fun is this a book? yeah i'm looking at something which is generally very good make sure to invest this is a text script make sure everything is working make sure we can make things happen and yeah that's all i mean yeah today was too much of a good day you can see but also a bad day too exhausting I'm exhausting the sense that it was submission day for the end sim and I have general miss and wish submissions, submissions are done and completed. Now I have to work for the practical exams which are on 27 which are again going to be hacked in because I mean 10 experiments for every practical exam and so I'm not going So yeah, that just shows

### `distil-small.en`

> 4 years of tool service. It is probably a good form. Hello. Is this a good job? Yeah. I'm looking at something which is generally very good. Make sure you invest. This is a text script. Make sure everything is working. Make sure we can make things happen. And yeah, that's all. No, I mean, yeah. Today was a... too much of a good day, you can say, but also a bad day. Too exhausting. Uh... Exhausting in the sense that it was submission day for the M-Sem, and I have done some missions, submissions are done, and completed. Now I have to work for the practical exams, which I'm 27th, which... exams which are on 27th which again going to be hacked it because I mean 10 experiments for every practical exam and semitulously and continuously so yeah that just shows

### `small.en`

> 4 years in tool service it doesn't always seem to work Hello, is this work? Yeah, I'm looking at something which is generally very good make sure you invent this as a text script make sure everything is working make sure we can make things happen and yeah, that's all I mean, yeah today was a too much of a good day you can say but also a bad day too exhausting uh, exhausting in the sense that it was submission day for the MSim, and I have done all the submissions, submissions are done and completed. Now I have to work for the practical exams which are in 27th which are again going to be hacked at because I mean 10 experiments for every practical exam and semitulacy and continuously.
