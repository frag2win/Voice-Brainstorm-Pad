# Local Whisper CPU Benchmark

A standalone Python benchmark tool (no UI, no Electron) designed to measure whether local Whisper models can keep up with fast continuous speech on a Windows laptop using **CPU only** (4 cores / 8 threads, no NVIDIA GPU).

---

## Benchmark Specifications

- **Models benchmarked**: `base.en`, `distil-small.en`, `small.en` (customizable via `--models`)
- **Inference engine**: `faster-whisper`
- **CPU Settings**: `compute_type="int8"`, `cpu_threads=4`, `num_workers=1`, `language="en"`, `beam_size=1`, `condition_on_previous_text=False`, `vad_filter=False`
- **Audio Capture**: 16 kHz mono 16-bit PCM recorded directly from the default microphone via `sounddevice` and saved to `bench/sample.wav`
- **Parity Mode**: Option `--file` allows reusing any existing WAV file so all models process identical audio
- **Warm-up**: A 5-second warm-up transcription is executed for each model and strictly excluded from benchmark timings
- **CPU Monitoring**: Automatically samples CPU clock frequency (MHz) and utilization (%) every 2 seconds via `psutil`
- **Results Output**: Displays a CLI summary table and writes a detailed markdown report with transcripts to `bench/results.md`
- **Offline Operation**: Zero network activity after initial model download (cached locally)

---

## Directory Structure

```text
bench/
├── bench.py          # Standalone benchmark script
├── requirements.txt  # Pinned dependencies (faster-whisper, sounddevice, numpy, psutil)
├── results.md        # Benchmark output table and transcripts
├── sample.wav        # Recorded audio sample (created upon first run)
└── README.md         # Instructions and documentation
```

---

## Setup Instructions

### Prerequisites
- Windows 10 or 11
- Python 3.11+ installed and accessible on PATH

### Option A: Using Standard Python Virtual Environment (`venv`)

Open PowerShell in the `bench` directory:

```powershell
cd bench

# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (If PowerShell blocks script execution, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)

# 3. Install pinned dependencies
pip install -r requirements.txt
```

### Option B: Using `uv`

If you have `uv` installed:

```powershell
cd bench

# 1. Create virtual environment
uv venv

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
uv pip install -r requirements.txt
```

---

## Running the Benchmark

### 1. Preparation for Accurate Results
1. Plug your laptop into the charger.
2. Set Windows Power Mode to **Best performance**:
   - *Windows Settings* > *System* > *Power & battery* > *Power mode* &rarr; **Best performance**.
3. Ensure microphone input is unmuted.

### 2. Standard 60-Second Live Mic Benchmark
Run the benchmark while speaking continuously at your normal brainstorming pace:

```powershell
python bench.py --seconds 60
```

1. The script gives a 3-second countdown.
2. Speak continuously for 60 seconds (a live progress bar tracks elapsed time).
3. Audio is automatically saved to `bench/sample.wav`.
4. Each model (`base.en`, `distil-small.en`, `small.en`) will load, warm up for 5 seconds, transcribe the sample, log CPU MHz/utilization, and print the results table.

### 3. Reusing the Sample Audio (Identical Input Benchmark)
To re-test models without recording new audio, pass `--file`:

```powershell
python bench.py --file sample.wav
```

### 4. Customizing Models & Threading
You can specify custom models or thread counts:

```powershell
# Benchmark only distil-small.en with 4 threads
python bench.py --file sample.wav --models distil-small.en

# Benchmark with custom thread allocation
python bench.py --file sample.wav --models base.en distil-small.en --cpu-threads 4
```

### 5. CLI Options Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--seconds` | `60` | Duration of microphone recording in seconds |
| `--file` | `None` | Path to existing audio file (skips mic recording) |
| `--models` | `base.en distil-small.en small.en` | Space-separated list of faster-whisper models |
| `--cpu-threads` | `4` | Number of CPU inference threads |
| `--compute-type` | `int8` | Quantization type (e.g. `int8`, `float32`) |
| `--no-cpu-log` | `False` | Disables 2-second background CPU MHz logging |
| `--sample-path` | `bench/sample.wav` | Destination path for recorded audio file |
| `--results-path` | `bench/results.md` | Destination path for markdown output report |

---

## Understanding the Output Metrics

- **Real-Time Factor (RTF)**:
  $$\text{RTF} = \frac{\text{Transcribe Time (seconds)}}{\text{Audio Duration (seconds)}}$$
  - **$\text{RTF} < 1.0$ (e.g., $0.85\times$) &rarr; YES (Real-Time)**: The model transcribes audio faster than the speaker talks. It can keep up with live speech without accumulating latency.
  - **$\text{RTF} \ge 1.0$ (e.g., $1.33\times$) &rarr; NO (Lagging)**: The model takes longer to transcribe than the audio duration. Latency will continuously grow during extended speech sessions.
- **Spoken WPM**: Calculated as $\frac{\text{Word Count}}{\text{Audio Duration in Minutes}}$. Typical brainstorming speech ranges between 130 and 190 WPM.
- **CPU MHz (Min / Avg)**: Measures whether CPU throttling or clock down-scaling occurs under sustained load.
