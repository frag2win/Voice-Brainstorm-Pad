import os
import subprocess
import wave

REF_TEXT_PARAGRAPHS = [
    "Welcome to the local voice brainstorm pad benchmark. Today we are stress testing real-time transcription on local CPU hardware without dedicated graphics acceleration. Our objective is to evaluate whether the faster whisper small English model running in integer eight precision can sustain high throughput during continuous fast dictation.",
    "When a user brainstorms ideas out loud, they rarely pause for full sentence boundaries. They speak in rapid streams of consciousness, jumping between concepts, refining thoughts, and uttering hundreds of words per minute. Traditional dictation systems break down under this pace because they assume frequent pauses or they introduce runaway latency that causes backlog accumulation.",
    "In our architecture, the audio stream is continuously written to a linear wave file while an energy and neural voice activity detection chunker isolates segments between eight and fifteen seconds. If a natural pause occurs after eight seconds, the chunker cuts immediately. If the user continues speaking without a pause, the system searches for the quietest frame within the trailing window to perform a clean cut.",
    "Every emitted chunk is delivered through a sequential queue to the transcriber worker. The worker evaluates the audio, applies initial prompt conditioning from previous context, logs the real time factor, and records the end to end latency from the moment the user started speaking until the text was produced.",
    "In addition to real time factor, queue depth is the primary indicator of system health. If the real time factor exceeds one point zero, chunks accumulate in memory and lag grows without bound. On a quad core processor with four compute threads, the small English model must achieve an average real time factor between point three and point six.",
    "During dictation stop events, the sidecar must enter a catching up state rather than blocking the main event loop. It continues to respond to websocket pings while sequentially draining all pending chunks. Once the queue depth drops to zero, the state safely transitions to idle without any crash or process restart.",
    "Now let us examine the text editor integration. The renderer maintains an insertion anchor point at the user's cursor position. As transcription events arrive over the websocket, text is formatted with intelligent punctuation, leading capitalization, and single space separation before being inserted into the CodeMirror document.",
    "If the user moves the cursor or continues typing manually during a pause, the anchor updates gracefully. Dictation never overwrites existing text or moves backward without explicit user interaction. In our testing, user typing while the microphone is silent remains smooth, responsive, and completely lag free.",
    "Memory management on consumer laptops is equally critical. In previous iterations, unconstrained thread creation and unbounded temperature fallback search arrays caused Intel Math Kernel Library heap allocation failures. We introduced strict thread limits, capped temperature candidate lists, and automatic memory garbage collection retries to eliminate crashes.",
    "We also verify that editor keystrokes and background room noise during quiet intervals do not trigger spurious hallucinations. The voice activity detector discards frames below the energy floor and prevents accidental speech triggers.",
    "To validate boundary accuracy between consecutive chunks, initial prompt conditioning passes the tail of the preceding chunk transcript to the whisper decoder. If the decoder repeats the prompt tail, a normalized prefix stripper removes redundant words to prevent duplicate phrases in the document.",
    "In addition to real-time transcription, session persistence and crash resilience are core functional requirements. Raw audio is saved as a linear wave file with live header patching every second. If an unexpected power outage occurs, reopening the application immediately restores the audio recording and existing transcripts without manual intervention.",
    "When brainstorming at three hundred to four hundred words per minute, speech is continuous, fluid, and dense with technical concepts. Under this sustained load, the system must process every chunk with a real time factor strictly below one point zero, keeping the processing queue drained and latency bounded.",
    "In summary, continuous brainstorming requires rock-solid architecture from audio capture through chunk boundary detection and neural inference. By measuring real-time factor, p95 latency, maximum queue depth, and word error rate under continuous fast speech, we establish that the system keeps up with fast-talking users on local hardware.",
    "Every milestone in iteration four is backed by empirical measurements directly on this workstation.",
    "Finally, we verify that the very last sentence spoken by the user before pressing stop is never lost. The audio chunker flushes any remaining speech samples immediately upon stop, enqueuing the final chunk for transcription and guaranteeing complete audit trail preservation.",
    "This concludes the five minute full speed benchmarking audio script. All systems are operating normally and ready for empirical validation."
]

FULL_REF_TEXT = " ".join(REF_TEXT_PARAGRAPHS)

def generate_audio():
    os.makedirs("bench", exist_ok=True)
    ref_txt_path = os.path.join("bench", "speech_5min_ref.txt")
    with open(ref_txt_path, "w", encoding="utf-8") as f:
        f.write(FULL_REF_TEXT)
    print(f"Wrote reference text: {ref_txt_path} ({len(FULL_REF_TEXT.split())} words)")

    # Write temporary powershell script to generate the audio using SAPI
    ps_script_path = os.path.join("bench", "gen_5min.ps1")
    wav_path = os.path.abspath(os.path.join("bench", "speech_5min.wav"))
    
    # We escape double quotes for PowerShell
    escaped_text = FULL_REF_TEXT.replace('"', '`"')
    
    ps_content = f'''Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 2
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile("{wav_path}", $format)
$synth.Speak(@'
{FULL_REF_TEXT}
'@)
$synth.Dispose()
Write-Output "Generated: {wav_path}"
'''
    with open(ps_script_path, "w", encoding="utf-8") as f:
        f.write(ps_content)

    print("Synthesizing benchmark speech audio via SAPI...")
    res = subprocess.run(["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", ps_script_path], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    if os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as wf:
            dur = wf.getnframes() / float(wf.getframerate())
            print(f"Generated WAV duration: {dur:.2f} seconds ({dur/60.0:.2f} minutes), {wf.getnframes()} frames, {wf.getframerate()} Hz")

if __name__ == "__main__":
    generate_audio()
