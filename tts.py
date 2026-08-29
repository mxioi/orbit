"""Text-to-speech: Piper (VITS-based), not Kokoro (StyleTTS2-based). Kokoro's
GPU path was a confirmed dead end on this card's Pascal generation (Quadro
P500, no tensor cores) -- "CUBLAS failure 8: the function requires an
architectural feature absent from the device" -- and even on CPU it ran at
~2.3x real-time (too slow for a voice assistant). Piper's simpler
architecture measured RTF 0.09x CPU-only on the exact same hardware and
text -- ~25x faster than Kokoro, no GPU needed at all.
"""
import re
import time

import numpy as np
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig

# Piper's default comma pause reads as too short/ignorable (a property of
# this voice model's training, not exposed as a tunable pause-duration
# param). Fixed by ear, A/B tested live: length_scale=1.2 slows the whole
# utterance ~20% (pauses scale with it too), combined with swapping commas
# for semicolons (a stronger pause point for this phonemizer). Confirmed
# on both a synthetic multi-comma sentence and a natural multi-sentence
# passage -- user's call: "better than being too fast, that's for sure."
PIPER_SYNTHESIS_CONFIG = SynthesisConfig(length_scale=1.2)
PIPER_MODEL = "piper_models/en_US-lessac-medium.onnx"


def load_piper():
    t0 = time.time()
    voice = PiperVoice.load(PIPER_MODEL)
    list(voice.synthesize("warmup."))  # first call pays a one-time cost too
    print(f"Loaded+warmed Piper in {time.time()-t0:.1f}s")
    return voice


def strip_markdown_for_speech(text):
    """Turnstone's responses are formatted for a text chat UI (bold,
    blockquotes, bullets, headers) -- Piper otherwise tries to pronounce
    those symbols literally. Light regex strip, not a full markdown
    parser; good enough for LLM-typical output."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)  # italic
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)  # blockquotes
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # bullet lists
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # numbered lists
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](links)

    # Piper's phonemizer (espeak-ng) only splits/pauses on sentence
    # punctuation, not bare newlines -- markdown headers and list items
    # commonly have no terminal punctuation once their marker is stripped,
    # so they run straight into the next line with no pause at all.
    # Confirmed live: a 5-section bulleted breakdown read as one unbroken
    # run-on. Give every non-empty line real terminal punctuation.
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not re.search(r"[.!?:;,]$", line):
            line += "."
        lines.append(line)
    text = "\n".join(lines)

    text = re.sub(r"[ \t]+", " ", text)

    # Piper's comma pause reads as too short/near-ignorable on this voice
    # (a training-data property, not a tunable param) -- semicolons get a
    # more pronounced pause from the same phonemizer. A/B tested live
    # against length_scale alone and combined; user's call.
    text = text.replace(",", ";")

    return text.strip()


def play_ack_chime():
    """Instant audio feedback the moment VAD decides you've stopped talking
    -- fires before STT/LLM latency even starts, so there's *something*
    immediately in response to speaking rather than dead air until a reply
    eventually arrives. Synthesized on the fly (no asset file to ship/load);
    two quick soft ascending tones with a raised-cosine envelope on each to
    avoid audible clicks at the start/end of a tone. sd.play() without a
    following sd.wait() is non-blocking -- the caller (mic_loop) continues
    immediately without waiting for this ~120ms sound to finish."""
    sr = 22050

    def tone(freq, dur):
        n = int(sr * dur)
        t = np.arange(n) / sr
        envelope = np.sin(np.pi * t / dur) ** 2
        return (np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)

    gap = np.zeros(int(sr * 0.015), dtype=np.float32)
    chime = np.concatenate([tone(660, 0.045), gap, tone(880, 0.06)]) * 0.25
    sd.play(chime, sr)


def speak(piper_voice, text, on_amplitude=None, stop_event=None):
    """Synthesize and play text. If on_amplitude is given, it's called
    repeatedly during playback with a 0..1 loudness estimate (for driving
    the avatar's speech-reactive swell) and once more with 0.0 when done.

    If stop_event is given (a threading.Event), playback checks it before
    writing each ~80ms block and cuts immediately via stream.abort() if
    it's set -- for barge-in, where "immediately" matters (stream.stop()
    would instead let whatever's already buffered on the audio device
    finish playing out first, not the instant cutoff barge-in needs).
    Returns True if playback completed normally, False if it was stopped
    early via stop_event."""
    clean = strip_markdown_for_speech(text)
    chunks = list(piper_voice.synthesize(clean, PIPER_SYNTHESIS_CONFIG))
    if not chunks:
        return True
    audio = np.concatenate([c.audio_int16_array for c in chunks])
    sample_rate = chunks[0].sample_rate

    if on_amplitude is None and stop_event is None:
        sd.play(audio, sample_rate)
        sd.wait()
        return True

    # Stream in small blocks (~80ms) instead of one sd.play() call so the
    # avatar can react to the audio's actual loudness envelope as it plays
    # (rather than jumping straight to a static "speaking" pose), and so
    # stop_event gets checked often enough for barge-in to feel immediate.
    block = max(1, int(sample_rate * 0.08))
    completed = True
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16") as stream:
        for i in range(0, len(audio), block):
            if stop_event is not None and stop_event.is_set():
                stream.abort()  # discard whatever's still buffered -- immediate silence
                completed = False
                break
            seg = audio[i:i + block]
            rms = float(np.sqrt(np.mean((seg.astype(np.float32) / 32768.0) ** 2)))
            # Empirical scale, not audibly verified (can't hear it myself) --
            # Piper's typical RMS peaks well under full-scale, so a flat
            # 1:1 mapping read as barely-moving; 4x gave visible swell in
            # the earlier avatar amplitude tests. Retune by ear if it looks
            # under/over-reactive once actually watched during real speech.
            if on_amplitude:
                on_amplitude(min(1.0, rms * 4.0))
            stream.write(seg)
    if on_amplitude:
        on_amplitude(0.0)
    return completed
