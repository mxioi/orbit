"""Realistic CPU vs GPU benchmark for the STT step.

Simulates a warm, already-running voice service: load the model once
(discarding that cost, same as a real service would only pay it at
startup), then transcribe several realistic synthetic utterances of
varying length back to back, timing each individually.

Utterances are real synthesized speech (via Kokoro), not silence/noise,
so decoding behaves like it would for an actual voice command.
"""
import os
import time

os.environ["PATH"] = (
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cublas\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cudnn\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_nvrtc\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_runtime\bin;"
) + os.environ["PATH"]

import numpy as np
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

UTTERANCES = [
    "Hey, what's the status of the backup job?",
    "Can you check if the turnstone console is still running and tell me if anything looks wrong?",
    "Archive that security alert email and mark it as read.",
    "What's the weather looking like for the rest of the week in the homelab server room, temperature-wise?",
    "Turn off the lights.",
]

CONFIGS = [
    ("cpu", "int8"),
    ("cuda", "int8"),
    ("cuda", "float32"),
]


def synthesize_utterances(kokoro):
    """Generate real speech audio for each test utterance, resampled to
    16kHz mono float32 (what whisper expects)."""
    import scipy.signal  # only needed here, for resampling 24kHz -> 16kHz

    clips = []
    for text in UTTERANCES:
        audio, sr = kokoro.create(text, voice="af_heart")
        n_out = int(len(audio) * 16000 / sr)
        resampled = scipy.signal.resample(audio, n_out).astype(np.float32)
        clips.append((text, resampled, len(resampled) / 16000))
    return clips


def benchmark_device(device, compute_type, clips):
    print(f"\n=== {device}/{compute_type} ===")
    t0 = time.time()
    model = WhisperModel("base.en", device=device, compute_type=compute_type)
    # Warmup: run once on a short throwaway clip, discard timing entirely.
    # This is the cost a real service pays exactly once at startup.
    warmup_audio = clips[0][1]
    list(model.transcribe(warmup_audio, language="en")[0])
    warmup_total = time.time() - t0
    print(f"model load + warmup: {warmup_total:.2f}s (one-time cost)")

    results = []
    for text, audio, duration_s in clips:
        t0 = time.time()
        segments, _ = model.transcribe(audio, language="en")
        transcript = " ".join(s.text.strip() for s in segments)
        dt = time.time() - t0
        rtf = dt / duration_s
        results.append((duration_s, dt, rtf, transcript))
        print(f"  {duration_s:5.1f}s audio -> {dt:5.2f}s transcribe (RTF {rtf:.2f}x)  "
              f"| {transcript!r}")

    avg_rtf = sum(r[2] for r in results) / len(results)
    print(f"  average RTF: {avg_rtf:.2f}x (lower is better; 1.0x = real-time)")
    return warmup_total, avg_rtf


def main():
    print("Synthesizing realistic test utterances with Kokoro...")
    kokoro = Kokoro("models/kokoro-v1.0.int8.onnx", "models/voices-v1.0.bin")
    clips = synthesize_utterances(kokoro)
    for text, audio, dur in clips:
        print(f"  [{dur:.1f}s] {text!r}")

    summary = []
    for device, compute_type in CONFIGS:
        try:
            warmup, avg_rtf = benchmark_device(device, compute_type, clips)
            summary.append((device, compute_type, warmup, avg_rtf))
        except Exception as e:
            print(f"{device}/{compute_type}: FAILED - {e}")
            summary.append((device, compute_type, None, None))

    print("\n=== Summary (steady-state, after one-time warmup) ===")
    print(f"{'device/type':<18} {'warmup (once)':<16} {'avg RTF':<10}")
    for device, ct, warmup, avg_rtf in summary:
        w = f"{warmup:.2f}s" if warmup is not None else "FAILED"
        r = f"{avg_rtf:.2f}x" if avg_rtf is not None else "-"
        print(f"{device+'/'+ct:<18} {w:<16} {r:<10}")


if __name__ == "__main__":
    main()
