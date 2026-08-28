"""Speech-to-text: faster-whisper, CUDA if available, CPU fallback.

Importing this module sets up the CUDA/cuDNN DLL PATH before importing
faster_whisper -- must happen before that import, not just before use,
since CTranslate2's internal LoadLibrary calls only respect PATH, not
os.add_dll_directory().
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

SAMPLE_RATE = 16000  # whisper (and Silero VAD) both want 16kHz mono


def load_whisper():
    for device, compute_type in [("cuda", "int8"), ("cpu", "int8")]:
        try:
            t0 = time.time()
            model = WhisperModel("base.en", device=device, compute_type=compute_type)
            silence = np.zeros(16000, dtype="float32")
            list(model.transcribe(silence, language="en")[0])
            print(f"Loaded+warmed whisper base.en on {device}/{compute_type} "
                  f"in {time.time()-t0:.1f}s")
            return model
        except Exception as e:
            print(f"Failed to load on {device}: {e}")
    raise SystemExit("Could not load whisper on cuda or cpu")


def transcribe(model, audio):
    segments, info = model.transcribe(audio, language="en")
    return " ".join(seg.text.strip() for seg in segments)
