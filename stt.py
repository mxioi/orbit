"""Speech-to-text: faster-whisper (CUDA/CPU) as the default, with an
optional whisper.cpp+Vulkan backend for GPUs faster-whisper can't
accelerate at all.

On Windows, importing this module sets up the CUDA/cuDNN DLL PATH before
importing faster_whisper -- must happen before that import, not just
before use, since CTranslate2's internal LoadLibrary calls only respect
PATH, not os.add_dll_directory(). Linux's nvidia-*-cu12 wheels don't need
this: they use standard RPATH-based dynamic linking, not the Windows DLL
search order that made the manual PATH edit necessary in the first place.

Why a second backend at all: CTranslate2 (faster-whisper's engine) only
supports NVIDIA GPUs in its prebuilt wheels -- confirmed via its own
hardware-support docs (Compute Capability >= 3.5, no AMD/Intel mention at
all). That's fine on this project's own Windows/NVIDIA dev machine, but
means AMD (e.g. Strix Halo) and Intel (e.g. Arc) GPUs get zero
acceleration through faster-whisper, full stop -- not a driver/version
problem, a real capability gap. whisper.cpp's Vulkan backend is
vendor-neutral (Vulkan itself doesn't care which vendor), and was
verified live against a real Intel Arc Pro B70: genuine GPU engagement
(ggml_vulkan found the device, whisper_backend_init_gpu confirmed
"using Vulkan0 backend", not a silent CPU fallback), correct
transcription, ~1s for an 11s clip.

This is opt-in, not a new hard dependency: `pywhispercpp` built with
Vulkan needs the Vulkan SDK + glslang/SPIRV-Headers + a C++ toolchain
(GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp),
which is a heavier ask than this project's other pip dependencies. If
it's not installed, or no Vulkan device is present, load_whisper() just
silently skips this path and falls through to CPU -- see
_try_load_vulkan_backend()'s docstring for the exact fallback order.
"""
import ctypes
import os
import sys
import sysconfig
import time

if sys.platform == "win32":
    os.environ["PATH"] = (
        r"C:\Program Files\Python314\Lib\site-packages\nvidia\cublas\bin;"
        r"C:\Program Files\Python314\Lib\site-packages\nvidia\cudnn\bin;"
        r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_nvrtc\bin;"
        r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_runtime\bin;"
    ) + os.environ["PATH"]

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # whisper (and Silero VAD) both want 16kHz mono
WHISPER_CPP_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper_cpp_models")


class _FasterWhisperBackend:
    def __init__(self, model):
        self._model = model

    def transcribe(self, audio):
        segments, info = self._model.transcribe(audio, language="en")
        return " ".join(seg.text.strip() for seg in segments)


class _PyWhisperCppBackend:
    def __init__(self, model):
        self._model = model

    def transcribe(self, audio):
        segments = self._model.transcribe(audio)
        return " ".join(seg.text.strip() for seg in segments)


def _try_load_vulkan_backend():
    """Optional acceleration path -- see module docstring for why this
    exists at all. Returns a _PyWhisperCppBackend on success, or None if
    pywhispercpp isn't installed, has no Vulkan device to use, or fails to
    load for any other reason; load_whisper() treats None as "move on to
    the next option in the chain", not a hard error.

    Not attempted on Windows: CUDA via faster-whisper already covers
    NVIDIA there, and this path has only been verified on Linux (Intel
    Arc B70) so far -- no reason to add an unverified code path on a
    platform that already has a working GPU option.
    """
    if sys.platform == "win32":
        return None

    try:
        # pywhispercpp's Vulkan build installs libwhisper.so.1/libggml*.so
        # as loose files directly in site-packages (not inside the
        # pywhispercpp package directory) -- confirmed live. Unlike
        # Windows (where prepending PATH before importing faster_whisper
        # works, because LoadLibrary re-reads the current process's PATH
        # on every call), Linux's dynamic linker does NOT re-read
        # LD_LIBRARY_PATH from a live-modified os.environ after the
        # process has already started -- confirmed live, setting it here
        # still raised "libwhisper.so.1: cannot open shared object file".
        # It's fixed at exec() time, before this Python process even
        # started, so setting it from inside the process is too late.
        #
        # Fix: explicitly ctypes.CDLL() each dependency by its exact path,
        # in dependency order (confirmed via ldd -- base has no ggml-
        # internal deps; cpu and vulkan both depend on base; the ggml
        # umbrella lib depends on all three; whisper depends on ggml +
        # base), with RTLD_GLOBAL so each is visible for the next one to
        # link against. This sidesteps LD_LIBRARY_PATH entirely -- an
        # explicit-path dlopen() doesn't need it. Once all of these are
        # already loaded in the process, the pywhispercpp import below
        # resolves its own DT_NEEDED entries against them directly
        # instead of searching for them again.
        site_packages = sysconfig.get_path("purelib")
        for lib_name in ("libggml-base.so.0", "libggml-cpu.so.0", "libggml-vulkan.so.0",
                          "libggml.so.0", "libwhisper.so.1"):
            lib_path = os.path.join(site_packages, lib_name)
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        from pywhispercpp.model import Model as WhisperCppModel
    except (ImportError, OSError):
        return None

    try:
        t0 = time.time()
        model = WhisperCppModel("base.en", models_dir=WHISPER_CPP_MODELS_DIR)
        silence = np.zeros(SAMPLE_RATE, dtype="float32")
        model.transcribe(silence)
        print(f"Loaded+warmed whisper base.en via whisper.cpp/Vulkan in {time.time()-t0:.1f}s")
        return _PyWhisperCppBackend(model)
    except Exception as e:
        print(f"Failed to load whisper.cpp/Vulkan backend: {e!r}")
        return None


def load_whisper():
    # Order: CUDA via faster-whisper (fastest, NVIDIA-only, proven) ->
    # Vulkan via whisper.cpp (AMD/Intel/NVIDIA, opt-in, see module
    # docstring) -> CPU via faster-whisper (always available, the
    # existing proven fallback).
    try:
        t0 = time.time()
        model = WhisperModel("base.en", device="cuda", compute_type="int8")
        silence = np.zeros(SAMPLE_RATE, dtype="float32")
        list(model.transcribe(silence, language="en")[0])
        print(f"Loaded+warmed whisper base.en on cuda/int8 in {time.time()-t0:.1f}s")
        return _FasterWhisperBackend(model)
    except Exception as e:
        print(f"Failed to load on cuda: {e}")

    vulkan_backend = _try_load_vulkan_backend()
    if vulkan_backend is not None:
        return vulkan_backend

    try:
        t0 = time.time()
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        silence = np.zeros(SAMPLE_RATE, dtype="float32")
        list(model.transcribe(silence, language="en")[0])
        print(f"Loaded+warmed whisper base.en on cpu/int8 in {time.time()-t0:.1f}s")
        return _FasterWhisperBackend(model)
    except Exception as e:
        print(f"Failed to load on cpu: {e}")

    raise SystemExit("Could not load whisper on cuda, vulkan, or cpu")


def transcribe(model, audio):
    return model.transcribe(audio)
