"""Silero VAD (voice activity detection), ONNX-only -- no torch/torchaudio.

The official `silero-vad` pip package hard-requires torch+torchaudio even
for its "onnx" extra (they're base deps, not optional) -- avoided that by
extracting just the .onnx model file and driving it directly through the
onnxruntime we already depend on for Whisper/Piper.

Model contract (confirmed against the actual file's I/O tensors AND the
official reference implementation, snakers4/silero-vad's utils_vad.py --
sess.get_inputs()/outputs() alone weren't enough, see below): stateful,
processes 512-sample chunks at 16kHz, carries a [2, 1, 128] hidden state
between calls, PLUS a 64-sample "context" window (the tail of the
previous chunk) that must be prepended to each new chunk before it's fed
to the model -- so the real "input" tensor is 576 samples (64 context +
512 new), not a bare 512. Missing this context-priming was a real,
confirmed bug: without it, the model's output stayed flatlined near
0.001 regardless of actual audio content (verified live against real
speech, not just synthetic TTS) -- the model was effectively seeing a
truncated/misaligned window on every single call.
"""
import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # ~32ms at 16kHz -- Silero's required chunk size
CONTEXT_SAMPLES = 64  # tail of the previous chunk, prepended to each new one at 16kHz


class SileroVAD:
    def __init__(self, model_path="vad_model/silero_vad.onnx"):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """chunk: float32 array of exactly CHUNK_SAMPLES samples in [-1, 1].
        Returns speech probability 0..1. Carries hidden state AND context
        internally -- call reset() when starting a fresh listening
        session/after a long gap."""
        if len(chunk) != CHUNK_SAMPLES:
            raise ValueError(f"expected {CHUNK_SAMPLES} samples, got {len(chunk)}")
        x = np.concatenate([self._context, chunk])
        inputs = {
            "input": x.reshape(1, -1),
            "state": self._state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }
        prob, self._state = self.sess.run(["output", "stateN"], inputs)
        self._context = x[-CONTEXT_SAMPLES:]
        return float(prob[0][0])


class UtteranceDetector:
    """Turns a stream of per-chunk speech probabilities into
    start-of-speech / end-of-speech events, with hysteresis so a couple of
    noisy frames don't false-trigger either transition.

    speech_frames_to_start: consecutive above-threshold chunks needed to
        confirm "you started talking" (default 3 * 32ms ~= 96ms).
    silence_frames_to_end: consecutive below-threshold chunks needed to
        confirm "you stopped talking" (default ~25 * 32ms ~= 800ms --
        long enough to survive a normal mid-sentence pause).
    """

    def __init__(self, threshold=0.5, speech_frames_to_start=3, silence_frames_to_end=25):
        self.threshold = threshold
        self.speech_frames_to_start = speech_frames_to_start
        self.silence_frames_to_end = silence_frames_to_end
        self._speech_run = 0
        self._silence_run = 0
        self.in_speech = False

    def update(self, prob: float) -> str | None:
        """Feed one chunk's probability. Returns "start", "end", or None."""
        is_speech = prob >= self.threshold
        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if not self.in_speech and self._speech_run >= self.speech_frames_to_start:
            self.in_speech = True
            return "start"
        if self.in_speech and self._silence_run >= self.silence_frames_to_end:
            self.in_speech = False
            return "end"
        return None
