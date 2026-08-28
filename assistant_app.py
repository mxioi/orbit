"""GUI voice assistant: floating avatar window + VAD-driven listening (no
push-to-talk) driving the same STT -> Turnstone -> TTS pipeline as
voice_test.py.

Barge-in (interrupting TTS playback mid-speech) is explicitly out of scope
for this phase -- while the assistant is THINKING or SPEAKING, incoming mic
audio is simply discarded, same as push-to-talk always effectively did by
not listening between turns. That's the next phase, once this one is
confirmed solid.
"""
import ctypes
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import sounddevice as sd
import webview

import stt
import tts
import turnstone_client as tc
from vad import SAMPLE_RATE, CHUNK_SAMPLES, SileroVAD, UtteranceDetector

_HERE = os.path.dirname(os.path.abspath(__file__))
AVATAR_HTML = os.path.join(_HERE, "web", "avatar.html")
VAD_MODEL_PATH = os.path.join(_HERE, "vad_model", "silero_vad.onnx")

# Mirror web/avatar.html's PALETTE keys exactly -- these strings are passed
# straight through to setState() on the JS side.
IDLE, LISTENING, RECORDING, THINKING, SPEAKING = "idle", "listening", "recording", "thinking", "speaking"

# UtteranceDetector's silence_frames_to_end (~800ms) is deliberately long so
# a mid-sentence pause doesn't cut you off -- which means ~800ms of
# near-silence is sitting at the tail of the buffer when "end" fires. Trim
# most of it back off before handing audio to Whisper.
TRAIL_TRIM_S = 0.5


class AssistantState:
    """Single mutable owner of everything the mic thread and the pipeline
    thread both touch. Plain attributes, not locks -- by construction only
    one of those two threads is ever "active" at a time (the mic thread
    stops acting on `state` the moment it flips to THINKING/SPEAKING, and
    only the pipeline thread changes state during that stretch), so
    Python's GIL is enough here without real synchronization."""

    def __init__(self):
        self.window = None  # attached right after webview.create_window()
        self.state = IDLE
        self.muted = False
        self.ws_id = None
        self.shutdown = threading.Event()

    def set_state(self, new_state):
        self.state = new_state
        try:
            self.window.evaluate_js(f"setState({new_state!r})")
        except Exception as e:
            print(f"[avatar] setState failed: {e}")

    def set_amplitude(self, level):
        try:
            self.window.evaluate_js(f"setAmplitude({level:.3f})")
        except Exception:
            pass  # fires often during playback -- don't spam errors over it

    def set_muted(self, is_muted):
        self.muted = is_muted


class Api:
    """Exposed to the page as window.pywebview.api -- avatar.html's mute
    button already calls window.pywebview.api.toggle_mute(next) directly
    and updates its own visual state locally; this just records the flag
    so mic_loop knows to stop listening.

    Stores app_state as `_app_state` (leading underscore), not `app_state`
    -- pywebview's js_api introspection (util.py get_functions) recursively
    walks every non-callable, non-underscore attribute of the exposed
    object to discover bindable methods, with no cycle detection. app_state
    holds a reference to the pywebview Window itself (`.window`), so a
    plain `self.app_state` attribute here sent that walk straight into the
    Window's native WinForms control and from there into .NET's
    AccessibilityObject graph, which is self-referential (Rectangle.Empty
    is itself a Rectangle with its own .Empty) -- infinite recursion,
    confirmed live: it hung the whole process (RecursionError spam,
    Whisper's background-thread load never completed) until this rename."""

    def __init__(self, app_state):
        self._app_state = app_state

    def toggle_mute(self, is_muted):
        print(f"[mic] {'muted' if is_muted else 'unmuted'}")
        self._app_state.set_muted(is_muted)

    def start_drag(self):
        """Called from avatar.html on mousedown in #drag-region. pywebview
        6.2.1's Windows backend (edgechromium.py + winforms.py) has NO
        built-in frameless-window dragging at all -- confirmed by reading
        both files, neither references easy_drag or any drag handling;
        only the legacy mshtml.py, plus qt.py/gtk.py/cocoa.py on other
        platforms, implement it. This is the standard Win32 workaround:
        releasing mouse capture and telling Windows the click landed on
        the (nonexistent) title bar makes the OS itself drive the whole
        drag natively -- smooth, correct, and exactly what mshtml.py's own
        easy_drag does under the hood (WebBrowserEx.ReleaseCapture() +
        WM_NCLBUTTONDOWN/HTCAPTION), just reimplemented here since
        edgechromium.py doesn't wire that up for WebView2."""
        hwnd = self._app_state.window.native.Handle.ToInt32()
        WM_NCLBUTTONDOWN = 0x00A1
        HTCAPTION = 2
        ctypes.windll.user32.ReleaseCapture()
        ctypes.windll.user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0)

    def minimize(self):
        self._app_state.window.minimize()


def process_utterance(app_state, whisper_model, piper_voice, audio):
    # mic_loop calls this directly on its own thread with no surrounding
    # try/except -- an uncaught exception here (a Turnstone network
    # hiccup, a transient Whisper/Piper error) would otherwise kill that
    # background thread silently, leaving the avatar stuck showing
    # THINKING/SPEAKING forever with no crash to even notice. Catch
    # broadly and always fall back to LISTENING so one bad turn doesn't
    # require restarting the app.
    try:
        app_state.set_state(THINKING)
        text = stt.transcribe(whisper_model, audio)
        print(f"You said: {text!r}")
        if not text.strip():
            return

        if app_state.ws_id is None:
            app_state.ws_id, response = tc.create_conversation(model=tc.TURNSTONE_MODEL, first_message=text)
            print(f"[turnstone] ws_id={app_state.ws_id}")
        else:
            response = tc.ask(app_state.ws_id, text)
        print(f"Turnstone: {response!r}")

        if response.strip():
            app_state.set_state(SPEAKING)
            tts.speak(piper_voice, response, on_amplitude=app_state.set_amplitude)
    except Exception as e:
        print(f"[process_utterance] turn failed, recovering to LISTENING: {e!r}")
    finally:
        app_state.set_state(LISTENING)


def mic_loop(app_state, whisper_model, piper_voice):
    vad = SileroVAD(model_path=VAD_MODEL_PATH)
    detector = UtteranceDetector()
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        # Keep the realtime audio callback cheap -- just hand the chunk
        # off, no VAD/state work here. Drop audio entirely while muted or
        # mid-turn so the queue doesn't quietly grow for the seconds a
        # Turnstone round-trip + TTS playback can take -- we're discarding
        # it anyway per the no-barge-in-yet policy in the module docstring.
        if app_state.muted or app_state.state in (THINKING, SPEAKING):
            return
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK_SAMPLES, callback=callback,
    )
    stream.start()
    try:
        dev = sd.query_devices(stream.device, "input")
        print(f"[mic] input device: {dev['name']!r} (default samplerate {dev['default_samplerate']})")
    except Exception as e:
        print(f"[mic] could not query input device: {e!r}")
    print("Listening (VAD-driven -- just start talking, no key to hold)...")

    # Rolling pre-roll so the few chunks Silero needs to *confirm* speech
    # started aren't lost off the front of the recording -- without this
    # the first ~100ms of every utterance gets clipped.
    preroll = deque(maxlen=detector.speech_frames_to_start + 2)
    recording_chunks = []
    was_muted = False

    # Temporary diagnostics: print mic level + VAD probability a few times
    # a second so we can tell, from a live run, whether audio is actually
    # arriving (RMS near 0 -> wrong/silent input device) or audio looks
    # fine but VAD never crosses threshold (VAD/threshold issue) --
    # can't hear the mic from here, so this is the fastest way to tell
    # those apart from the printed output. Remove once VAD is confirmed
    # working against real speech.
    debug_chunk_count = 0

    while not app_state.shutdown.is_set():
        if app_state.muted:
            if not was_muted:
                app_state.set_state(IDLE)
                detector = UtteranceDetector()  # drop any half-finished utterance
                recording_chunks = []
                preroll.clear()
                was_muted = True
            time.sleep(0.05)
            continue
        if was_muted:
            was_muted = False
            app_state.set_state(LISTENING)

        if app_state.state in (THINKING, SPEAKING):
            time.sleep(0.05)
            continue

        try:
            chunk = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        prob = vad.process_chunk(chunk)
        event = detector.update(prob)

        debug_chunk_count += 1
        if debug_chunk_count % 15 == 0:  # ~0.5s of audio at 512 samples/32ms per chunk
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            print(f"[mic-debug] rms={rms:.4f} vad_prob={prob:.3f} in_speech={detector.in_speech}")

        if detector.in_speech:
            recording_chunks.append(chunk)
        else:
            preroll.append(chunk)

        if event == "start":
            recording_chunks = list(preroll) + recording_chunks
            preroll.clear()
            app_state.set_state(RECORDING)
        elif event == "end":
            audio = np.concatenate(recording_chunks) if recording_chunks else np.zeros(0, dtype="float32")
            recording_chunks = []
            trim_samples = int(TRAIL_TRIM_S * SAMPLE_RATE)
            if len(audio) > trim_samples:
                audio = audio[:-trim_samples]
            process_utterance(app_state, whisper_model, piper_voice, audio)

    stream.stop()
    stream.close()
    print("Mic loop stopped (window closed).")


def main():
    app_state = AssistantState()
    api = Api(app_state)

    window = webview.create_window(
        "Voice Assistant",
        AVATAR_HTML,
        js_api=api,
        width=380,
        height=380,
        frameless=True,
        easy_drag=False,  # drag happens via #drag-region's -webkit-app-region:drag instead
        on_top=True,
        # transparent=True does NOT give real desktop-see-through on Windows
        # in pywebview 6.2.1 -- confirmed by reading platforms/winforms.py:
        # it sets the WebView2 control's own background to transparent, but
        # never sets the enclosing Form's AllowTransparency/TransparencyKey,
        # so the Form paints an opaque (effectively white) background behind
        # it regardless. Long-standing, still-open upstream issue, not
        # something to hand-patch here (r0x0r/pywebview #1611, #1200, #745).
        # Verified live 2026-08-28 via an actual screenshot: transparent=True
        # rendered a solid white square, not a floating orb. background_color
        # is the part of the API that actually works -- use it deliberately
        # instead of getting an unintended white box.
        transparent=False,
        background_color="#14141c",
    )
    app_state.window = window
    # mic_loop only checks this flag between iterations (every ~200ms at
    # most, via the queue.get timeout) -- good enough to stop the mic
    # stream and let close_conversation() run in worker()'s finally block
    # instead of leaving a non-daemon thread hanging after the window closes.
    window.events.closing += app_state.shutdown.set

    def worker():
        # Piper's load+warmup overlaps with Whisper loading instead of
        # adding to it, same reasoning as voice_test.py.
        piper_future = ThreadPoolExecutor(max_workers=1).submit(tts.load_piper)
        whisper_model = stt.load_whisper()
        piper_voice = piper_future.result()

        app_state.set_state(LISTENING)
        try:
            mic_loop(app_state, whisper_model, piper_voice)
        finally:
            if app_state.ws_id:
                tc.close_conversation(app_state.ws_id)

    webview.start(worker)


if __name__ == "__main__":
    main()
