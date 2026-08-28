"""GUI voice assistant: floating avatar window + VAD-driven listening (no
push-to-talk) driving the same STT -> Turnstone -> TTS pipeline as
voice_test.py.

Barge-in: interrupting TTS mid-speech is supported, but ONLY while SPEAKING
-- audio is still fully discarded during THINKING (interrupting a live
Turnstone round-trip is out of scope; server-side work isn't cancellable
from here anyway). By the time SPEAKING starts, the Turnstone call has
already returned -- we're just replaying local audio -- so a barge-in never
collides with an in-flight server request; it just cuts local playback and
starts an independent new turn. See mic_loop's audio callback and the
event == "start" handling below for where this is wired in.
"""
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
    thread(s) both touch. Plain attributes, not locks -- by construction
    only one thread is ever the "active writer" of `state` at a time (the
    mic thread stops acting on it during THINKING; during SPEAKING, the mic
    thread and the TTS thread both touch it, but per the barge-in race
    analysis in process_utterance()/mic_loop, any ordering between them
    still converges on the correct final state), so Python's GIL is enough
    here without real synchronization."""

    def __init__(self):
        self.window = None  # attached right after webview.create_window()
        self.state = IDLE
        self.muted = False
        self.ws_id = None
        self.shutdown = threading.Event()
        # Current TTS playback thread + its OWN stop Event, if any is
        # active. A fresh Event() is created per turn rather than reusing/
        # clearing one shared instance -- reusing one caused a real bug:
        # if turn N's speak() thread hadn't yet checked is_set() by the
        # time turn N+1 called .clear() on the SAME object, N's thread
        # would never see the stop signal and kept playing while N+1's
        # audio also started, i.e. two overlapping TTS streams. A fresh
        # object per turn makes that impossible -- old and new threads
        # can never reference the same flag.
        self._tts_thread = None
        self._tts_stop_event = None

    def stop_tts_if_speaking(self):
        """Signal the CURRENT tts thread (if any) to stop and block until
        it actually has -- called from mic_loop on a barge-in. Blocking
        here (briefly, ~50-100ms per the measured stop_event response
        time) is what turns "asked it to stop" into "guaranteed stopped
        before anything else starts speaking", which a fire-and-forget
        .set() with no join doesn't guarantee."""
        if self._tts_stop_event is not None:
            self._tts_stop_event.set()
        if self._tts_thread is not None:
            self._tts_thread.join(timeout=2.0)

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
        self._last_drag_pos = None

    def toggle_mute(self, is_muted):
        print(f"[mic] {'muted' if is_muted else 'unmuted'}")
        self._app_state.set_muted(is_muted)

    # --- Dragging ----------------------------------------------------
    # pywebview 6.2.1's Windows backend (edgechromium.py + winforms.py) has
    # NO built-in frameless-window dragging -- confirmed by reading both
    # files, neither references easy_drag or any drag handling; only the
    # legacy mshtml.py, plus qt.py/gtk.py/cocoa.py on other platforms,
    # implement it. First attempt used the classic Win32 trick (release
    # capture + send WM_NCLBUTTONDOWN/HTCAPTION to fake a title-bar click,
    # same thing mshtml.py's easy_drag does) -- didn't move the window at
    # all in testing, on either a mousedown or mousemove trigger, which
    # points at the ctypes call itself rather than timing (a common gotcha:
    # SendMessageW's args are pointer-sized on 64-bit Windows, and ctypes
    # silently mis-marshals them without explicit argtypes declared).
    # Switched to pywebview's own documented, cross-platform window.move()
    # API instead -- confirmed in platforms/winforms.py to call the real
    # Win32 SetWindowPos under the hood, so it's not meaningfully less
    # "native" than the message-hijack trick, just far more traceable:
    # track screen-space mouse deltas in JS, add them to the window's
    # current position each move.
    def start_drag(self, screen_x, screen_y):
        self._last_drag_pos = (screen_x, screen_y)

    def drag_move(self, screen_x, screen_y):
        if self._last_drag_pos is None:
            return
        last_x, last_y = self._last_drag_pos
        dx, dy = screen_x - last_x, screen_y - last_y
        self._last_drag_pos = (screen_x, screen_y)
        if dx == 0 and dy == 0:
            return
        loc = self._app_state.window.native.Location
        self._app_state.window.move(loc.X + dx, loc.Y + dy)

    def end_drag(self):
        self._last_drag_pos = None

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
            app_state.set_state(LISTENING)
            return

        if app_state.ws_id is None:
            app_state.ws_id, response = tc.create_conversation(model=tc.TURNSTONE_MODEL, first_message=text)
            print(f"[turnstone] ws_id={app_state.ws_id}")
        else:
            response = tc.ask(app_state.ws_id, text)
        print(f"Turnstone: {response!r}")
    except Exception as e:
        print(f"[process_utterance] turn failed, recovering to LISTENING: {e!r}")
        app_state.set_state(LISTENING)
        return

    if not response.strip():
        app_state.set_state(LISTENING)
        return

    # TTS runs on its own thread and this function returns immediately
    # once it's kicked off -- NOT blocking here is what lets mic_loop's
    # thread keep consuming audio/running VAD during playback, which is
    # what makes barge-in possible at all. (Earlier versions of this
    # function blocked on tts.speak() directly; that's why barge-in wasn't
    # possible before -- the one thread that could've noticed you talking
    # was busy sitting inside the TTS call.)
    app_state.set_state(SPEAKING)
    stop_event = threading.Event()  # this turn's own -- see AssistantState's __init__ comment

    def _speak():
        try:
            tts.speak(
                piper_voice, response,
                on_amplitude=app_state.set_amplitude,
                stop_event=stop_event,
            )
        except Exception as e:
            print(f"[tts] playback failed: {e!r}")
        finally:
            # Only reset to LISTENING if nothing else already moved us on
            # (a barge-in sets state to RECORDING itself, from mic_loop's
            # thread, the moment it fires -- whichever of these two race
            # ends up running last, the final state is correct either way:
            # if barge-in already fired, don't stomp RECORDING back to
            # LISTENING; if playback just finished naturally, do the
            # normal reset).
            if app_state.state == SPEAKING:
                app_state.set_state(LISTENING)

    thread = threading.Thread(target=_speak, daemon=True)
    app_state._tts_stop_event = stop_event
    app_state._tts_thread = thread
    thread.start()


def mic_loop(app_state, whisper_model, piper_voice):
    vad = SileroVAD(model_path=VAD_MODEL_PATH)
    detector = UtteranceDetector()
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        # Keep the realtime audio callback cheap -- just hand the chunk
        # off, no VAD/state work here. Still drop audio during THINKING
        # (interrupting a live Turnstone round-trip is out of scope, and
        # we'd otherwise just be queueing audio nothing will consume for
        # however long that takes) and while muted. SPEAKING is let
        # through -- that's what makes barge-in possible.
        if app_state.muted or app_state.state == THINKING:
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
    # Separate wall-clock heartbeat (not chunk-count-based) so it still
    # fires even if we're stuck in the muted/thinking/speaking branches
    # below and never reach a single chunk -- if app_state.state ever gets
    # stuck somewhere unexpected (a bug elsewhere leaving it in THINKING
    # forever, say), this is what would reveal it instead of just silently
    # printing nothing at all.
    last_heartbeat = time.time()
    got_since_heartbeat = 0
    empty_since_heartbeat = 0

    while not app_state.shutdown.is_set():
        if time.time() - last_heartbeat > 2.0:
            print(f"[mic-heartbeat] state={app_state.state} muted={app_state.muted} "
                  f"chunks_received={got_since_heartbeat} queue_timeouts={empty_since_heartbeat} "
                  f"(queue_timeouts high + chunks_received 0 -> no audio arriving at all)")
            last_heartbeat = time.time()
            got_since_heartbeat = 0
            empty_since_heartbeat = 0

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

        if app_state.state == THINKING:
            time.sleep(0.05)
            continue

        try:
            chunk = audio_q.get(timeout=0.2)
            got_since_heartbeat += 1
        except queue.Empty:
            empty_since_heartbeat += 1
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
            if app_state.state == SPEAKING:
                # Barge-in: cut TTS playback and BLOCK until it's actually
                # stopped (not just signaled) before continuing -- see
                # AssistantState.stop_tts_if_speaking's docstring for why
                # this guarantees no overlap instead of just hoping for
                # good timing. Everything else below is identical to a
                # normal utterance start -- the interrupting speech just
                # becomes the next turn once its own "end" fires further
                # down.
                print("[mic] barge-in detected, interrupting TTS")
                app_state.stop_tts_if_speaking()
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
