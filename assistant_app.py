#!/usr/bin/env python3
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
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IS_WINDOWS = sys.platform == "win32"

if not IS_WINDOWS:
    # Must be set before `import webview` -- pywebview's GTK backend
    # defaults to Wayland when WSLg/a Wayland session sets WAYLAND_DISPLAY,
    # which showed inconsistent/premature-exit behavior in testing (a
    # window's GTK main loop returning almost immediately instead of
    # actually running). Forcing X11 (WSLg also provides an X11 socket)
    # was confirmed live to fix it -- the same window then ran its full
    # expected duration and fired real WebKit load events. Harmless outside
    # WSLg too, since any Linux desktop with an X server (or Xwayland,
    # which is standard) still has X11 available.
    os.environ.setdefault("GDK_BACKEND", "x11")

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
CONFIG_PATH = os.path.join(_HERE, "voice_config.json")
# Without this, pywebview's Windows backend falls back to extracting python.exe's
# own icon for the taskbar/window -- generic and not this app's identity. Matches
# web/avatar.html's idle-state gradient (PALETTE.idle: #ffb27a -> #ff8a5c) so the
# taskbar icon and the floating orb read as the same thing. Generated once via
# scratch script, not regenerated at runtime -- see git history for how.
ICON_PATH = os.path.join(_HERE, "icon.ico")

# Mirror web/avatar.html's PALETTE keys exactly -- these strings are passed
# straight through to setState() on the JS side. TRANSCRIBING is its own
# state (not folded into THINKING) so the avatar can visibly distinguish
# "running Whisper locally" (brief) from "waiting on the LLM" (usually the
# longest part of a turn) -- splitting one long undifferentiated wait into
# labeled stages reads as faster even when the total time is identical.
IDLE, LISTENING, RECORDING, TRANSCRIBING, THINKING, SPEAKING = (
    "idle", "listening", "recording", "transcribing", "thinking", "speaking"
)

# UtteranceDetector's silence_frames_to_end (~800ms) is deliberately long so
# a mid-sentence pause doesn't cut you off -- which means ~800ms of
# near-silence is sitting at the tail of the buffer when "end" fires. Trim
# most of it back off before handing audio to Whisper.
TRAIL_TRIM_S = 0.5

# VAD sensitivity: user-adjustable since the right threshold genuinely
# depends on the mic and room (louder/closer mic + quiet room can afford
# "high" without picking up noise; a quiet/far mic in a noisy room needs
# "low" to avoid false triggers). Cycled via the avatar's sensitivity
# button, persisted across restarts in voice_config.json (gitignored --
# a per-machine preference, not project config).
VAD_SENSITIVITY_LEVELS = ["low", "medium", "high"]
VAD_THRESHOLDS = {"low": 0.7, "medium": 0.5, "high": 0.3}
DEFAULT_VAD_SENSITIVITY = "medium"

# Barge-in candidate confirmation: once VAD flags speech while SPEAKING,
# TTS is NOT cut immediately -- that's what let ordinary background noise
# (or, worse, the assistant's own voice bleeding back into the mic) cut
# itself off. Instead, up to this much candidate audio is captured first
# and run through Whisper; only a real transcribed word confirms a genuine
# barge-in and actually stops TTS. If nothing intelligible comes back,
# it's treated as noise and TTS just keeps playing, uninterrupted.
BARGE_IN_CONFIRM_S = 1.0


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = {}
    if cfg.get("vad_sensitivity") not in VAD_SENSITIVITY_LEVELS:
        cfg["vad_sensitivity"] = DEFAULT_VAD_SENSITIVITY
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    except OSError as e:
        print(f"[config] failed to save {CONFIG_PATH}: {e!r}")


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
        self.config = load_config()
        # mic_loop reads this each iteration (detector.threshold = ...) --
        # a plain attribute update here takes effect on the very next VAD
        # check, no restart or detector-rebuild needed.
        self.vad_sensitivity = self.config["vad_sensitivity"]
        self.vad_threshold = VAD_THRESHOLDS[self.vad_sensitivity]
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

    def cycle_sensitivity(self):
        """Advances to the next VAD sensitivity level, persists it, and
        pushes the new label to the avatar. Returns the new level."""
        idx = VAD_SENSITIVITY_LEVELS.index(self.vad_sensitivity)
        self.vad_sensitivity = VAD_SENSITIVITY_LEVELS[(idx + 1) % len(VAD_SENSITIVITY_LEVELS)]
        self.vad_threshold = VAD_THRESHOLDS[self.vad_sensitivity]
        self.config["vad_sensitivity"] = self.vad_sensitivity
        save_config(self.config)
        print(f"[vad] sensitivity -> {self.vad_sensitivity} (threshold={self.vad_threshold})")
        self._push_sensitivity_label()
        return self.vad_sensitivity

    def _push_sensitivity_label(self):
        try:
            self.window.evaluate_js(f"setSensitivityLabel({self.vad_sensitivity!r})")
        except Exception as e:
            print(f"[avatar] setSensitivityLabel failed: {e}")

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
        # Only avatar.html's own mute-button click handler previously kept
        # the JS-side icon in sync (it calls setMuted() locally right after
        # telling Python) -- fine as long as Python-side muted only ever
        # changed in response to that click. Now that the tray icon can
        # also flip this flag, push it to JS here too so the floating
        # window's icon doesn't go stale when muted from the tray instead.
        # setMuted() is idempotent, so this is a harmless no-op on the
        # original JS-initiated path.
        try:
            self.window.evaluate_js(f"setMuted({str(is_muted).lower()})")
        except Exception:
            pass


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
    #
    # Linux's GTK backend (platforms/gtk.py) DOES implement easy_drag
    # properly on its own -- confirmed by reading it -- so none of this
    # manual dragging is needed there at all. main() passes
    # easy_drag=True on Linux instead, and these methods become no-ops:
    # avatar.html's JS still calls them (it doesn't know which platform
    # it's running under), they just have nothing to do. window.native
    # is a GTK widget on Linux, not a WinForms Form -- it has no
    # `.Location`, so this code must not run there regardless.
    def start_drag(self, screen_x, screen_y):
        if not IS_WINDOWS:
            return
        self._last_drag_pos = (screen_x, screen_y)

    def drag_move(self, screen_x, screen_y):
        if not IS_WINDOWS or self._last_drag_pos is None:
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

    def cycle_sensitivity(self):
        return self._app_state.cycle_sensitivity()


def process_utterance(app_state, whisper_model, piper_voice, audio, known_text=None):
    # mic_loop calls this directly on its own thread with no surrounding
    # try/except -- an uncaught exception here (a Turnstone network
    # hiccup, a transient Whisper/Piper error) would otherwise kill that
    # background thread silently, leaving the avatar stuck showing
    # THINKING/SPEAKING forever with no crash to even notice. Catch
    # broadly and always fall back to LISTENING so one bad turn doesn't
    # require restarting the app.
    #
    # known_text: a confirmed barge-in already ran this exact audio
    # through Whisper once to decide whether it was real speech (see
    # mic_loop's BARGE_IN_CONFIRM_S handling) -- pass that result through
    # instead of transcribing the same audio a second time.
    try:
        if known_text is not None:
            # A barge-in confirmation already ran Whisper on this audio --
            # no local transcription work left to show, go straight to
            # THINKING rather than flashing TRANSCRIBING for something
            # that's already done.
            text = known_text
            app_state.set_state(THINKING)
        else:
            app_state.set_state(TRANSCRIBING)
            text = stt.transcribe(whisper_model, audio)
            app_state.set_state(THINKING)
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
        # off, no VAD/state work here. Still drop audio during
        # TRANSCRIBING/THINKING (nothing is playing yet for barge-in to
        # apply to, and interrupting a live Turnstone round-trip is out of
        # scope anyway -- we'd otherwise just be queueing audio nothing
        # will consume for however long that takes) and while muted.
        # SPEAKING is let through -- that's what makes barge-in possible.
        if app_state.muted or app_state.state in (TRANSCRIBING, THINKING):
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
    barge_in_pending = False
    barge_in_started_at = None

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
                barge_in_pending = False
                was_muted = True
            time.sleep(0.05)
            continue
        if was_muted:
            was_muted = False
            app_state.set_state(LISTENING)

        if app_state.state in (TRANSCRIBING, THINKING):
            time.sleep(0.05)
            continue

        try:
            chunk = audio_q.get(timeout=0.2)
            got_since_heartbeat += 1
        except queue.Empty:
            empty_since_heartbeat += 1
            continue

        # Cheap attribute update, takes effect on this very check -- no
        # detector rebuild needed for a live sensitivity change.
        detector.threshold = app_state.vad_threshold

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
            if app_state.state == SPEAKING:
                # Barge-in CANDIDATE -- do NOT cut TTS yet. Background
                # noise or the assistant's own voice bleeding back into
                # the mic can trigger VAD too; committing to a cut here
                # would make either one a false interrupt. Keep TTS
                # playing and keep recording -- the candidate gets
                # confirmed against a real Whisper transcript below
                # (either right here once it ends, or after
                # BARGE_IN_CONFIRM_S if it's a longer utterance).
                barge_in_pending = True
                barge_in_started_at = time.time()
            else:
                app_state.set_state(RECORDING)
        elif event == "end":
            audio = np.concatenate(recording_chunks) if recording_chunks else np.zeros(0, dtype="float32")
            recording_chunks = []
            trim_samples = int(TRAIL_TRIM_S * SAMPLE_RATE)
            if len(audio) > trim_samples:
                audio = audio[:-trim_samples]
            if barge_in_pending:
                barge_in_pending = False
                text = stt.transcribe(whisper_model, audio) if len(audio) else ""
                if text.strip():
                    print(f"[mic] barge-in confirmed: {text!r}")
                    app_state.stop_tts_if_speaking()
                    app_state.set_state(RECORDING)
                    process_utterance(app_state, whisper_model, piper_voice, audio, known_text=text)
                else:
                    print("[mic] barge-in candidate had no recognizable words -- ignoring, TTS continues")
                    detector = UtteranceDetector()  # this one thinks it's mid-utterance; needs a clean reset
            else:
                # Fire-and-forget: instant feedback that something was
                # heard, before the (much slower, seconds-scale) STT/LLM
                # round-trip even starts. Mic capture is already dropped
                # during TRANSCRIBING/THINKING (see callback() above), so
                # this can't bleed into the next utterance's recording.
                tts.play_ack_chime()
                process_utterance(app_state, whisper_model, piper_voice, audio)

        # Barge-in candidate ran long enough to hit the confirmation cap
        # without a natural "end" yet (still talking) -- decide now using
        # whatever's captured so far, so a genuine interrupt on a long
        # sentence doesn't sit there uncut for its whole duration. If
        # confirmed, recording just continues normally below (detector.
        # in_speech is already True) and the eventual real "end" processes
        # the FULL utterance fresh, including whatever's said afterward.
        if barge_in_pending and time.time() - barge_in_started_at >= BARGE_IN_CONFIRM_S:
            barge_in_pending = False
            partial_audio = np.concatenate(recording_chunks) if recording_chunks else np.zeros(0, dtype="float32")
            text = stt.transcribe(whisper_model, partial_audio) if len(partial_audio) else ""
            if text.strip():
                print(f"[mic] barge-in confirmed (still talking): {text!r}")
                app_state.stop_tts_if_speaking()
                app_state.set_state(RECORDING)
            else:
                print("[mic] barge-in candidate had no recognizable words -- ignoring, TTS continues")
                recording_chunks = []
                preroll.clear()
                detector = UtteranceDetector()

    stream.stop()
    stream.close()
    print("Mic loop stopped (window closed).")


def _preflight_check():
    """Fail fast with one clear, actionable message before opening any
    window, instead of two much worse first-run experiences this project
    actually hit during development: a raw FileNotFoundError traceback
    from deep inside a background thread when piper_models/ is missing
    (worker() -> piper_future.result() re-raises it with no guidance --
    confirmed live earlier in this project), or a confusing connection
    error only once the mic loop is already running, if CONSOLE_BASE is
    still the unconfigured placeholder. The Turnstone token already gets
    this treatment (see turnstone_client.py's own SystemExit at import
    time) -- this covers the other two prerequisites that didn't have
    one yet.
    """
    problems = []
    if not os.path.isfile(VAD_MODEL_PATH):
        problems.append(
            f"VAD model not found at {VAD_MODEL_PATH!r}.\n"
            f"    This ships with the repo -- if it's missing, re-clone or check "
            f"vad_model/ wasn't excluded."
        )
    piper_onnx = tts.PIPER_MODEL
    piper_json = piper_onnx + ".json"
    if not (os.path.isfile(piper_onnx) and os.path.isfile(piper_json)):
        problems.append(
            f"Piper voice model not found ({piper_onnx} + .json).\n"
            f"    Download it from "
            f"https://github.com/rhasspy/piper/blob/master/VOICES.md and place "
            f"both files under piper_models/ next to this script."
        )
    if tc.CONSOLE_BASE == "http://your-turnstone-host:8095/v1/api":
        problems.append(
            "No Turnstone console address configured (still using the "
            "placeholder).\n"
            "    Set the TURNSTONE_CONSOLE_BASE env var, or put your real "
            "console URL in a .turnstone_console_base file next to this script."
        )
    if problems:
        print("\nOrbit can't start yet -- missing setup:\n")
        for i, p in enumerate(problems, 1):
            print(f"{i}. {p}\n")
        raise SystemExit(1)


def _build_tray_icon(app_state):
    """A system tray icon + right-click menu, so the floating window can be
    shown/hidden/muted/quit without hunting for a small frameless window
    that's easy to lose behind other windows once minimized. Reuses
    ICON_PATH (the same taskbar/window icon set in main()) so the tray,
    taskbar, and floating orb all read as the same identity instead of
    each having their own separately-drawn approximation; falls back to a
    plain filled circle in the avatar's idle color if that file is ever
    missing (matches this function's own best-effort philosophy -- see
    below).

    Imports are lazy (not at module level) because this is verified working
    on Windows only -- pystray's Linux backends need a system tray protocol
    (AppIndicator3, or a DE that still supports the legacy systray spec)
    that install-linux.sh doesn't provision, and this project has no way to
    verify that here. Callers should treat this as best-effort: see
    main()'s try/except around calling this.
    """
    import pystray
    from PIL import Image, ImageDraw

    if os.path.isfile(ICON_PATH):
        img = Image.open(ICON_PATH).convert("RGBA")
    else:
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((4, 4, 60, 60), fill=(255, 138, 92, 255))

    def on_show(icon, item):
        app_state.window.show()
        app_state.window.restore()

    def on_hide(icon, item):
        app_state.window.hide()

    def on_toggle_mute(icon, item):
        app_state.set_muted(not app_state.muted)

    def on_quit(icon, item):
        icon.stop()
        app_state.window.destroy()

    menu = pystray.Menu(
        pystray.MenuItem("Show", on_show, default=True),
        pystray.MenuItem("Hide", on_hide),
        pystray.MenuItem("Muted", on_toggle_mute, checked=lambda item: app_state.muted),
        pystray.MenuItem("Quit", on_quit),
    )
    return pystray.Icon("orbit", img, "Orbit", menu)


def main():
    _preflight_check()
    app_state = AssistantState()
    api = Api(app_state)

    window = webview.create_window(
        "Voice Assistant",
        AVATAR_HTML,
        js_api=api,
        width=380,
        height=380,
        frameless=True,
        # Windows: False -- its backend has no built-in drag support at
        # all (see Api.drag_move's comment), so dragging is hand-rolled
        # via window.move() instead. Linux: True -- confirmed by reading
        # platforms/gtk.py that its easy_drag is a real, working
        # implementation; no need to duplicate it.
        easy_drag=not IS_WINDOWS,
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
        #
        # Kept transparent=False on Linux too, not because it's confirmed
        # broken there (GTK's compositor path is genuinely different, and
        # WSLg's window rendering couldn't be visually verified from this
        # dev environment at all -- a capture limitation, not evidence
        # either way), but because background_color is already a known-
        # good choice on both platforms and there was no way to safely
        # verify transparency working on Linux before shipping this.
        # Worth revisiting once someone can actually look at it running.
        transparent=False,
        background_color="#14141c",
    )
    app_state.window = window
    # mic_loop only checks this flag between iterations (every ~200ms at
    # most, via the queue.get timeout) -- good enough to stop the mic
    # stream and let close_conversation() run in worker()'s finally block
    # instead of leaving a non-daemon thread hanging after the window closes.
    window.events.closing += app_state.shutdown.set

    # pystray needs its own run loop; on Windows (unlike macOS) it's fine
    # off the main thread, which pywebview's own loop (webview.start below)
    # needs for itself. Stop it alongside the window closing so it doesn't
    # linger as an orphaned tray icon after the app exits. Best-effort: a
    # missing/unsupported tray backend (expected on some Linux setups, see
    # _build_tray_icon's docstring) shouldn't take down the whole assistant
    # over what's just a convenience feature.
    def _run_tray_icon(icon):
        # Runs on its own daemon thread -- an exception here would
        # otherwise just print an unhandled-thread-exception traceback
        # (Python's default) instead of this project's usual clear,
        # one-line "here's what happened and it's not fatal" message.
        try:
            icon.run()
        except Exception as e:
            print(f"[tray] system tray icon unavailable, continuing without it: {e!r}")

    try:
        tray_icon = _build_tray_icon(app_state)
        threading.Thread(target=_run_tray_icon, args=(tray_icon,), daemon=True).start()
        window.events.closing += tray_icon.stop
    except Exception as e:
        print(f"[tray] system tray icon unavailable, continuing without it: {e!r}")

    def worker():
        # Piper's load+warmup overlaps with Whisper loading instead of
        # adding to it, same reasoning as voice_test.py.
        piper_future = ThreadPoolExecutor(max_workers=1).submit(tts.load_piper)
        whisper_model = stt.load_whisper()
        piper_voice = piper_future.result()

        app_state.set_state(LISTENING)
        app_state._push_sensitivity_label()
        try:
            mic_loop(app_state, whisper_model, piper_voice)
        finally:
            if app_state.ws_id:
                tc.close_conversation(app_state.ws_id)

    webview.start(worker, icon=ICON_PATH if os.path.isfile(ICON_PATH) else None)


if __name__ == "__main__":
    main()
