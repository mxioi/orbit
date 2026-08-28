"""Throwaway spike: full voice loop -- mic -> STT -> Turnstone -> TTS -> speakers.

Keeps one Turnstone workstream open for the whole run so it's a real
back-and-forth conversation, not a fresh session per question. Press
Ctrl+C to stop; the workstream is closed cleanly on exit.

Uses TURNSTONE_MODEL = "voice-fast" (Qwen3.5-9B-Q4_K_M, reasoning disabled
via chat-template-kwargs enable_thinking=false in the llama-server preset)
instead of the cluster default (qwen3.8-27b), which "thinks" through even
trivial questions -- confirmed live: ~30s of internal reasoning for "what
is 2+2" on the default vs ~1.6s/turn once warm on voice-fast.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console can't print emoji by default

# Must be set before faster_whisper/ctranslate2 import -- CTranslate2's
# internal LoadLibrary calls only respect PATH, not os.add_dll_directory().
os.environ["PATH"] = (
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cublas\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cudnn\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_nvrtc\bin;"
    r"C:\Program Files\Python314\Lib\site-packages\nvidia\cuda_runtime\bin;"
) + os.environ["PATH"]

# TTS engine: Piper (VITS-based), not Kokoro (StyleTTS2-based). Kokoro's
# GPU path was a confirmed dead end on this card's Pascal generation
# (Quadro P500, no tensor cores) -- "CUBLAS failure 8: the function
# requires an architectural feature absent from the device" -- and even
# on CPU it ran at ~2.3x real-time (too slow for a voice assistant).
# Piper's simpler architecture measured RTF 0.09x CPU-only on the exact
# same hardware and text -- ~25x faster than Kokoro, no GPU needed at all.

import keyboard
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig

SAMPLE_RATE = 16000  # whisper wants 16kHz mono
PUSH_TO_TALK_KEY = "space"
MODEL_DIR = "models"

# Piper's default comma pause reads as too short/ignorable (a property of
# this voice model's training, not exposed as a tunable pause-duration
# param). Fixed by ear, A/B tested live: length_scale=1.2 slows the whole
# utterance ~20% (pauses scale with it too), combined with swapping commas
# for semicolons (a stronger pause point for this phonemizer). Confirmed
# on both a synthetic multi-comma sentence and a natural multi-sentence
# passage -- user's call: "better than being too fast, that's for sure."
PIPER_SYNTHESIS_CONFIG = SynthesisConfig(length_scale=1.2)
PIPER_MODEL = "piper_models/en_US-lessac-medium.onnx"

CONSOLE_BASE = "http://REDACTED-LAN-IP:8095/v1/api"
# Never hardcode the token here -- it goes into git now. Set it via env var,
# or drop it in a local .turnstone_token file (gitignored) as a fallback.
TURNSTONE_TOKEN = os.environ.get("TURNSTONE_TOKEN", "")
if not TURNSTONE_TOKEN:
    _token_file = os.path.join(os.path.dirname(__file__), ".turnstone_token")
    if os.path.isfile(_token_file):
        TURNSTONE_TOKEN = open(_token_file, encoding="utf-8").read().strip()
if not TURNSTONE_TOKEN:
    raise SystemExit(
        "No Turnstone token found. Set the TURNSTONE_TOKEN env var, or put it "
        "(just the raw ts_... value, nothing else) in a .turnstone_token file "
        "next to this script."
    )
TURNSTONE_MODEL = "voice-fast"  # Qwen3.5-9B, reasoning disabled -- confirmed ~1.6s/turn once warm
TURNSTONE_PERSONA = "researcher"  # matches "quizzing me / tech info" use case, not orchestration
HEADERS = {"Authorization": f"Bearer {TURNSTONE_TOKEN}", "Content-Type": "application/json"}

# Set after create_conversation() -- interactive workstreams live on a
# node (turnstone-server), not in-process on the console like
# coordinators. Events/send must go to the node directly.
TURNSTONE_BASE = CONSOLE_BASE


# --- Turnstone (plain HTTP + SSE, see docs/coordinator-api-tour.md) --------

def _call(method, path, body=None, params=None, base=None):
    url = (base or TURNSTONE_BASE) + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode()
        return resp.status, (json.loads(raw) if raw else {})


def create_conversation(model="", persona=TURNSTONE_PERSONA, first_message=""):
    """Create a real interactive workstream (not a coordinator) routed to
    an actual turnstone-server node. Sets the module-level TURNSTONE_BASE
    to that node's own URL -- events/send for interactive workstreams
    must go directly to the owning node, not the console's proxy.

    Unlike the coordinator path, initial_message here did NOT reliably
    auto-send in testing (message_count stayed 0) -- so this always
    explicitly sends via watch-then-send on the node, same ordering as
    the coordinator fix, which is not vulnerable to the coordinator's
    worker-thread startup race since interactive workstreams don't queue
    at a "tool-result seam."
    """
    global TURNSTONE_BASE
    # auto_approve_tools (not blanket auto_approve) -- persists for every
    # future tool call on this ws, but scoped to exactly the researcher
    # persona's own tool_allowlist (read-only/informational: no bash, no
    # writes, nothing destructive). A blanket auto_approve=True would also
    # silently cover any tool a future persona/config change adds, which
    # is a broader grant than this task actually needs.
    body = {
        "persona": persona,
        "auto_approve_tools": [
            "read_file", "search", "web_fetch", "web_search", "recall",
            "memory", "tool_search",
        ],
    }
    if model:
        body["model"] = model
    status, resp = _call("POST", "/route/workstreams/new", body, base=CONSOLE_BASE)
    if status not in (200, 201):
        raise SystemExit(f"create failed: {status} {resp}")
    ws_id = resp["ws_id"]
    node_url = resp["node_url"]

    # The console picks node_url and hands it back -- don't blindly trust
    # a server-supplied URL as the target for every subsequent authenticated
    # call (that's a straight SSRF/token-exfiltration path if the console
    # were ever compromised or pointed at something untrusted). Constrain
    # to hosts we actually expect this homelab's single node to report.
    node_host = urllib.parse.urlparse(node_url).hostname
    allowed_hosts = {"turnstone", "REDACTED-LAN-IP", "127.0.0.1", "localhost"}
    if node_host not in allowed_hosts:
        raise SystemExit(
            f"refusing to send credentials to unexpected node_url host {node_host!r} "
            f"(expected one of {allowed_hosts}) -- got node_url={node_url!r}"
        )

    TURNSTONE_BASE = node_url.rstrip("/") + "/v1/api"
    if first_message:
        return ws_id, ask(ws_id, first_message)
    return ws_id, ""


def ask(ws_id, message, timeout_s=120, poll_interval=0.5):
    """Send a message and poll for the response instead of trusting SSE.

    SSE turned out to have a real subscribe/publish race that survived
    multiple fix attempts (confirmed on both coordinator AND plain
    interactive workstreams -- not kind-specific, a race in the event
    stream itself), causing responses to arrive empty or bleed into the
    NEXT turn's window. History polling sidesteps it entirely: /send is
    fire-and-forget, and /cluster/ws/{id}/detail's `live.state` plus
    `tail` reliably reflect ground truth regardless of SSE delivery.
    Trades away live token-by-token progress for actually being correct
    every time -- the right trade for a voice assistant that only needs
    the final text to speak, not a live-updating transcript."""
    # limit must cover the WHOLE conversation so far, not just the last
    # message -- with a small limit, older assistant rows fall outside
    # ids_before and look "new" to the completion check below, so the
    # poll loop can grab a stale reply from 2+ turns back before the
    # actual fresh one exists yet. Confirmed live: turns 3 and 4 of a
    # 4-turn conversation returned turn 1 and 2's exact text, instantly.
    _, hist_before = _call("GET", f"/workstreams/{ws_id}/history", params={"limit": 1000})
    ids_before = {m.get("event_id") for m in hist_before.get("messages", [])}

    _call("POST", f"/workstreams/{ws_id}/send", {"message": message})

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(poll_interval)
        print(".", end="", flush=True)
        _, detail = _call(
            "GET", f"/cluster/ws/{ws_id}/detail", params={"message_limit": 15}, base=CONSOLE_BASE
        )
        # A new assistant row with NON-EMPTY content is the real completion
        # signal. The researcher persona auto-checks a "memory pointer" on
        # nearly every turn -- that shows up as an assistant row that's
        # PURELY a tool call (content="", real payload in tool_calls),
        # followed later by the actual text-bearing assistant row after the
        # tool result comes back. Confirmed live: grabbing the tool-call
        # row as "the answer" returns '' even though the real answer
        # completes moments later in the same turn -- skip empty rows and
        # keep polling for the one that actually has text.
        # NOTE: this endpoint's message dicts use "messages" (not "tail"
        # per the docs) and "_event_id" (underscore-prefixed, unlike
        # /history's bare "event_id") -- confirmed against a live response.
        for msg in reversed(detail.get("messages", [])):
            if (msg.get("role") == "assistant" and msg.get("_event_id") not in ids_before
                    and msg.get("content", "").strip()):
                print()
                return msg.get("content", "")

    print("\n[timeout waiting for response]")
    return ""


def close_conversation(ws_id):
    try:
        _call("POST", f"/workstreams/{ws_id}/close", {})
    except urllib.error.URLError:
        pass


# --- STT / TTS (unchanged from the earlier isolated test) -----------------

def record():
    """Push-to-talk: hold PUSH_TO_TALK_KEY down to record, release to stop.
    No fixed duration -- as short or long as the question needs."""
    print(f"Hold [{PUSH_TO_TALK_KEY.upper()}] to talk, release when done...")
    keyboard.wait(PUSH_TO_TALK_KEY)  # blocks until first pressed

    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        print("Recording...", end="", flush=True)
        while keyboard.is_pressed(PUSH_TO_TALK_KEY):
            time.sleep(0.02)
    print(" (released)")

    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()


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


def speak(piper_voice, text):
    clean = strip_markdown_for_speech(text)
    chunks = list(piper_voice.synthesize(clean, PIPER_SYNTHESIS_CONFIG))
    if not chunks:
        return
    audio = np.concatenate([c.audio_int16_array for c in chunks])
    sd.play(audio, chunks[0].sample_rate)
    sd.wait()


def main():
    # Piper's load+warmup runs on a background thread so it overlaps with
    # Whisper loading and the first turn's record/transcribe/Turnstone-wait
    # time instead of adding to it -- by the time speak() is actually
    # called, it's almost certainly already done. (Piper's own warmup is
    # much cheaper than Kokoro's was, but the overlap costs nothing.)
    piper_future = ThreadPoolExecutor(max_workers=1).submit(load_piper)

    whisper_model = load_whisper()

    ws_id = None
    try:
        first_turn = True
        while True:
            audio = record()
            text = transcribe(whisper_model, audio)
            print(f"You said: {text!r}")
            if not text.strip():
                print("(nothing transcribed -- hold the key a bit longer, try again)\n")
                continue

            print("Thinking", end="", flush=True)
            if first_turn:
                ws_id, response = create_conversation(model=TURNSTONE_MODEL, first_message=text)
                print(f" [ws_id={ws_id}]", end="", flush=True)
                first_turn = False
            else:
                response = ask(ws_id, text)
            print(f"Turnstone: {response!r}\n")

            if response.strip():
                if not piper_future.done():
                    print("(waiting on Piper to finish loading -- only happens once)")
                speak(piper_future.result(), response)
    except KeyboardInterrupt:
        print("\nClosing conversation...")
    finally:
        if ws_id:
            close_conversation(ws_id)


if __name__ == "__main__":
    main()
