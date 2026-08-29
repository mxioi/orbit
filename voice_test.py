#!/usr/bin/env python3
"""Push-to-talk CLI voice loop -- mic -> STT -> Turnstone -> TTS -> speakers.

Keeps one Turnstone workstream open for the whole run so it's a real
back-and-forth conversation, not a fresh session per question. Press
Ctrl+C to stop; the workstream is closed cleanly on exit.

Superseded for daily use by assistant_app.py (GUI + VAD instead of
push-to-talk), kept as a lightweight terminal-only fallback and as the
simplest place to sanity-check STT/TTS/Turnstone changes without the GUI.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console can't print emoji by default

from concurrent.futures import ThreadPoolExecutor

import keyboard
import numpy as np
import sounddevice as sd

import stt
import tts
import turnstone_client as tc

PUSH_TO_TALK_KEY = "space"


def record():
    """Push-to-talk: hold PUSH_TO_TALK_KEY down to record, release to stop.
    No fixed duration -- as short or long as the question needs."""
    print(f"Hold [{PUSH_TO_TALK_KEY.upper()}] to talk, release when done...")
    keyboard.wait(PUSH_TO_TALK_KEY)  # blocks until first pressed

    chunks = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=stt.SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        print("Recording...", end="", flush=True)
        while keyboard.is_pressed(PUSH_TO_TALK_KEY):
            time.sleep(0.02)
    print(" (released)")

    if not chunks:
        return np.zeros(0, dtype="float32")
    return np.concatenate(chunks).flatten()


def _preflight_check():
    """Same rationale as assistant_app.py's _preflight_check() -- fail fast
    with one clear, actionable message instead of a raw FileNotFoundError
    the first time tts.speak() actually needs the Piper model (previously
    surfaced deep in the run, after a full record/transcribe/Turnstone
    round-trip had already happened), or a confusing connection error if
    CONSOLE_BASE is still the unconfigured placeholder. No VAD model check
    here -- unlike assistant_app.py, this is push-to-talk and never touches
    vad.py at all."""
    problems = []
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


def main():
    _preflight_check()
    # Piper's load+warmup runs on a background thread so it overlaps with
    # Whisper loading and the first turn's record/transcribe/Turnstone-wait
    # time instead of adding to it -- by the time speak() is actually
    # called, it's almost certainly already done. (Piper's own warmup is
    # much cheaper than Kokoro's was, but the overlap costs nothing.)
    piper_future = ThreadPoolExecutor(max_workers=1).submit(tts.load_piper)

    whisper_model = stt.load_whisper()

    ws_id = None
    try:
        first_turn = True
        while True:
            audio = record()
            text = stt.transcribe(whisper_model, audio)
            print(f"You said: {text!r}")
            if not text.strip():
                print("(nothing transcribed -- hold the key a bit longer, try again)\n")
                continue

            print("Thinking", end="", flush=True)
            if first_turn:
                ws_id, response = tc.create_conversation(
                    model=tc.TURNSTONE_MODEL, first_message=text
                )
                print(f" [ws_id={ws_id}]", end="", flush=True)
                first_turn = False
            else:
                response = tc.ask(ws_id, text, on_poll=lambda: print(".", end="", flush=True))
            print(f"\nTurnstone: {response!r}\n")

            if response.strip():
                if not piper_future.done():
                    print("(waiting on Piper to finish loading -- only happens once)")
                tts.speak(piper_future.result(), response)
    except KeyboardInterrupt:
        print("\nClosing conversation...")
    finally:
        if ws_id:
            tc.close_conversation(ws_id)


if __name__ == "__main__":
    main()
