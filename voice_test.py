"""Push-to-talk CLI voice loop -- mic -> STT -> Turnstone -> TTS -> speakers.

Keeps one Turnstone workstream open for the whole run so it's a real
back-and-forth conversation, not a fresh session per question. Press
Ctrl+C to stop; the workstream is closed cleanly on exit.

Superseded for daily use by assistant_app.py (GUI + VAD instead of
push-to-talk), kept as a lightweight terminal-only fallback and as the
simplest place to sanity-check STT/TTS/Turnstone changes without the GUI.
"""
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


def main():
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
