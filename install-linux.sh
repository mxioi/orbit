#!/usr/bin/env bash
# Linux setup for Orbit. Verified live against a real Ubuntu 24.04
# (WSL2 + WSLg, though nothing here is WSL-specific except where noted) --
# both of the two highest-risk unknowns for a Linux port, real-time audio
# I/O and a native GUI window, were confirmed genuinely working, not just
# theoretically compatible:
#   - sounddevice's exact InputStream/OutputStream patterns this project
#     uses were run directly: real device enumerated, correct chunk
#     timing on capture, clean playback.
#   - pywebview's exact create_window() parameters this project uses were
#     run directly: GTK's main loop ran the full expected duration,
#     WebKit fired a real `loaded` event for the actual avatar.html file
#     (served via pywebview's own local HTTP server, 12789 bytes, 200 OK).
#
# This installs a venv + Python deps, matching requirements-linux.txt's
# own header comment for what differs from the Windows requirements.txt
# and why. It intentionally does NOT bundle a frozen binary/installer --
# see the project notes on why a plain script is the right stage for
# this project right now (heavy native ML deps + a GUI backend that's a
# system package, not a pip package, make freezing meaningfully riskier
# than usual; revisit once the app's confirmed stable across more
# machines).
#
# Real gotchas found during the investigation, encoded below:
#   1. Ubuntu's stock libportaudio2 package is built WITHOUT PulseAudio
#      support (a known Debian/Ubuntu packaging gap) -- confirmed via
#      `ldd` showing no libpulse link, and sounddevice.query_hostapis()
#      showing only ALSA/OSS, both with zero devices. Fixed by installing
#      libasound2-plugins (the ALSA-to-PulseAudio bridge plugin), which
#      makes PortAudio's existing ALSA backend transparently route
#      through Pulse -- no portaudio rebuild needed.
#   2. pywebview's Linux GUI backend (GTK + WebKit2GTK) needs system
#      packages pip can't provide, AND PyGObject needs dev headers to
#      build against.
#   3. GDK_BACKEND=x11 needs to be set before pywebview's GTK backend
#      initializes -- assistant_app.py already does this itself
#      (os.environ.setdefault, Linux-only) when a Wayland session is
#      detected, so nothing extra is needed here; noted for context on
#      why gir1.2-webkit2-4.1 (not a Wayland-specific package) is enough.
#
# NOT included here (deliberately): CUDA/GPU packages for Whisper --
# untested on Linux by this project so far. stt.py's load_whisper()
# already tries cuda then falls back to cpu regardless, so this is a
# safe default, not a missing feature. See requirements-linux.txt's
# comment for how to add GPU support once it's actually verified.
set -euo pipefail

if [ "$EUID" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "Installing system packages (audio + GUI backend + build headers)..."
$SUDO apt-get update
$SUDO apt-get install -y \
  python3-venv \
  libportaudio2 \
  libasound2-plugins \
  gir1.2-webkit2-4.1 \
  python3-gi \
  gir1.2-gtk-3.0 \
  libgirepository-2.0-dev \
  libcairo2-dev \
  pkg-config

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HERE/.venv-linux"

echo "Creating venv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r "$HERE/requirements-linux.txt"

echo ""
echo "Done. To run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 assistant_app.py"
echo ""
echo "You'll also need piper_models/en_US-lessac-medium.onnx (+ its .json) --"
echo "same as the Windows setup, this is gitignored and not fetched by this"
echo "script; grab it separately if it's not already present."
