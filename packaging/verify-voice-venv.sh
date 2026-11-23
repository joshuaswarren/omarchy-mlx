#!/usr/bin/env bash
# Release-time, aarch64: prove the vendored set runs the voice assistant offline.
#
#   packaging/verify-voice-venv.sh <vendor-dir>
#
# Two offline proofs run inside a network namespace with no route out
# (unshare -rn). First a scratch venv installs the lock under pip
# --require-hashes: the resolvable lines with full resolution, the --no-deps
# wheels (those declaring a deliberately unvendored dependency) without
# resolution — split-vendor-lock.py writes the two files. Then build-venv.sh
# builds the system venv and imports what voice_loop and synthesis load,
# from this tree's serve/. vendor-wheels.sh runs it before the lock is
# final, so a release whose vendored wheels cannot install and import the
# voice stack offline gets no tarball.
set -euo pipefail

self_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$self_dir/paths.sh"
vendor="$(cd "${1:?usage: verify-voice-venv.sh <vendor-dir>}" && pwd)"
lock="$vendor/requirements-lock.txt"
[[ -f $lock ]] || { echo "error: $lock is missing" >&2; exit 1; }
[[ $(uname -m) == aarch64 ]] || { echo "error: the vendored set is aarch64; run this on an aarch64 host" >&2; exit 1; }
command -v unshare >/dev/null || { echo "error: unshare is missing (util-linux)" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
py="$(command -v "python$PYTHON_VERSION")"

unshare -rn env -i PATH="$PATH" HOME="$work" PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$self_dir/../serve" bash -c '
set -euo pipefail
"$5" "$1/split-vendor-lock.py" --nodeps "$1/requirements-nodeps.in" --wheels "$2" \
  --lock "$3" --resolvable "$4/lock-resolve.txt" --nodeps-only "$4/lock-nodeps.txt"
"$5" -m venv "$4/probe"
"$4/probe/bin/python" -m pip install --no-index --find-links "$2" --require-hashes \
  -r "$4/lock-resolve.txt" >"$4/resolve-install.log" 2>&1 ||
  { cat "$4/resolve-install.log" >&2; exit 1; }
if [[ -s $4/lock-nodeps.txt ]]; then
  "$4/probe/bin/python" -m pip install --no-index --find-links "$2" --require-hashes \
    --no-deps -r "$4/lock-nodeps.txt" >"$4/nodeps-install.log" 2>&1 ||
    { cat "$4/nodeps-install.log" >&2; exit 1; }
fi
echo "==> vendored lock resolves offline under pip --require-hashes"
bash "$1/build-venv.sh" --vendor "$2" --lock "$3" --venv "$4/venv" --python "$5" >"$4/build-venv.log" 2>&1 ||
  { cat "$4/build-venv.log" >&2; exit 1; }
"$4/venv/bin/python" - <<"PY"
import importlib, socket, sys
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3)
    sys.exit("error: the namespace reaches the network; the proof must run offline")
except OSError:
    pass
modules = [
    "mlx.core", "mlx_audio", "mlx_audio.tts.generate", "mlx_audio.stt.generate",
    "onnxruntime", "openwakeword", "misaki", "misaki.en", "num2words", "spacy",
    "en_core_web_sm", "phonemizer", "espeakng_loader", "sounddevice", "scipy.signal",
    "sklearn", "miniaudio",
    "mlx_omarchy_assistant.voice_loop", "mlx_omarchy_assistant.vad",
    "mlx_omarchy_assistant.wake_word", "mlx_omarchy_assistant.synthesis",
    "mlx_omarchy_assistant.recognition", "mlx_omarchy_assistant.kokoro_stream",
]
failed = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {exc.__class__.__name__}: {exc}")
import spacy
spacy.load("en_core_web_sm")
for line in failed:
    print(f"error: {line}", file=sys.stderr)
if failed:
    sys.exit(1)
print(f"ok: {len(modules)} voice modules import offline; en_core_web_sm loads")
PY
' _ "$self_dir" "$vendor" "$lock" "$work" "$py"
