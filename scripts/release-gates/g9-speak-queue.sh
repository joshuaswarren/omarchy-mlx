#!/usr/bin/env bash
# Gate 9 — read-aloud playout through the installed release's static/js in
# headless Chromium against a serialising fake /api/speak (no GPU, no model):
# sentences in order, no overlap, no gap, no busy retry, one audio-done.
# Needs a Chromium-family browser: GATE_CHROMIUM, else chromium /
# chromium-browser / google-chrome on PATH; refuses loudly when none exists.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
P="$GATE_HOME/.local/share/mlx-omarchy"
LOG="$LOG_DIR/g9-speak-queue.log"
: > "$LOG"
gate_begin "$LOG"
gate_log_wheel_identity "$LOG"
BROWSER="${GATE_CHROMIUM:-$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)}"
if [[ -z "$BROWSER" || ! -x "$BROWSER" ]]; then
  gate_log "$LOG" "REFUSING: no Chromium-family browser (set GATE_CHROMIUM)"
  gate_log "$LOG" "G9_SPEAK_QUEUE_EXIT 2"
  exit 2
fi
env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" PYTHONPATH="$P" TERM=dumb \
  G9_NO_SANDBOX="${G9_NO_SANDBOX:-0}" \
  "$P/venv/bin/python" "$GATES_DIR/g9-speak-queue-driver.py" "$BROWSER" >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "G9_SPEAK_QUEUE_EXIT $RC"
grep -E "SPEAK_QUEUE_SMOKE|STATIC_JS" "$LOG" | tail -2
exit "$RC"
