#!/usr/bin/env bash
# Gate 8 — DEFAULT speech path end to end through the installed release
# wheel: Kokoro-82M with voice af_heart, no voice argument, no saved choice.
# Honest scope: the RTF here includes the first model load; it proves the
# default path boots and the pipeline is wired, NOT real-time qualification.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
P="$GATE_HOME/.local/share/mlx-omarchy"
LOG="$LOG_DIR/g8-kokoro.log"
: > "$LOG"
gate_begin "$LOG"

"$P/venv/bin/pip" install --quiet --no-deps "mlx-audio==0.5.6" >>"$LOG" 2>&1
"$P/venv/bin/pip" install --quiet "huggingface_hub>=1.0" "miniaudio>=1.61" "scipy>=1.10.0" "tqdm>=4.67.1" >>"$LOG" 2>&1
"$P/venv/bin/pip" install --quiet "espeakng-loader" "phonemizer" "misaki" "num2words" "spacy" >>"$LOG" 2>&1
"$P/venv/bin/pip" install --quiet \
  "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >>"$LOG" 2>&1 \
  || "$P/venv/bin/python" -m spacy download en_core_web_sm >>"$LOG" 2>&1

cp "$GATES_DIR/g8-kokoro-driver.py" "$GATE_HOME/kokoro-smoke-driver.py"
flock "$GPU_LOCK" env -i PATH="$GATE_INSTALL_PATH" HOME="$GATE_HOME" \
  PYTHONPATH="$P" MLX_OMARCHY_TTS_HOME="$TTS_PACK_HOME" TERM=dumb \
  "$P/venv/bin/python" "$GATE_HOME/kokoro-smoke-driver.py" >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "G8_KOKORO_EXIT $RC"
grep -E "KOKORO_SMOKE|LOAD " "$LOG" | tail -3 >>"$LOG"
exit "$RC"
