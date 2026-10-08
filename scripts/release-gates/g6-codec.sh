#!/usr/bin/env bash
# Gate 6 — TTS codec regression against the INSTALLED wheel, zero skips.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
P="$GATE_HOME/.local/share/mlx-omarchy"
LOG="$LOG_DIR/g6-codec.log"
T="$GATE_ROOT/${TAG}-tag-tests"
: > "$LOG"
gate_log_wheel_identity "$LOG"

"$P/venv/bin/pip" install --quiet --no-deps "mlx-audio==0.5.6" >>"$LOG" 2>&1
"$P/venv/bin/pip" install --quiet "huggingface_hub>=1.0" "miniaudio>=1.61" "scipy>=1.10.0" "tqdm>=4.67.1" >>"$LOG" 2>&1

rm -rf "$T"; mkdir -p "$T"
if [[ -n "${GATE_TAG_TARBALL:-}" ]]; then
  tar xzf "$GATE_TAG_TARBALL" -C "$T" --strip-components=1
else
  curl -fsSL "https://codeload.github.com/joshuaswarren/omarchy-mlx/tar.gz/refs/tags/$TAG" \
    | tar xz -C "$T" --strip-components=1
fi
gate_log "$LOG" "TAG_TESTS_EXIT $?"

gate_lock env -C "$T" PYTHONPATH="$P" \
  MLX_OMARCHY_TTS_TEST_PACK="$TTS_PACK_HOME/voice/qwen3-tts-0.6b-customvoice-4bit" \
  "$P/venv/bin/python" -m unittest tests.test_qwen3_tts_codec_regress -v >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "GATE6_EXIT $RC"
grep -E "^Ran|^OK|^FAILED|skipped" "$LOG" | tail -3 >>"$LOG"
gate_log "$LOG" "GATE6_DONE"
exit "$RC"
