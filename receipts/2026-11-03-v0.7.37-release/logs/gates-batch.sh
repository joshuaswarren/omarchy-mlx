#!/usr/bin/env bash
# Run the named v0.7.37 release gates, in order, against the DRAFT assets. Stops at the first failure.
# Adapted from w7Q's v0.7.36 gates-batch.sh (same flow; the tag, pin and expected hashes are the only changes, and the hashes come from the ticket).
# Usage: EXPECTED_WHEEL_SHA256=... EXPECTED_VTAR_SHA256=... gates-batch.sh <item> [<item> ...]
#   item: NAME | NAME:MODE (g13-gdn-maskless:build) | cleanup:hf
#   cleanup:hf deletes only this run's fresh Hugging Face cache and its g3 assistant home (literal names from TAG and GATE_ROOT), to give disk back.
# Every path can be overridden by the environment; the gate scripts come from GATE_WORKTREE (the tag's tree), so what runs is what the tag contains.
set -uo pipefail
export TAG=v0.7.37
export TAG_SHA=06d1038d2aa26ed235e65d2aa68677b9e904d65b
export ASSETS_DIR="${ASSETS_DIR:-$HOME/scratch/w72/rel-0737-gate-assets}"
export GATE_ROOT="${GATE_ROOT:-$HOME/scratch/w72/rel-0737-gates}"
export GATE_WORKTREE="${GATE_WORKTREE:-$HOME/scratch/w72/rel-0737-gate-tree}"
export TTS_PACK_HOME="${TTS_PACK_HOME:-$HOME/scratch/w7q-ttspack/mlx-tts-home}"
: "${EXPECTED_WHEEL_SHA256:?set EXPECTED_WHEEL_SHA256}" "${EXPECTED_VTAR_SHA256:?set EXPECTED_VTAR_SHA256}"
export EXPECTED_WHEEL_SHA256 EXPECTED_VTAR_SHA256
# A private lock file: the gates flock it freely and cannot deadlock against the guard's own gpu-turn lock.
export GPU_LOCK=/tmp/w72-v0737-gate.lock
export GATE_CHROMIUM="${GATE_CHROMIUM:-$(command -v chromium || true)}"
export PY_AARCH64=python3.14
unset GPU_TURN_TICKET RUN_ANE_GATES SERVING_VENV
mkdir -p "$GATE_ROOT/$TAG-gate-logs"
DONE="$GATE_ROOT/$TAG-gate-logs/gates.done"
GD="$GATE_WORKTREE/scripts/release-gates"
for item in "$@"; do
  if [ "$item" = cleanup:hf ]; then
    for T in "$GATE_ROOT/$TAG-fresh-hf" "$GATE_ROOT/$TAG-assist-4b"; do
      case "$T" in
        "$GATE_ROOT/v0.7.37-fresh-hf" | "$GATE_ROOT/v0.7.37-assist-4b") ;;
        *) echo "bad cleanup target $T"; exit 2 ;;
      esac
      [ -d "$T" ] || { echo "cleanup: $T not present"; continue; }
      echo "cleanup: $T $(du -sm "$T" | cut -f1) MB"
      rm -rf -- "$T"
    done
    echo "cleanup done, free: $(df -BG --output=avail "$HOME" | tail -1)"
    continue
  fi
  g="${item%%:*}"; mode=""
  [ "$g" != "$item" ] && mode="${item#*:}"
  echo "=== $g ${mode:+($mode)} $(date -u +%T) free $(df -BG --output=avail "$HOME" | tail -1) ==="
  if [ -n "$mode" ]; then bash "$GD/$g.sh" "$mode"; else bash "$GD/$g.sh"; fi
  rc=$?
  echo "${g}${mode:+_$mode}_RC=$rc $(date -u +%T)" | tee -a "$DONE"
  [ "$rc" -eq 0 ] || { echo "STOPPING: $g failed with rc $rc"; exit "$rc"; }
done
echo "BATCH_OK $*"
