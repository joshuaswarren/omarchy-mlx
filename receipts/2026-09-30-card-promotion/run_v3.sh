#!/bin/bash
# HELD-OUT v3 on both pairs, then the 2B first-text latency, on the project M2.
# Every chunk is its own gpu-turn ticket (<= 15 min) that starts and stops its
# own assistant; results/<suite>_<model>.json skips finished prompts, so rerun
# this script after a reboot.  Set MARKCARDS_HOME to the M2 home first; the
# repo checkout must be at $MARKCARDS_HOME/agents/MarkdownCards/repo.
set -u
_HOME=${MARKCARDS_HOME:?set MARKCARDS_HOME}
VENV=$_HOME/.local/share/mlx-omarchy/venv
REPO=$_HOME/agents/MarkdownCards/repo
LOGS=$_HOME/agents/MarkdownCards/logs
mkdir -p "$LOGS"
cd "$REPO"
chunk() {
  local minutes="$1" label="$2"; shift 2
  "$_HOME/bin/gpu-turn" -m "$minutes" -- \
    env MARKCARDS_HOME="$_HOME" "$VENV/bin/python" "$@" \
    >> "$LOGS/$label.log" 2>&1 || echo "FAIL $label" >&2
  sync
}
SUITE=tests/fixtures/cards_held_out_v3.json
RUN=receipts/2026-09-30-card-promotion/run_suite.py
for start in 0 6 12 18 24 30; do
  chunk 8 "v3_2b_$start" "$RUN" --suite "$SUITE" --model qwen3.8-2b-4bit \
    --start "$start" --end $((start + 6))
done
for start in $(seq 0 3 33); do
  chunk 15 "v3_27b_$start" "$RUN" --suite "$SUITE" --model qwen3.8-27b-4bit \
    --start "$start" --end $((start + 3))
done
chunk 8 latency_2b receipts/2026-09-30-card-promotion/run_latency.py --model qwen3.8-2b-4bit
echo "=== ALL DONE ===" >&2
