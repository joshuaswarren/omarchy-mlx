#!/bin/bash
# 2026-09-29 20:10Z: chunks of 3 prompts each, gpu-turn -m 8 (or -m 28
# for 27B labels). Override MARKCARDS_HOME and MARKCARDS_VENV before
# running.  Each chunk writes results_<model>.json and skips prompts
# already recorded.
set -u
_HOME=${MARKCARDS_HOME:-<home>}
VENV=${MARKCARDS_VENV:-$_HOME/.local/share/mlx-omarchy/venv}
REPO=${MARKCARDS_REPO:-$_HOME/agents/MarkdownCards/repo}
export MARKCARDS_HOME MARKCARDS_VENV
cd "$REPO"
run_chunk() {
  local label="$1"; shift
  # Clamp to 15 min per gpu-turn's broadcast cap.  27B setup needs
  # ~10 min on the v0.7.6 wheel, leaving ~5 min for prompts; with 3
  # prompts per chunk we may not finish every prompt in one ticket.
  # Per-prompt results_<model>.json means each chunk resumes where
  # the last left off, so partial progress is safe.
  local minutes=8
  if [[ "$label" == *27b* ]]; then
    minutes=15
  fi
  "$_HOME/bin/gpu-turn" -m "$minutes" -- \
    env MARKCARDS_HOME="$_HOME" \
        MARKCARDS_VENV="$VENV" \
        "$VENV/bin/python" \
    "$@" \
    > "$_HOME/agents/MarkdownCards/${label}.log" 2>&1 || \
    echo "FAIL ${label}" >&2
  sync
}
for model in qwen3.8-2b-4bit qwen3.8-27b-4bit; do
  for range in "0 3" "3 6" "6 9" "9 12" "12 15" "15 18" "18 21" "21 24"; do
    set -- $range
    run_chunk "${model}_${1}_${2}" \
      receipts/2026-09-30-card-promotion/run_held_out.py \
      --model "$model" --start "$1" --end "$2"
  done
done
run_chunk "latency_2b_v076" \
  receipts/2026-09-30-card-promotion/run_latency.py \
  --model qwen3.8-2b-4bit
echo "=== ALL DONE ===" >&2
