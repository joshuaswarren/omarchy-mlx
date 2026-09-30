#!/bin/bash
# HELD-OUT v3 on the Everyday (9B), Quality (27B) and Compact (4B) pairs, then
# first-text latency on the 9B and the 4B, on the project M2.  One gpu-turn
# ticket at a time (<= 15 min).  Each ticket starts and stops its own
# assistant and stops starting prompts 150 s before the ticket ends;
# results/<suite>_<model>.json skips finished prompts, so the next ticket (or a
# rerun of this script after a reboot) resumes.  Set MARKCARDS_HOME first; the
# repo checkout must be at $MARKCARDS_HOME/agents/MarkdownCards/repo.
# (The 2B, the Everyday chat model before 8a1e25843, ran this suite first.)
set -u
_HOME=${MARKCARDS_HOME:?set MARKCARDS_HOME}
VENV=$_HOME/.local/share/mlx-omarchy/venv
REPO=$_HOME/agents/MarkdownCards/repo
LOGS=$_HOME/agents/MarkdownCards/logs
RESULTS=$_HOME/agents/MarkdownCards/results
SUITE=tests/fixtures/cards_held_out_v3.json
mkdir -p "$LOGS"
cd "$REPO"
recorded() {
  python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" \
    "$RESULTS/held_out_v3_$1.json" 2>/dev/null || echo 0
}
for model in qwen3.5-9b-mlx-4bit qwen3.8-27b-4bit qwen3-4b-instruct-2507-4bit; do
  for attempt in $(seq 1 12); do
    [ "$(recorded "$model")" -ge 36 ] && break
    "$_HOME/bin/gpu-turn" -m 15 -- \
      env MARKCARDS_HOME="$_HOME" "$VENV/bin/python" \
      receipts/2026-09-30-card-promotion/run_suite.py --suite "$SUITE" \
      --model "$model" --budget-s 750 \
      >> "$LOGS/v3_$model.log" 2>&1 || echo "ticket $attempt for $model exited $?" >&2
    sync
  done
  echo "$model recorded $(recorded "$model")/36" >&2
done
for model in qwen3.5-9b-mlx-4bit qwen3-4b-instruct-2507-4bit; do
  "$_HOME/bin/gpu-turn" -m 15 -- env MARKCARDS_HOME="$_HOME" "$VENV/bin/python" \
    receipts/2026-09-30-card-promotion/run_latency.py --model "$model" \
    >> "$LOGS/latency_$model.log" 2>&1 || echo "latency $model exited $?" >&2
done
echo "=== ALL DONE ===" >&2
