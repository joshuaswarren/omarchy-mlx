#!/bin/bash
# HELD-OUT v3 on both pairs, then the 2B first-text latency, on the project M2.
# One gpu-turn ticket at a time.  Each ticket starts and stops its own
# assistant and stops starting prompts two minutes before the ticket ends;
# results/<suite>_<model>.json skips finished prompts, so the next ticket (or a
# rerun of this script after a reboot) resumes.  Set MARKCARDS_HOME first; the
# repo checkout must be at $MARKCARDS_HOME/agents/MarkdownCards/repo.
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
for spec in "qwen3.8-2b-4bit 8" "qwen3.8-27b-4bit 15"; do
  set -- $spec
  model=$1 minutes=$2
  for attempt in $(seq 1 12); do
    [ "$(recorded "$model")" -ge 36 ] && break
    "$_HOME/bin/gpu-turn" -m "$minutes" -- \
      env MARKCARDS_HOME="$_HOME" "$VENV/bin/python" \
      receipts/2026-09-30-card-promotion/run_suite.py --suite "$SUITE" \
      --model "$model" --budget-s $((minutes * 60 - 150)) \
      >> "$LOGS/v3_$model.log" 2>&1 || echo "ticket $attempt for $model exited $?" >&2
    sync
  done
  echo "$model recorded $(recorded "$model")/36" >&2
done
"$_HOME/bin/gpu-turn" -m 8 -- env MARKCARDS_HOME="$_HOME" "$VENV/bin/python" \
  receipts/2026-09-30-card-promotion/run_latency.py --model qwen3.8-2b-4bit \
  >> "$LOGS/latency_2b.log" 2>&1 || echo "latency exited $?" >&2
echo "=== ALL DONE ===" >&2
