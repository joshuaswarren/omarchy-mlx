#!/bin/bash
# MarkdownCards lane for the frozen-M2 window, in the pre-registered order:
# fenced-json v3 on the 4B then the 9B (config selection), choose each pair's
# config by the pre-registered rule, v4 on the 9B and 4B, latency on both,
# then v4 on the 27B.  One gpu-turn ticket (<= 15 min) at a time; every step
# resumes from its results checkpoint, so rerunning this script after a
# reboot or a slot boundary continues where it stopped.
set -u
H=/home/joshuawarren
MC=$H/agents/MarkdownCards
PY=$H/.local/share/mlx-omarchy/venv/bin/python
RUN=receipts/2026-09-30-card-promotion/run_suite.py
LOGS=$MC/logs
echo "lane start $(date -Is) kernel $(uname -r) boot $(cat /proc/sys/kernel/random/boot_id)" >> $LOGS/lane.log

recorded() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['prompts']))" "$MC/results/$1" 2>/dev/null || echo 0; }

suite() {  # repo suite-file tag model
  local repo=$1 file=$2 tag=$3 model=$4 name
  name="$(basename "$file" .json | sed 's/^cards_//')${tag}_${model}.json"
  for attempt in $(seq 1 14); do
    [ "$(recorded "$name")" -ge 36 ] && return 0
    (cd "$MC/$repo" && $H/bin/gpu-turn -m 15 -- env MARKCARDS_HOME=$H $PY $RUN \
       --suite "$file" --tag "$tag" --model "$model" --budget-s 750) \
       >> "$LOGS/${name%.json}.log" 2>&1 || echo "ticket $attempt $name exited $?" >> $LOGS/lane.log
    sync
  done
  echo "$name stopped at $(recorded "$name")/36" >> $LOGS/lane.log
}

choose() {  # model -> prints repo-v4 or repo-v4-fenced by the pre-registered rule
  python3 - "$MC/results" "$1" <<'EOF'
import json, os, sys
res, model = sys.argv[1], sys.argv[2]
def score(name):
    try:
        rows = json.load(open(os.path.join(res, f"{name}_{model}.json")))["prompts"]
    except FileNotFoundError:
        return None
    if len(rows) < 36:
        return None
    cards = sum(p["pass"] for p in rows if p["category"] == "card-worthy")
    spurious = sum(not p["pass"] for p in rows if p["category"] != "card-worthy")
    return cards, spurious
default, fenced = score("held_out_v3"), score("held_out_v3_fenced")
ok = lambda s: s is not None and s[0] >= 15 and s[1] == 0
pick = "repo-v4-fenced" if ok(fenced) and (not ok(default) or fenced[0] > default[0]) else "repo-v4"
print(pick)
print(f"choose {model}: default={default} fenced={fenced} -> {pick}", file=sys.stderr)
EOF
}

suite repo-v3fenced tests/fixtures/cards_held_out_v3.json _fenced qwen3-4b-instruct-2507-4bit
suite repo-v3fenced tests/fixtures/cards_held_out_v3.json _fenced qwen3.5-9b-mlx-4bit
for model in qwen3.5-9b-mlx-4bit qwen3-4b-instruct-2507-4bit; do
  repo=$(choose $model 2>> $LOGS/lane.log)
  echo "$model -> $repo" >> $LOGS/lane.log
  suite $repo tests/fixtures/cards_held_out_v4.json "" $model
done
for model in qwen3.5-9b-mlx-4bit qwen3-4b-instruct-2507-4bit; do
  [ -f "$MC/results/latency_stock_$model.json" ] && continue
  repo=$(choose $model 2>/dev/null)
  (cd "$MC/$repo" && $H/bin/gpu-turn -m 15 -- env MARKCARDS_HOME=$H MARKCARDS_LATENCY_TAG=stock \
     $PY receipts/2026-09-30-card-promotion/run_latency.py --model $model) \
     >> "$LOGS/latency_stock_$model.log" 2>&1 || echo "latency $model exited $?" >> $LOGS/lane.log
done
suite repo-v4 tests/fixtures/cards_held_out_v4.json "" qwen3.8-27b-4bit
echo "=== LANE DONE $(date -Is) ===" >> $LOGS/lane.log
