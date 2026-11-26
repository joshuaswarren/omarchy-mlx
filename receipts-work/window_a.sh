#!/usr/bin/env bash
# BF16 prefill 1K attribution window (lock held by caller; <=25 min).
# Phase 1: dispatch census + GPU ticks on the diag wheel at the three legs.
# Phase 2: component isolation probe on the release wheel (m=30/262/1053).
set -euo pipefail
cd ~/src/mlx-Bf16PrefillGap
export MLX_DISABLE_COMPILE=1 HF_HUB_OFFLINE=1
ulimit -c 0
MODEL=~/models/Qwen2.5-0.5B-Instruct-bf16-mlx

PYD=$PWD/.work/venv-gap/bin/python
PYR=$PWD/.work/venv-gap-rel/bin/python

# ---- prompts (same construction as receipts census windows) ----
$PYD - <<'PYEOF'
import json
m = json.load(open("scripts/bench_matrix.json"))
open("/tmp/gap-short.txt", "w").write(m["prompts"]["short"]["text"])
open("/tmp/gap-long.txt", "w").write(m["prompts"]["long"]["text"])
e = m["prompts"]["ctx1024"]
parts = [e["base"]] + [
    "%s Entry %d of %d." % (e["item"], i, e["items"])
    for i in range(1, e["items"] + 1)]
open("/tmp/gap-ctx1024.txt", "w").write(" ".join(parts))
PYEOF

# ---- Phase 1: census (diag wheel) ----
for leg in short long ctx1024; do
  PROMPT="$(cat /tmp/gap-$leg.txt)"
  MLX_OMARCHY_GPU_PROFILE=/tmp/gap-census-$leg.jsonl \
    $PYD scripts/profile_generate.py \
    --model $MODEL --prompt "$PROMPT" --max-tokens 2 --temp 0 --seed 0 \
    --markers /tmp/gap-census-$leg-markers.jsonl \
    > /tmp/gap-census-$leg.log 2>&1
  echo "census $leg rc=$?"
done

for leg in short long ctx1024; do
  $PYD scripts/profile_analyze.py /tmp/gap-census-$leg.jsonl \
    --markers /tmp/gap-census-$leg-markers.jsonl \
    --compute-h overlay/mlx/backend/omarchy/compute.h \
    > /tmp/gap-analysis-$leg.txt 2>&1
  echo "analyze $leg rc=$?"
done

# ---- Phase 2: component isolation (release wheel) ----
$PYR receipts-work/prefill_components3.py \
  --out /tmp/gap-components.ndjson --tag rel-63c9a8d8
echo "components rc=$?"

mkdir -p receipts-work/m1-logs
cp /tmp/gap-census-*.jsonl /tmp/gap-census-*-markers.jsonl \
   /tmp/gap-census-*.log /tmp/gap-analysis-*.txt \
   /tmp/gap-components.ndjson receipts-work/m1-logs/ 2>/dev/null || true
echo WINDOW_A_DONE
