#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: GDU dispatch counts on the diag wheel,
# route ON (=1) vs OFF (=0), 9B d64.
set -euo pipefail
C=/var/tmp/dfuse-venv/bin/python
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
O=/var/tmp/dfuse/ordermatch
mkdir -p "$O"

count() { # envset label
  local envv=$1 label=$2
  local prof="$O/prof-cnt-$label.ndjson"
  timeout 600 env MLX_OMARCHY_GDN_RAW_REPEAT="$envv" \
    MLX_OMARCHY_GPU_PROFILE="$prof" MLX_OMARCHY_GPU_PROFILE_LABEL="cnt-$label" \
    "$C" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 --passes 1 \
    --new-tokens 64 --prefill-tokens 512 --label "cnt-$label" --out "$O/cnt-$label.json" \
    > "$O/cnt-$label.log" 2> "$O/cnt-$label.err"
  python3 - "$prof" "$label" <<'PYEOF'
import json, sys
sys.path.insert(0, "/var/tmp/dfuse")
from analyze_w1 import load
ev = load(sys.argv[1])
gdu = sum(1 for r in ev if "GatedDeltaUpdate" in r["p"])
soup = sum(1 for r in ev if r["p"] in ("AsType", "Multiply", "Sum", "Subtract"))
print(f"{sys.argv[2]}: GDU-kernel dispatches={gdu} soup-dispatches={soup}")
PYEOF
}

count 1 on
count 0 off
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
