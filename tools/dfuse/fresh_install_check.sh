#!/usr/bin/env bash
# Runs INSIDE a gpuwin window: fresh-install check for the 9B raw route.
set -euo pipefail
WHEEL=$(ls /var/tmp/dfuse-build/dist/*dfuse.d86ea8815*.whl | head -1)
APPLY=/var/tmp/dfuse/fresh-root/scripts/apply-mlx-lm-patches.sh
M9B=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1)
B=${HOME}/bench-scripts/qwen38-mlx-bench.py
P=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
O=/var/tmp/dfuse/fresh
FRESH=/var/tmp/dfuse-fresh2-venv
mkdir -p "$O"

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
case $(cat /proc/pressure/cpu) in *"some avg10=0.00"*) ;; *) echo "PSI gate"; exit 1;; esac
printf "host=%s boot=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" > "$O/gates.txt"

if [ ! -e "$FRESH/bin/python" ]; then
  python3 -m venv "$FRESH"
  "$FRESH/bin/pip" install -q "$WHEEL" mlx-lm==0.31.3 2>/dev/null \
    || "$FRESH/bin/pip" install -q "$WHEEL" mlx-lm
  bash "$APPLY" "$FRESH" > "$O/apply.log" 2>&1 || { echo "APPLY FAILED"; tail -5 "$O/apply.log"; exit 1; }
  "$FRESH/bin/python" -c "import mlx.core as mx; print('fresh venv:', mx.__version__)"
  grep -c "patch-mlx-lm-gdn-raw-repeat" "$O/apply.log" || true
fi

count_gdu() { # envset label
  local envflag=$1 label=$2
  local PENV=()
  if [ "$envflag" = off ]; then PENV=(MLX_OMARCHY_GDN_RAW_REPEAT=0); fi
  local prof="$O/prof-$label.ndjson"
  timeout 600 env MLX_OMARCHY_GPU_PROFILE="$prof" MLX_OMARCHY_GPU_PROFILE_LABEL="$label" \
    "${PENV[@]}" "$FRESH/bin/python" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens 64 --prefill-tokens 512 --label "$label" --out "$O/$label.json" \
    > "$O/$label.log" 2> "$O/$label.err"
  if [ -e "$prof" ]; then
    python3 - "$prof" "$label" <<'PYEOF'
import json, sys
sys.path.insert(0, "/var/tmp/dfuse")
from analyze_w1 import load
ev = load(sys.argv[1])
gdu = sum(1 for r in ev if "GatedDeltaUpdate" in r["p"])
soup = sum(1 for r in ev if r["p"] in ("AsType", "Multiply", "Sum", "Subtract"))
print(f"{sys.argv[2]}: GDU-kernel dispatches={gdu} soup-dispatches={soup}")
PYEOF
  else
    echo "$label: (release wheel - no profiler; dispatch counts run on the diag fresh venv)"
  fi
  grep -o '"ordered_records_sha256[^,}]*' "$O/$label.json" | sed "s/^/$label digest: /"
  python3 -c "import json;print('$label rate:', round(json.load(open('$O/$label.json'))['decode_tok_rate']['median'],2))"
}

count_gdu default fresh-default-d64
count_gdu off fresh-gdoff-d64
run_digest() { # depth label [off]
  local depth=$1 label=$2
  local PENV=()
  [ "${3:-}" = off ] && PENV=(MLX_OMARCHY_GDN_RAW_REPEAT=0)
  timeout 600 env "${PENV[@]}" "$FRESH/bin/python" "$B" --model "$M9B" --prompts "$P" --limit 1 --warmup 1 \
    --passes 1 --new-tokens "$depth" --prefill-tokens 512 --label "$label" --out "$O/$label.json" \
    > "$O/$label.log" 2> "$O/$label.err"
  echo "$label digest: $(grep -o '"ordered_records_sha256[^,}]*' "$O/$label.json") rate=$(python3 -c "import json;print(round(json.load(open('$O/$label.json'))['decode_tok_rate']['median'],2))")"
}
run_digest 512 fresh-default-d512

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc"
