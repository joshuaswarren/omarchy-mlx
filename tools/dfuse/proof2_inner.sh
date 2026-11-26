#!/usr/bin/env bash
# DispatchFuse proof window v2: two gated census runs into w1d + remote
# dispatch-delta analysis. Runs INSIDE a gpuwin window.
set -euo pipefail
PY=/var/tmp/dfuse-venv/bin/python
OUT=/var/tmp/dfuse/w1d
HF=${HOME}/.cache/huggingface/hub

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" \
  | tee "$OUT/proof2-gates.txt"

run_one() { # name model depth env...
  local name=$1 model=$2 depth=$3; shift 3
  local m
  case "$model" in
    4b) m=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1) ;;
    9b) m=$(ls -d $HF/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1) ;;
  esac
  timeout 600 env MLX_OMARCHY_GPU_PROFILE="$OUT/prof-$name.ndjson" \
    MLX_OMARCHY_GPU_PROFILE_LABEL="$name" "$@" \
    "$PY" "$HOME/bench-scripts/qwen38-mlx-bench.py" --model "$m" \
      --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
      --limit 1 --warmup 1 --passes 1 --new-tokens "$depth" \
      --prefill-tokens 512 --label "$name" --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2> "$OUT/$name.err" \
    || { echo "RUN FAILED $name rc=$?"; return 1; }
  echo "done $name $(grep -o '"ordered_records_sha256[^,}]*' "$OUT/$name.json" | head -1)"
}

run_one 4b-d64-rn2 4b 64 MLX_OMARCHY_ROPE_NORM_FUSE=1
run_one 9b-d64-gdn2 9b 64 MLX_OMARCHY_GDN_RAW_REPEAT=1

# remote delta analysis
python3 /var/tmp/dfuse/delta_proof.py "$OUT" > "$OUT/proof2-deltas.txt" 2>&1 || echo "delta analysis failed"
cat "$OUT/proof2-deltas.txt"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a "$OUT/proof2-gates.txt"
