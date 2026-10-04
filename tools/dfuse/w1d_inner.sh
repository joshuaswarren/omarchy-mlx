#!/usr/bin/env bash
# DispatchFuse W1 census inner. Runs INSIDE a gpuwin window. SET=2b|4b|9b.
set -euo pipefail
SET=${SET:?missing SET}
test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s set=%s\n" "$(hostname)" "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" "$SET" | tee /var/tmp/dfuse/w1-gates-$SET.txt

PY=${DF_PY:-/var/tmp/v072-venv-fused/bin/python}
BENCH=${HOME}/bench-scripts/qwen38-mlx-bench.py
PROMPTS=${HOME}/bench-scripts/qwen38-2b-prompts.jsonl
OUT=${DF_OUT:-/var/tmp/dfuse/w1}
mkdir -p "$OUT"
"$PY" /var/tmp/dfuse/mlx_provenance.py > "$OUT/provenance-$SET.txt" 2>&1 || true

case "$SET" in
  2b) M=${HOME}/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381 ;;
  4b) M=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1) ;;
  9b) M=$(ls -d ${HOME}/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1) ;;
  *) echo "bad SET"; exit 2 ;;
esac
test -d "$M" || { echo "missing model $M"; exit 1; }

run_one() { # name depth extra-env...
  local name=$1 depth=$2; shift 2
  echo "=== run $name depth=$depth env=$* start $(date -u +%FT%TZ)"
  timeout 300 env MLX_OMARCHY_GPU_PROFILE="$OUT/prof-$name.ndjson" \
    MLX_OMARCHY_GPU_PROFILE_LABEL="$name" "$@" \
    "$PY" "$BENCH" --model "$M" --prompts "$PROMPTS" --limit 1 --warmup 1 --passes 1 \
      --new-tokens "$depth" --prefill-tokens 512 --label "dfuse-$name" --out "$OUT/$name.json" \
    > "$OUT/$name.log" 2> "$OUT/$name.err" || { echo "RUN FAILED $name rc=$?"; return 1; }
  echo "=== run $name done $(date -u +%FT%TZ)"
}

case "$SET" in
  2b)
    run_one 2b-d64 64
    run_one 2b-d512 512
    run_one 2b-d64-kvoff 64 MLX_OMARCHY_KV_DIRECT=0 MLX_OMARCHY_KV_TRACE=1
    ;;
  4b)
    run_one 4b-d64 64
    run_one 4b-d512 512
    run_one 4b-d64-kvoff 64 MLX_OMARCHY_KV_DIRECT=0 MLX_OMARCHY_KV_TRACE=1
    "$PY" /var/tmp/dfuse/rope_norm_bitcheck4_qwen3.py "$OUT/iso4b.json" > "$OUT/iso4b.log" 2>&1 || echo "ISO4B FAILED"
    tail -1 "$OUT/iso4b.log"
    ;;
  9b)
    run_one 9b-d64 64
    run_one 9b-d512 512
    run_one 9b-d64-kvoff 64 MLX_OMARCHY_KV_DIRECT=0 MLX_OMARCHY_KV_TRACE=1
    ;;
esac

svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a /var/tmp/dfuse/w1-gates-$SET.txt
curl -s -m 5 http://127.0.0.1:8002/health > "$OUT/health-$SET.txt" 2>&1 || true
