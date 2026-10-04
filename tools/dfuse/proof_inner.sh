#!/usr/bin/env bash
# DispatchFuse proof inner: ONE census run with gates, INSIDE a gpuwin window.
# Env: PROOF_PY, PROOF_OUT, PROOF_MODEL(2b|4b|9b), PROOF_DEPTH, PROOF_NAME,
#      PROOF_ENV (comma-separated K=V pairs, may be empty).
set -euo pipefail
PROOF_PY=${PROOF_PY:?missing}
PROOF_OUT=${PROOF_OUT:?missing}
PROOF_MODEL=${PROOF_MODEL:?missing}
PROOF_DEPTH=${PROOF_DEPTH:?missing}
PROOF_NAME=${PROOF_NAME:?missing}
PROOF_ENV=${PROOF_ENV:-}

test ! -e /var/tmp/JW16_MAINTENANCE
uptime_s=$(cut -d. -f1 /proc/uptime); test "$uptime_s" -ge 360 || { echo "boot settle gate: ${uptime_s}s"; exit 1; }
load1=$(cut -d" " -f1 /proc/loadavg)
li=${load1%%.*}; lf=${load1#*.}00
(( $((10#$li * 100 + 10#${lf:0:2})) < 50 )) || { echo "load gate: $load1"; exit 1; }
psi=$(cat /proc/pressure/cpu)
case "$psi" in *"some avg10=0.00"*) ;; *) echo "PSI gate: $psi"; exit 1;; esac
mkdir -p "$PROOF_OUT"
printf "host=%s boot=%s uptime_s=%s load1=%s psi=%s\n" "$(hostname)" \
  "$(cat /proc/sys/kernel/random/boot_id)" "$uptime_s" "$load1" "$psi" \
  | tee "$PROOF_OUT/gates-$PROOF_NAME.txt"

HF=${HOME}/.cache/huggingface/hub
case "$PROOF_MODEL" in
  2b) M=$HF/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381 ;;
  4b) M=$(ls -d $HF/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*/ | head -1) ;;
  9b) M=$(ls -d $HF/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/ | head -1) ;;
  *) echo "bad model"; exit 2 ;;
esac

EXTRA=()
[ -n "$PROOF_ENV" ] && IFS=',' read -ra EXTRA <<< "$PROOF_ENV"
timeout 600 env MLX_OMARCHY_GPU_PROFILE="$PROOF_OUT/prof-$PROOF_NAME.ndjson" \
  MLX_OMARCHY_GPU_PROFILE_LABEL="$PROOF_NAME" "${EXTRA[@]}" \
  "$PROOF_PY" "$HOME/bench-scripts/qwen38-mlx-bench.py" --model "$M" \
    --prompts "$HOME/bench-scripts/qwen38-2b-prompts.jsonl" \
    --limit 1 --warmup 1 --passes 1 --new-tokens "$PROOF_DEPTH" \
    --prefill-tokens 512 --label "$PROOF_NAME" --out "$PROOF_OUT/$PROOF_NAME.json" \
  > "$PROOF_OUT/$PROOF_NAME.log" 2> "$PROOF_OUT/$PROOF_NAME.err" \
  || { echo "PROOF RUN FAILED $PROOF_NAME rc=$?"; exit 1; }
echo "done $PROOF_NAME $(grep -o '"ordered_records_sha256[^,}]*' "$PROOF_OUT/$PROOF_NAME.json" | head -1)"
svc=$(systemctl is-active llm-inference || true)
printf "post-window llm-inference: %s\n" "$svc" | tee -a "$PROOF_OUT/gates-$PROOF_NAME.txt"
