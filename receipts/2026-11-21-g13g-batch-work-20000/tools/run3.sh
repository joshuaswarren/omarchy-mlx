#!/usr/bin/env bash
# wAZ: G13G batch-work tail check on Qwen3-4B 4-bit (the calibration model): tok/s, submits/token, ids at default 8000, 20000, 40000,
# 60000, 80000 (balanced), then read-only ftrace of per-submission GPU time at each. v0.7.49. run3.sh OUTDIR
set -euo pipefail
OUT=$1; D=$(dirname "$0"); PY=$VENV/bin/python; M=$MODEL_4B; T=/sys/kernel/tracing
export MLX_OMARCHY_NO_MLXLM_WARNING=1
mkdir -p "$OUT"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load=$(cut -d' ' -f1 /proc/loadavg)" | tee "$OUT/run.log"
r(){ local tag=$1 w=$2; if [ "$w" = def ]; then "$PY" "$D/decode_sub.py" "$M" "$OUT/$tag.json" "$tag" 96; else MLX_OMARCHY_BATCH_WORK=$w "$PY" "$D/decode_sub.py" "$M" "$OUT/$tag.json" "$tag" 96; fi 2>&1 | tail -n 1 | tee -a "$OUT/run.log"; }
i=0; for w in def 20000 40000 60000 80000 80000 60000 40000 20000 def; do i=$((i+1)); r "w${w}_$i" $w; done
tr(){ local w=$1
  sudo -n sh -c "echo 0 > $T/tracing_on; echo > $T/trace; echo mono > $T/trace_clock; echo 32768 > $T/buffer_size_kb; echo > $T/set_event; echo 1 > $T/events/gpu_scheduler/enable; echo 1 > $T/tracing_on"
  if [ "$w" = def ]; then "$PY" "$D/decode_jobs.py" "$M" 64; else MLX_OMARCHY_BATCH_WORK=$w "$PY" "$D/decode_jobs.py" "$M" 64; fi > "$OUT/dj-w$w.txt" 2>&1 || echo "DJ $w rc=$?" | tee -a "$OUT/run.log"
  sudo -n sh -c "echo 0 > $T/tracing_on; cat $T/trace > '$OUT/ft-w$w.txt'; echo 0 > $T/events/enable; echo > $T/trace; echo local > $T/trace_clock"; }
trap 'sudo -n sh -c "echo 0 > $T/tracing_on; echo 0 > $T/events/enable; echo local > $T/trace_clock" || true' EXIT
for w in def 20000 40000 60000 80000; do tr $w; done
echo WAZ_SUB3_DONE | tee -a "$OUT/run.log"
