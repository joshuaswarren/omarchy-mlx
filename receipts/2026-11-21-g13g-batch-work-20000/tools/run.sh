#!/usr/bin/env bash
# wAZ: decode submissions vs MLX_OMARCHY_BATCH_WORK on the 13-inch M1 (G13G table row 8000), v0.7.49, Qwen3.8-2B 4-bit.
# Part 1: tok/s, submits/token, ids for default / 0 (cap off) / 40000, balanced ABBA-style order, 128 tokens.
# Part 2: read-only ftrace of 64 decode tokens per setting: per-submission GPU job time (drm_sched run->done) p50/p99/max. run.sh OUTDIR
set -euo pipefail
OUT=$1; D=$(dirname "$0"); PY=$VENV/bin/python; M=$MODEL_2B; T=/sys/kernel/tracing
export MLX_OMARCHY_NO_MLXLM_WARNING=1
mkdir -p "$OUT"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) kernel=$(uname -r) mesa=$(pacman -Q vulkan-asahi-omarchy | tr ' ' '=') load=$(cut -d' ' -f1 /proc/loadavg)" | tee "$OUT/run.log"
r(){ local tag=$1; shift; env "$@" "$PY" "$D/decode_sub.py" "$M" "$OUT/$tag.json" "$tag" 128 2>&1 | tail -n 1 | tee -a "$OUT/run.log"; }
i=0; for o in def off w40k w40k off def def off w40k w40k off def; do i=$((i+1)); case $o in def) r def_$i X=1;; off) r off_$i MLX_OMARCHY_BATCH_WORK=0;; w40k) r w40k_$i MLX_OMARCHY_BATCH_WORK=40000;; esac; done
tr(){ local tag=$1; shift
  sudo -n sh -c "echo 0 > $T/tracing_on; echo > $T/trace; echo mono > $T/trace_clock; echo 32768 > $T/buffer_size_kb; echo > $T/set_event; echo 1 > $T/events/gpu_scheduler/enable; echo 1 > $T/tracing_on"
  env "$@" "$PY" "$D/decode_jobs.py" "$M" 64 > "$OUT/dj-$tag.txt" 2>&1 || echo "DJ $tag rc=$?" | tee -a "$OUT/run.log"
  sudo -n sh -c "echo 0 > $T/tracing_on; cat $T/trace > '$OUT/ft-$tag.txt'; echo 0 > $T/events/enable; echo > $T/trace; echo local > $T/trace_clock"
  grep ^DECODE "$OUT/dj-$tag.txt" | sed "s/^/$tag /" | tee -a "$OUT/run.log"; }
trap 'sudo -n sh -c "echo 0 > $T/tracing_on; echo 0 > $T/events/enable; echo local > $T/trace_clock" || true' EXIT
tr def X=1; tr off MLX_OMARCHY_BATCH_WORK=0; tr w40k MLX_OMARCHY_BATCH_WORK=40000
echo WAZ_SUB_DONE | tee -a "$OUT/run.log"
