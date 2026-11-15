#!/usr/bin/env bash
# M2 Linux (G14C), 2B: default vs 120000, 4 pairs interleaved, probe beside.
# 600-token sustained decode, 6 processes per arm on 4B, 4 per arm on 2B, interleaved, 20 s gaps, pacing probe beside each.
# Scheduling only: ids_sha must match across arms.
#   w9c-linux-batchwork24.sh OUTDIR
set -uo pipefail
test "$(hostname)" = jwm2-linux
test ! -f /var/tmp/ane-hold-m2
OUT=$1
Q=$HOME/scratch/w9c-quietrows
H=$HOME/.cache/huggingface/hub
PY=$HOME/scratch/w9c-dt-dtc/venv/bin/python
export VK_DRIVER_FILES=$HOME/scratch/w9c-mesa-main/asahi_icd.main.json
mkdir -p "$OUT/pace"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) load=$(cut -d' ' -f1 /proc/loadavg)" | tee "$OUT/run.log"
g++ -x c++ -O2 -o "$OUT/submit-latency" "$Q/submit-latency.c" -lvulkan || { echo "PROBE_BUILD_FAILED" | tee -a "$OUT/run.log"; exit 1; }
for spec in models--SiddhJagani--Qwen3.8-2B-mlx-4Bit:4 ; do
  m=${spec%%:*}
  n=${spec##*:}
  SNAP=$(echo "$H/$m"/snapshots/*)
  for i in $(seq 1 "$n"); do
    for arm in default 120000; do
      echo "== $m $arm p$i $(date -u +%T)" >> "$OUT/run.log"
      if [ "$arm" = default ]; then
        r=$("$PY" "$Q/w9c_pace.py" "$SNAP" "$OUT/pace/$m-$arm-p$i.json" "$OUT/submit-latency" 600 2>&1 | tail -n 1)
      else
        r=$(MLX_OMARCHY_BATCH_WORK=$arm "$PY" "$Q/w9c_pace.py" "$SNAP" "$OUT/pace/$m-$arm-p$i.json" "$OUT/submit-latency" 600 2>&1 | tail -n 1)
      fi
      echo "$arm $r" >> "$OUT/run.log"
      sleep 20
    done
  done
done
echo BATCHWORK2B_DONE | tee -a "$OUT/run.log"
