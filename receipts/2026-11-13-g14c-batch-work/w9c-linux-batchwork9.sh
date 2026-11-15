#!/usr/bin/env bash
# M2 Linux (G14C), tracker-fix build (dtc). 9B submission budget confirmation: MLX_OMARCHY_BATCH_WORK unset (table, 40000) vs 120000 vs 200000,
# 600-token sustained decode, 6 processes per arm interleaved, 20 s gaps, with the 60 Hz pacing probe beside each. Then the probe positive control:
# long chained f16 matmul evals with the cap off (one long submission) and with the default cap, probe beside each, 3 runs each.
# Scheduling only: ids_sha must match across arms.
#   w9c-linux-batchwork9.sh OUTDIR
set -uo pipefail
test "$(hostname)" = jwm2-linux
test ! -f /var/tmp/ane-hold-m2
OUT=$1
Q=$HOME/scratch/w9c-quietrows
H=$HOME/.cache/huggingface/hub
PY=$HOME/scratch/w9c-dt-dtc/venv/bin/python
export VK_DRIVER_FILES=$HOME/scratch/w9c-mesa-main/asahi_icd.main.json
mkdir -p "$OUT/pace" "$OUT/ctl"
echo "PREFLIGHT host=$(hostname) boot=$(cat /proc/sys/kernel/random/boot_id) load=$(cut -d' ' -f1 /proc/loadavg)" | tee "$OUT/run.log"
g++ -x c++ -O2 -o "$OUT/submit-latency" "$Q/submit-latency.c" -lvulkan || { echo "PROBE_BUILD_FAILED" | tee -a "$OUT/run.log"; exit 1; }
m=models--nvythong--Qwen3.8-9B-Distill-mlx-4Bit
SNAP=$(echo "$H/$m"/snapshots/*)
for i in 1 2 3 4 5 6; do
  for arm in default 120000 200000; do
    echo "== $arm p$i $(date -u +%T)" >> "$OUT/run.log"
    if [ "$arm" = default ]; then
      r=$("$PY" "$Q/w9c_pace.py" "$SNAP" "$OUT/pace/$m-$arm-p$i.json" "$OUT/submit-latency" 600 2>&1 | tail -n 1)
    else
      r=$(MLX_OMARCHY_BATCH_WORK=$arm "$PY" "$Q/w9c_pace.py" "$SNAP" "$OUT/pace/$m-$arm-p$i.json" "$OUT/submit-latency" 600 2>&1 | tail -n 1)
    fi
    echo "$arm $r" >> "$OUT/run.log"
    sleep 20
  done
done
for i in 1 2 3; do
  echo "== ctl capoff p$i $(date -u +%T)" >> "$OUT/run.log"
  echo "ctl-capoff $(MLX_OMARCHY_BATCH_WORK=0 "$PY" "$Q/w9c_pacectl.py" "$OUT/ctl/capoff-p$i.json" "$OUT/submit-latency" 2>&1 | tail -n 1)" >> "$OUT/run.log"
  sleep 15
  echo "== ctl default p$i $(date -u +%T)" >> "$OUT/run.log"
  echo "ctl-default $("$PY" "$Q/w9c_pacectl.py" "$OUT/ctl/default-p$i.json" "$OUT/submit-latency" 2>&1 | tail -n 1)" >> "$OUT/run.log"
  sleep 15
done
echo BATCHWORK9_DONE | tee -a "$OUT/run.log"
