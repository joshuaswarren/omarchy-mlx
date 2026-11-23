#!/usr/bin/env bash
# run_ab.sh: two-build A/B, main wheel against head wheel, every MLX_OMARCHY_SDPA variable unset, one process per run, timed legs (WAIT_LOAD=1).
#   AB: ab_decode.py, two models, rounds ordered main head head main.
#   DF: sdpa_sweep.py default route of each build at both widths, same order.
#   TR: dispatch trace under count_cpu.gdb.py (CPU-stream control, then decode on each build).
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt; G=$L/scripts/gpuleg.sh
GDB_CPU=$R/receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py
KV=128,192,256,320,448,640,641,1024,1280,1281,1408,2048,2304,2305,3072,4096,8192,16384
mkdir -p $L/runs/ab $L/runs/df $L/runs/trace
py() { echo $L/venv-$1/bin/python; }; wh() { ls $L/dist-$1/*.whl; }
for s in main head; do $(py $s) $R/scripts/mlx_provenance.py --expect-wheel $(wh $s) | grep -E '"(dist_version|verified|harness_commit)"'; done
i=0
for s in main head head main; do i=$((i+1))
  for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do
    n=AB$i-$s-$m
    WAIT_LOAD=1 $G $n 3600 env HF_HUB_OFFLINE=1 $(py $s) $L/scripts/ab_decode.py $M/$m $L/runs/ab/$n.json $L/runs/ab/$n.npz $(wh $s) $R $TXT
  done
done
i=0
for s in main head head main; do i=$((i+1))
  WAIT_LOAD=1 $G DF$i-$s-a 1800 $(py $s) $L/scripts/sdpa_sweep.py 7 128 32 8 $KV $L/runs/df/DF$i-$s-a.json $(wh $s) $R
  WAIT_LOAD=1 $G DF$i-$s-b 1800 $(py $s) $L/scripts/sdpa_sweep.py 7 256 8 2 $KV $L/runs/df/DF$i-$s-b.json $(wh $s) $R
done
$G TR0-cpu-control 600 gdb -batch -x $GDB_CPU --args $(py head) $L/scripts/trace_split.py $M/qwen3-4b-2507-4bit 1 1 $(wh head) $R $TXT --cpu-control
for s in main head; do
  for job in "qwen3-4b-2507-4bit 443" "qwen3-4b-2507-4bit 2048" "qwen3.8-2b-4bit 2048"; do set -- $job
    $G TR-$s-$1-$2 900 gdb -batch -x $GDB_CPU --args $(py $s) $L/scripts/trace_split.py $M/$1 $2 32 $(wh $s) $R $TXT
  done
done
echo AB_DONE
