#!/usr/bin/env bash
# run_nocpu.sh: the CPU-disabled head wheel: explicit CPU-stream add refused, the decode trace under count_cpu.gdb.py, and ab_decode.py at two lengths.
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt; G=$L/scripts/gpuleg.sh
GDB_CPU=$R/receipts/2026-09-30-speech-input-gpu/harness/count_cpu.gdb.py
PN=$L/venv-nocpu/bin/python; WN=$(ls $L/dist-nocpu/*.whl); mkdir -p $L/runs/nocpu
$G N00-gpu-add 120 $PN $L/scripts/cpu_refused.py gpu $WN $R
$G N01-cpu-refused 120 $PN $L/scripts/cpu_refused.py cpu $WN $R
for job in "qwen3-4b-2507-4bit 443" "qwen3-4b-2507-4bit 2048" "qwen3.8-2b-4bit 2048"; do set -- $job
  $G N1-trace-$1-$2 900 gdb -batch -x $GDB_CPU --args $PN $L/scripts/trace_split.py $M/$1 $2 32 $WN $R $TXT
done
for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do
  WAIT_LOAD=1 $G N2-ab-$m 3600 env HF_HUB_OFFLINE=1 CTX=443,2048 $PN $L/scripts/ab_decode.py $M/$m $L/runs/nocpu/N2-$m.json $L/runs/nocpu/N2-$m.npz $WN $R $TXT
done
echo NOCPU_DONE
