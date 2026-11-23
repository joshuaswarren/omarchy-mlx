#!/usr/bin/env bash
# run_gate.sh: numerics-gate captures (numerics_gate.py) per model: float32 reference on the main wheel, main bf16 and head bf16 (both with
# the prefill-perplexity harness check), then the layout sweep: default route of main and head per call at head layouts outside the
# calibration shapes (rounds main head head main).
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt; G=$L/scripts/gpuleg.sh
py() { echo $L/venv-$1/bin/python; }; wh() { ls $L/dist-$1/*.whl; }
mkdir -p $L/runs/gate $L/runs/layout
for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do
  $G NG-fp32-$m 3600 env HF_HUB_OFFLINE=1 $(py main) $L/scripts/numerics_gate.py $M/$m fp32 $L/runs/gate/NG-fp32-$m.npz $(wh main) $R $TXT
  for s in main head; do
    $G NG-$s-$m 3600 env HF_HUB_OFFLINE=1 $(py $s) $L/scripts/numerics_gate.py $M/$m bf16 $L/runs/gate/NG-$s-$m.npz $(wh $s) $R $TXT --validate
  done
done
KA=128,192,256,320,448,640,641,1024,1280,1281,2048,4096
KB=1024,1280,1407,1408,1536,2048,2304,2305,3072,4096
i=0
for s in main head head main; do i=$((i+1))
  for shape in "128 8 8 g1" "128 16 8 g2" "128 64 8 g8" "256 16 4 kv4" "256 32 8 kv8"; do set -- $shape
    K=$KA; [ $1 = 256 ] && K=$KB
    WAIT_LOAD=1 $G LY$i-$s-$4 1800 $(py $s) $L/scripts/sdpa_sweep.py 7 $1 $2 $3 $K $L/runs/layout/LY$i-$s-$4.json $(wh $s) $R
  done
done
echo GATE_DONE
