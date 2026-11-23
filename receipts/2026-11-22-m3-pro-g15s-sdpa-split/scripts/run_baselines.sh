#!/usr/bin/env bash
# run_baselines.sh: docs/parity-contract.md 4.2 PPL noise: five identical-protocol baseline captures (main bf16, numerics_gate.py) per model.
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt; G=$L/scripts/gpuleg.sh
mkdir -p $L/runs/baseline
for m in qwen3.8-2b-4bit qwen3-4b-2507-4bit; do for i in 1 2 3 4 5; do
  $G NB$i-main-$m 3600 env HF_HUB_OFFLINE=1 $L/venv-main/bin/python $L/scripts/numerics_gate.py $M/$m bf16 $L/runs/baseline/NB$i-main-$m.npz $(ls $L/dist-main/*.whl) $R $TXT
done; done
echo BASELINES_DONE
