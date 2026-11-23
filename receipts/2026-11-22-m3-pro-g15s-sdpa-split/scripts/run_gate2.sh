#!/usr/bin/env bash
# run_gate2.sh: the bf16 numerics-gate captures with the fixed prefill-perplexity check (MLX_OMARCHY_FULL_LOGITS=1 for the check only).
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}; TXT=$L/scripts/context-text.txt; G=$L/scripts/gpuleg.sh
for m in qwen3-4b-2507-4bit qwen3.8-2b-4bit; do for s in main head; do
  $G NG-$s-$m 3600 env HF_HUB_OFFLINE=1 $L/venv-$s/bin/python $L/scripts/numerics_gate.py $M/$m bf16 $L/runs/gate/NG-$s-$m.npz $(ls $L/dist-$s/*.whl) $R $TXT --validate
done; done
echo GATE2_DONE
