#!/usr/bin/env bash
# The PR's scenario (switch and budget unset) on newest green main and on the merged tree: per-call probe, then one e2e process each.
set -uo pipefail
L=${LANE}
M=${MODELS}/qwen3-4b-2507-4bit
G=$L/scripts/gpuleg.sh
mkdir -p $L/runs/merged2
for side in main2 merged2; do
  W=$L/scratch/wt-$side; PY=$L/venv-$side/bin/python; WH=$(ls $L/dist-$side/*.whl)
  WAIT_LOAD=1 $G M3-percall-$side 600 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK SDPA_AB_SHAPES=r2 $PY -I $L/scripts/sdpa_f16_ab.py $L/runs/merged2/M3-$side.json $L/runs/merged2/M3-$side.npz $WH $W M3-$side
done
for side in main2 merged2; do
  W=$L/scratch/wt-$side; PY=$L/venv-$side/bin/python; WH=$(ls $L/dist-$side/*.whl)
  sleep 15
  WAIT_LOAD=1 $G M4-e2e-$side 1200 env -u MLX_OMARCHY_SDPA_F16_NO_COOPMAT -u MLX_OMARCHY_BATCH_WORK E2E_CTX=64,512,1100 MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 $PY -I $L/scripts/e2e_f16.py $M $L/runs/merged2/M4-$side.json $L/runs/merged2/M4-$side.npz $WH $W M4-$side $L/scripts/context-text.txt
done
echo "MERGED_DONE $(date -u +%FT%TZ)" > $L/runs/merged2/MARKER
