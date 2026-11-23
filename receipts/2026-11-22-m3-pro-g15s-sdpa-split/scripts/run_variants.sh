#!/usr/bin/env bash
# run_variants.sh SIDE: variant_decode.py per (model, variant) on venv-SIDE, each a gpuleg.sh timed leg (WAIT_LOAD=1).
# Width 128 model: split, pv, bk64; width 256 model: split, pv (bk64 is width 128 only).
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}
SIDE=$1; PY=$L/venv-$SIDE/bin/python; WH=$(ls $L/dist-$SIDE/*.whl)
mkdir -p $L/runs/variants
for job in "qwen3-4b-2507-4bit split" "qwen3-4b-2507-4bit pv" "qwen3-4b-2507-4bit bk64" "qwen3.8-2b-4bit split" "qwen3.8-2b-4bit pv"; do
  set -- $job
  n=VD-$SIDE-$1-$2
  WAIT_LOAD=1 bash $L/scripts/gpuleg.sh $n 3600 $PY $L/scripts/variant_decode.py $M/$1 $2 $L/runs/variants/$n.json $WH $R $L/scripts/context-text.txt
done
echo VARIANTS_DONE
