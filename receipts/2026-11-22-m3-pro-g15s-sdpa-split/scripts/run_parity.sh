#!/usr/bin/env bash
# run_parity.sh SIDE JOB...: variant_decode.py (JOB = MODEL:VARIANT) and parity_ref.py (JOB = MODEL:ref:ARMS) legs on venv-SIDE, each a gpuleg.sh timed leg.
set -uo pipefail
L=${LANE}
R=${REPO}
M=${MODELS}
SIDE=$1; shift; PY=$L/venv-$SIDE/bin/python; WH=$(ls $L/dist-$SIDE/*.whl)
mkdir -p $L/runs/variants $L/runs/parity
for job in "$@"; do
  IFS=: read -r m v arms <<< "$job"
  if [ "$v" = ref ]; then
    n=PR-$SIDE-$m
    WAIT_LOAD=1 bash $L/scripts/gpuleg.sh $n 3600 $PY $L/scripts/parity_ref.py $M/$m $arms $L/runs/parity/$n.json $WH $R $L/scripts/context-text.txt
  else
    n=VD-$SIDE-$m-$v
    WAIT_LOAD=1 bash $L/scripts/gpuleg.sh $n 3600 $PY $L/scripts/variant_decode.py $M/$m $v $L/runs/variants/$n.json $WH $R $L/scripts/context-text.txt
  fi
done
echo PARITY_DONE
