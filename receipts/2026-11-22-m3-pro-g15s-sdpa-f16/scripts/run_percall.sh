#!/usr/bin/env bash
# Per-call A/B, 3 interleaved pairs (off on on off off on), budget unset (fallback 5000 on main).
set -uo pipefail
L=<lane>
WS=<base-build>
R=<repo>
W=$(ls $WS/dist-base/*.whl); PY=$WS/venv-base/bin/python
mkdir -p $L/runs/percall
i=0
for arm in off on on off off on; do
  i=$((i+1)); v=0; [ $arm = on ] && v=1
  name=$(printf 'B%02d-percall-%s' $i $arm)
  sleep 10
  $L/scripts/gpuleg.sh $name 600 env -u MLX_OMARCHY_BATCH_WORK MLX_OMARCHY_SDPA_F16_NO_COOPMAT=$v $PY -I $L/scripts/sdpa_f16_ab.py \
    $L/runs/percall/$name.json $L/runs/percall/$name.npz $W $R $name || { echo "FAILED $name" > $L/runs/percall/MARKER; exit 1; }
done
echo "PERCALL_DONE $(date -u +%FT%TZ)" > $L/runs/percall/MARKER
