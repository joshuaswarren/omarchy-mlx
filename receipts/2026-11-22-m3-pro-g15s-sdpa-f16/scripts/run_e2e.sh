#!/usr/bin/env bash
# End-to-end f16 Qwen3-4B A/B: 3 interleaved pairs at budget unset (fallback 5000 on main), 3 at 15000.
set -uo pipefail
L=<lane>
WS=<base-build>
R=<repo>
W=$(ls $WS/dist-base/*.whl); PY=$WS/venv-base/bin/python
M=<models>/qwen3-4b-2507-4bit
i=0
run() { # budget arm
  local bw=$1 arm=$2 v=0; [ $arm = on ] && v=1
  i=$((i+1)); local name=$(printf 'E%02d-bw%s-%s' $i $bw $arm)
  local benv="-u MLX_OMARCHY_BATCH_WORK"; [ $bw != unset ] && benv="MLX_OMARCHY_BATCH_WORK=$bw"
  sleep 15
  $L/scripts/gpuleg.sh $name 1200 env $benv MLX_OMARCHY_NO_MLXLM_WARNING=1 HF_HUB_OFFLINE=1 MLX_OMARCHY_SDPA_F16_NO_COOPMAT=$v \
    $PY -I $L/scripts/e2e_f16.py $M $L/runs/e2e/$name.json $L/runs/e2e/$name.npz $W $R $name $L/scripts/context-text.txt \
    || { echo "FAILED $name" > $L/runs/e2e/MARKER; exit 1; }
}
for a in off on on off off on; do run unset $a; done
for a in on off off on on off; do run 15000 $a; done
echo "E2E_DONE $(date -u +%FT%TZ)" > $L/runs/e2e/MARKER
