#!/usr/bin/env bash
# run_sweep.sh SIDE [TAG KV_W128 KV_W256]: the variant sweep on venv-SIDE, one process per (run, width, variant), each a gpuleg.sh timed leg (WAIT_LOAD=1).
# Width 128 = Qwen3-4B attention (32 query / 8 key-value heads); width 256 = Qwen3.8-2B full attention (8 / 2). Run 2 reverses the variant order.
set -uo pipefail
L=${LANE}
R=${REPO}
SIDE=$1; TAG=${2:-SW}; PY=$L/venv-$SIDE/bin/python; WH=$(ls $L/dist-$SIDE/*.whl)
KVA=${3:-128,192,256,320,384,448,512,640,768,1024,1536,2048,3072,4096,6144,8192,16384}; KVB=${4:-$KVA}
mkdir -p $L/runs/sweep
venv() {
  case $1 in
    default) echo "";;
    split) echo "MLX_OMARCHY_SDPA_DECODE_SPLIT=1 MLX_OMARCHY_SDPA_SPLIT_PV=0 MLX_OMARCHY_SDPA_SPLIT_BK64=0";;
    pv) echo "MLX_OMARCHY_SDPA_DECODE_SPLIT=1 MLX_OMARCHY_SDPA_SPLIT_PV=1 MLX_OMARCHY_SDPA_SPLIT_BK64=0";;
    bk64) echo "MLX_OMARCHY_SDPA_DECODE_SPLIT=1 MLX_OMARCHY_SDPA_SPLIT_PV=1 MLX_OMARCHY_SDPA_SPLIT_BK64=1";;
  esac
}
for run in 1 2; do
  for w in a b; do
    if [ $w = a ]; then shape="128 32 8"; vs="default split pv bk64"; KV=$KVA; else shape="256 8 2"; vs="default split pv"; KV=$KVB; fi
    [ $run = 2 ] && vs=$(echo $vs | tr ' ' '\n' | tac | tr '\n' ' ')
    for v in $vs; do
      n=$TAG-$SIDE-$w-$run-$v
      WAIT_LOAD=1 bash $L/scripts/gpuleg.sh $n 1800 env $(venv $v) $PY $L/scripts/sdpa_sweep.py 7 $shape $KV $L/runs/sweep/$n.json $WH $R
    done
  done
done
echo SWEEP_DONE
