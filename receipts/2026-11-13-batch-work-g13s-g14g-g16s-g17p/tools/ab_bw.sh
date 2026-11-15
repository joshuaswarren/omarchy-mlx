#!/bin/bash
# usage: ab_bw.sh <label> <budget> : release wheel, default budget vs MLX_OMARCHY_BATCH_WORK=<budget>, both models, score_bench protocol
cd $HOME/mlxpr; export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
for arm in default $2; do
  for m in SiddhJagani/Qwen3.8-2B-mlx-4Bit mlx-community/Qwen3-4B-Instruct-2507-4bit; do
    k=$(basename $m)
    if [ $arm = default ]; then unset MLX_OMARCHY_BATCH_WORK; else export MLX_OMARCHY_BATCH_WORK=$arm; fi
    echo "$1 arm=$arm $k load $(cut -d' ' -f1 /proc/loadavg) temp $(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)"
    venv/bin/python score_bench.py --model $m --prompts prompts.jsonl --label $1-bw$arm --out results/ab-$1-bw$arm-$k.json 2>&1 | tail -n1
  done
done
