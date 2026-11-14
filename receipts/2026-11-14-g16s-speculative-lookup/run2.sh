#!/bin/bash
# round 2: venv-spec (G16S cap 8). k sweep with SMALL_T, plus draft models
B=${BENCH:?}; D=$B/spec; O=$D/r2.txt; V=$B/venv-spec/bin/python
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 TMPDIR=$B/tmp-pr
ICD=${ICD:?}
export VK_ICD_FILENAMES=$ICD VK_DRIVER_FILES=$ICD
. $B/prof/guard.sh
run(){ # model st k extra...
  m=$1; st=$2; k=$3; shift 3; guard
  echo "## $(date -Is) model=$m st=$st k=$k $* tmax=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)/1000 ))" >> $O
  flock /tmp/m1-gpu.lock timeout 1800 env MLX_OMARCHY_QMV_SMALL_T=$st $V -I $D/spec.py $m --k $k --max 256 "$@" >> $O 2>&1; echo "rc=$?" >> $O; }
Q8=mlx-community/Qwen3-8B-4bit; Q35=mlx-community/Qwen3.5-4B-MLX-4bit
run $Q8 4 3; run $Q8 8 3; run $Q8 8 5; run $Q8 8 7
run $Q35 4 3; run $Q35 8 5; run $Q35 8 7
run $Q8 8 3 --draft mlx-community/Qwen3-0.6B-4bit; run $Q8 8 5 --draft mlx-community/Qwen3-0.6B-4bit
run $Q35 8 3 --draft mlx-community/Qwen3.5-0.8B-MLX-4bit; run $Q35 8 5 --draft mlx-community/Qwen3.5-0.8B-MLX-4bit
echo R2_DONE >> $O
