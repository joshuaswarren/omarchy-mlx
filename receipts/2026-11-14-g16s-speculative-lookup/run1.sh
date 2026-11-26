#!/bin/bash
# round 1: prompt-lookup k=3, venv-r2 + prefix-fix ICD, Qwen3-8B then Qwen3.5-4B; arms SMALL_T=4 / unset
B=${BENCH:?}; D=$B/spec; O=$D/r1.txt
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 TMPDIR=$B/tmp-pr
ICD=${ICD:?}
export VK_ICD_FILENAMES=$ICD VK_DRIVER_FILES=$ICD
. $B/prof/guard.sh
for m in mlx-community/Qwen3-8B-4bit mlx-community/Qwen3.5-4B-MLX-4bit; do
 for arm in st4 st0; do
  guard
  echo "## $(date -Is) model=$m arm=$arm pmax=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq) tmax=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)/1000 ))" >> $O
  if [ $arm = st4 ]; then E="MLX_OMARCHY_QMV_SMALL_T=4"; else E="MLX_OMARCHY_QMV_SMALL_T="; fi
  flock /tmp/m1-gpu.lock timeout 1500 env $E $B/venv-r2/bin/python -I $D/spec.py $m --k 3 --max 256 >> $O 2>&1
  echo "rc=$?" >> $O
 done
done
echo R1_DONE >> $O
