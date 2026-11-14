#!/bin/bash
B=${BENCH:?}; D=$B/spec; O=$D/e2e.txt; V=$B/venv-spec/bin/python
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 TMPDIR=$B/tmp-pr MLX_OMARCHY_NO_MLXLM_WARNING=1
ICD=${ICD:?}
export VK_ICD_FILENAMES=$ICD VK_DRIVER_FILES=$ICD
unset MLX_OMARCHY_QMV_SMALL_T
. $B/prof/guard.sh
run(){ guard; echo "## $(date -Is) $* tmax=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)/1000 ))" >> $O
  flock /tmp/m1-gpu.lock timeout 1800 $V $D/e2e.py "$@" >> $O 2>&1; echo "rc=$?" >> $O; }
run mlx-community/Qwen3-8B-4bit 3
run mlx-community/Qwen3.5-4B-MLX-4bit 3
run mlx-community/Qwen3-8B-4bit 3 256 mlx-community/Qwen3-0.6B-4bit
run mlx-community/Qwen3.5-4B-MLX-4bit 3 256 mlx-community/Qwen3.5-0.8B-MLX-4bit
echo E2E_DONE >> $O
