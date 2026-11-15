#!/bin/bash
B=${BENCH:?}; D=$B/spec; O=$D/tcost.txt; V=$B/venv-spec/bin/python
export HF_HOME=$B/hf HF_HUB_OFFLINE=1 TMPDIR=$B/tmp-pr
ICD=${ICD:?}
export VK_ICD_FILENAMES=$ICD VK_DRIVER_FILES=$ICD
. $B/prof/guard.sh
for m in mlx-community/Qwen3-8B-4bit mlx-community/Qwen3.5-4B-MLX-4bit; do for st in 8 0; do
 guard; echo "## $(date -Is) $m st=$st" >> $O
 flock /tmp/m1-gpu.lock env MLX_OMARCHY_QMV_SMALL_T=$st $V -I $D/tcost.py $m >> $O 2>&1
done; done; echo TC_DONE >> $O
