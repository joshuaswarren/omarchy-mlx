#!/bin/bash
B=${BENCH:?}; D=$B/spec; O=$D/qmvT.txt; V=$B/venv-spec/bin/python
export TMPDIR=$B/tmp-pr; ICD=${ICD:?}; export VK_ICD_FILENAMES=$ICD VK_DRIVER_FILES=$ICD
. $B/prof/guard.sh; guard; echo "## $(date -Is) st=8" >> $O
flock /tmp/m1-gpu.lock env MLX_OMARCHY_QMV_SMALL_T=8 $V -I $D/qmvT.py >> $O 2>&1; echo QT_DONE >> $O
