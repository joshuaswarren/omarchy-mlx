#!/bin/bash
# #317 diagnostic (not part of the gate): route engagement and bits of the bf16 linear matmul.
cd $HOME/gpu/upr/probes; O=$HOME/gpu/upr/gate/out/diag317
python3 once.py $O/runs.jsonl p2 -- linprobe317.py > $O/p2-default.out 2>&1
AB_ENV="MLX_OMARCHY_NO_MATMUL_FMA=1" python3 once.py $O/runs.jsonl p2 -- linprobe317.py > $O/p2-no-matmul-fma.out 2>&1
python3 once.py $O/runs.jsonl main -- linprobe317.py > $O/main.out 2>&1
echo "DIAG317_DONE $(date -u +%FT%TZ)"
