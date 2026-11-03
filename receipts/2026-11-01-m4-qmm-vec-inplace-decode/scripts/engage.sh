#!/bin/bash
# Route engagement of the bit probes on a profiling wheel: which kernels
# each probe dispatched. usage: engage.sh <diag-arm>
U=$HOME/gpu/upr; A=$1; WT=$U/wt-$A; cd $U/probes; O=$U/logs/engage-$A; mkdir -p $O
export AB_NOWAIT=1
for p in qmv rope sdpa qmm_fma; do
  env="MLX_OMARCHY_GPU_PROFILE=$O/$p.jsonl"; [ $p = qmm_fma ] && env="$env HK_APPLE9_COOPMAT=0"
  rm -f $O/$p.jsonl
  AB_ENV="$env" python3 once.py $O/run.jsonl $A -- ${p}_bits.py dump $O/$p.npz > $O/$p.out 2>&1
  (cd $WT/scripts && python3 $U/probes/kernels_used.py $O/$p.jsonl $WT/overlay/mlx/backend/omarchy/compute.h) > $O/$p.kernels.txt 2>&1
  gzip -f $O/$p.jsonl; rm -f $O/$p.npz
done
for p in qmv rope sdpa qmm_fma; do echo "== $p"; head -14 $O/$p.kernels.txt; done
