#!/bin/bash
# usage: trace.sh <diag-variant> <tag> <repo> <max_tokens> [ENV=V ...]
# Dispatch trace of one greedy generation on a profiling wheel, analyzed
# with the same worktree's scripts/profile_analyze.py and compute.h.
U=$HOME/gpu/upr; V=$1; TAG=$2; REPO=$3; N=$4; shift 4
O=$U/traces/$V-$TAG; mkdir -p $O; rm -f $O/*
WT=$U/wt-$V
cd $U/probes
AB_NOWAIT=1 AB_ENV="MLX_OMARCHY_GPU_PROFILE=$O/profile.jsonl $*" python3 once.py $O/run.jsonl $V -- \
  trace_gen.py $WT --model $REPO --prompt "Explain in plain language how a rainbow forms. Give four distinct sentences." \
  --max-tokens $N --temp 0 --seed 0 --markers $O/markers.jsonl > $O/run.out 2>&1
echo "rc=$?" >> $O/run.out
python3 $WT/scripts/profile_analyze.py $O/profile.jsonl --markers $O/markers.jsonl --compute-h $WT/overlay/mlx/backend/omarchy/compute.h > $O/analysis.txt 2>&1
gzip -f $O/profile.jsonl
head -5 $O/run.out; grep -m3 -iE "decode|per token" $O/analysis.txt
