#!/bin/bash
# Diagnostic only (not part of the gate): which #316 route moves Qwen S=1 PPL.
U=$HOME/gpu/upr; O=$U/gate/out; H=$HOME/.cache/huggingface/hub
M=$(ls -d $H/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/*); T=$O/teacher-Qwen2.5-0.5B-Instruct-4bit.json
export AB_NOWAIT=1; cd $U/probes
for c in "both_on:MLX_OMARCHY_FUSED_GEMV=1 MLX_OMARCHY_FUSED_TRIO=1" "gemv_on:MLX_OMARCHY_FUSED_GEMV=1" "trio_on:MLX_OMARCHY_FUSED_TRIO=1"; do
  n=${c%%:*}; e=${c#*:}; d=$O/diag316/$n; mkdir -p $d
  ls $d/*.json >/dev/null 2>&1 && continue
  AB_ENV="$e" python3 once.py $O/diag316/runs.jsonl p1 -- gate_tf.py measure --model "$M" --teacher "$T" --chunk 1 --outdir $d --label s1
  echo "$n done $(date +%H:%M)" >> $O/diag316/progress.txt
done
