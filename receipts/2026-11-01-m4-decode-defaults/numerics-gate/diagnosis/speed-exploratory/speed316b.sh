#!/bin/bash
O=$HOME/gpu/upr/gate/out/speed316; cd $HOME/gpu/upr/probes
left=$(( $(date -d 15:22 +%s) - $(date +%s) ))
echo "$(date +%H:%M) start Llama-3.2-3B (${left}s)" >> $O/log.txt
AB_ENV_B="MLX_OMARCHY_FUSED_TRIO=1" timeout $left python3 ab.py $O/ab-Llama-3.2-3B-Instruct-4bit.jsonl 2 main p1 -- decode_bench.py mlx-community/Llama-3.2-3B-Instruct-4bit 64 4 >> $O/log.txt 2>&1
echo "$(date +%H:%M) Llama-3.2-3B end" >> $O/log.txt
