#!/bin/bash
# Exploratory speed A/B (not a receipt): main vs #316 with only FUSED_GEMV off (MLX_OMARCHY_FUSED_TRIO=1).
O=$HOME/gpu/upr/gate/out/speed316; mkdir -p $O; L=$O/log.txt
end=$(date -d "15:22" +%s)
while pgrep -f "gate_tf.py measure --cpu" >/dev/null || pgrep -f "finish317.sh" >/dev/null; do
  [ $(date +%s) -ge $(date -d "15:10" +%s) ] && { echo "$(date +%H:%M) ref still running; using AB_NOWAIT (CPU busy, numbers noisy)" >> $L; export AB_NOWAIT=1; break; }
  sleep 15
done
echo "$(date +%H:%M) start" >> $L
cd $HOME/gpu/upr/probes
for m in Qwen2.5-0.5B-Instruct-4bit Qwen2.5-1.5B-Instruct-4bit granite-4.0-1b-4bit; do
  left=$(( end - $(date +%s) )); [ $left -lt 120 ] && { echo "$(date +%H:%M) out of time before $m" >> $L; break; }
  AB_ENV_B="MLX_OMARCHY_FUSED_TRIO=1" timeout $left python3 ab.py $O/ab-$m.jsonl 2 main p1 -- decode_bench.py mlx-community/$m 64 4 >> $L 2>&1
  echo "$(date +%H:%M) $m done" >> $L
done
echo "$(date +%H:%M) end" >> $L
