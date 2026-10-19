#!/bin/bash
# Two distinct verified builds, default environment in both: O = mlx_omarchy 49e6b3c1 (before this change), N = mlx_omarchy a2b0864 (this change).
# Qwen3.6-35B-A3B-4bit on the 16-inch M1 Max (G13C). 5 rounds, 2 prompts, 64 tokens, order alternating, fresh process each.
set -uo pipefail
D=$HOME/scratch/w7b-gqmm/ab35
SC=$HOME/scratch/w7b-gqmm/tile35
VO=$HOME/scratch/w7q/step1-49e6b3c1b/venv
VN=$HOME/scratch/w7b-gqmm/verify35/venv
WO=$(find "$HOME/scratch/w7q/step1-49e6b3c1b/wheel" -name 'mlx_omarchy-*+49e6b3c1-*.whl' | head -1)
WN=$(find "$HOME/.local/share/coreglass/build/dist-bf16a" -name 'mlx_omarchy-*+a2b0864-*.whl' | head -1)
O=$D/out-1
MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "AB $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "O wheel $(basename "$WO") sha256 $(sha256sum "$WO" | cut -c1-64) libmlx $(sha256sum "$VO"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
say "N wheel $(basename "$WN") sha256 $(sha256sum "$WN" | cut -c1-64) libmlx $(sha256sum "$VN"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
for i in 1 2 3 4 5; do
  for pn in P1 P2; do
    p=$P1; [ $pn = P2 ] && p=$P2
    if [ $((i % 2)) = 1 ]; then order="O N"; else order="N O"; fi
    for arm in $order; do
      if [ $arm = O ]; then vp=$VO; w=$WO; else vp=$VN; w=$WN; fi
      env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 BENCH_SCRIPTS_DIR="$SC/scripts" timeout 600 "$vp/bin/python" "$SC/scripts/bench_decode.py" --model "$MODEL" \
        --prompt "$p" --tokens 64 --wheel "$w" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*') $(grep -a -o 'mx=[^ ]*\|version[= ][^ ]*' "$O/t-$pn-$arm-$i.out" | head -1)"
    done
  done
done
say "end $(date -u +%FT%TZ)"
