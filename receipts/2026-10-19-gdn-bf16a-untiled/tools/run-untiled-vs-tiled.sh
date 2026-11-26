#!/bin/bash
# 16-inch M1 Max (G13C): Qwen3.6-35B-A3B-4bit (bf16 A_log GDN route, kernel id 455 by default). Arms: D = default, U = MLX_OMARCHY_GDN_DECODE_TILE=0.
# usage: runfold.sh num|time TAG     num = numerics + route census (no timing claim); time = 5 rounds x 2 prompts under /tmp/m1-gpu.lock
set -uo pipefail
PART=${1:?num|time}
TAG=${2:?tag}
B0=$HOME/scratch/w7q/step1-49e6b3c1b
VP=$B0/venv
WHL=$(find "$B0/wheel" -maxdepth 1 -name 'mlx_omarchy-*+49e6b3c1-cp314-cp314-linux_aarch64.whl' | head -1)
D=$HOME/scratch/w7b-gqmm/tile35
O=$D/out-$PART-$TAG
MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
[ -n "$WHL" ] || { echo "ABORT no wheel"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "T35 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) part=$PART kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1) libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) driver $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$D/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$D/gdn_logits.py" | cut -c1-16)"
export HF_HUB_OFFLINE=1
env_of() { case $1 in D|D2) echo "MLX_AB_UNUSED=1" ;; U) echo "MLX_OMARCHY_GDN_DECODE_TILE=0" ;; esac; }
prompt_of() { case $1 in P1) echo "$P1" ;; P2) echo "$P2" ;; esac; }
if [ "$PART" = num ]; then
  for run in "P1 D" "P1 U" "P1 D2" "P2 D" "P2 U"; do
    set -- $run
    # shellcheck disable=SC2046
    env $(env_of "$2") timeout 900 "$VP/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$(prompt_of "$1")" --tokens 128 --full-steps 32 \
      --out "$O/num-$1-$2" --wheel "$WHL" --scripts "$D/scripts" > "$O/num-$1-$2.out" 2> "$O/num-$1-$2.err"
    say "numerics $1 $2 rc=$? $(grep -a 'logits-run' "$O/num-$1-$2.out" | head -1 | cut -c1-120) $(grep -a 'provenance' "$O/num-$1-$2.out" | grep -o 'verified=[a-z]*')"
  done
  for arm in D U; do
    for n in 6 10; do
      # shellcheck disable=SC2046
      env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 900 "$VP/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 32 \
        --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$D/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
        | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
      say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt")"
    done
  done
  for pair in "P1 D U" "P1 D D2" "P2 D U"; do
    set -- $pair
    say "compare $1 $2 vs $3"
    python3 "$D/gdn_numcmp.py" "$O/num-$1-$2" "$O/num-$1-$3" 2>&1 | tee -a "$L"
  done
else
  for i in 1 2 3 4 5; do
    for pn in P1 P2; do
      if [ $((i % 2)) = 1 ]; then order="D U"; else order="U D"; fi
      for arm in $order; do
        # shellcheck disable=SC2046
        env $(env_of $arm) timeout 600 "$VP/bin/python" "$D/scripts/bench_decode.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
        say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
      done
    done
  done
fi
say "end $(date -u +%FT%TZ)"
