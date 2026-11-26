#!/bin/bash
# dense 27B GDN route comparison on the 16-inch M1 Max (G13C): Qwen3.8-27B-4bit (bf16 A_log, 16 key heads, 48 value heads).
# Arms: D = default (tiled kernel), U = MLX_OMARCHY_GDN_DECODE_TILE=0 (untiled prefetch), G = MLX_OMARCHY_GDN_GQA_BF16A=1 (tiled native GQA).
# usage: run27.sh census|timeP1|timeP2 OUTDIR   (OUTDIR must not exist; timing parts take 5 rounds, arm order rotated each round)
set -uo pipefail
PART=${1:?census|timeP1|timeP2}
O=${2:?output dir}
B0=$HOME/scratch/w7q/step1-49e6b3c1b
VP=$B0/venv
WHL=$B0/wheel/mlx_omarchy-0.32.4.dev202610181348+49e6b3c1-cp314-cp314-linux_aarch64.whl
D=$HOME/scratch/w7q/dense27
MODEL=mlx-community/Qwen3.8-27B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
[ ! -e "$O" ] || { echo "ABORT: $O exists"; exit 1; }
test -f "$WHL" && test -x "$VP/bin/python" || { echo "ABORT: wheel or venv missing"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "D27 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) part=$PART kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1) libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) driver $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$D/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$D/gdn_logits.py" | cut -c1-16)"
export HF_HUB_OFFLINE=1
env_of() { case $1 in D) echo "MLX_AB_UNUSED=1" ;; U) echo "MLX_OMARCHY_GDN_DECODE_TILE=0" ;; G) echo "MLX_OMARCHY_GDN_GQA_BF16A=1" ;; esac; }
if [ "$PART" = census ]; then
  for arm in D U G; do
    for n in 6 10; do
      # shellcheck disable=SC2046
      env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 600 "$VP/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 8 \
        --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$D/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
        | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
      say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") $(grep -a 'provenance' "$O/cen-$arm-$n.out" | grep -o 'verified=[a-z]*')"
    done
  done
else
  case $PART in timeP1) pn=P1; PROMPT=$P1 ;; timeP2) pn=P2; PROMPT=$P2 ;; *) echo "bad part"; exit 2 ;; esac
  orders=("D U G" "U G D" "G D U" "D G U" "U D G")
  for i in 1 2 3 4 5; do
    for arm in ${orders[$((i - 1))]}; do
      # shellcheck disable=SC2046
      env $(env_of $arm) timeout 600 "$VP/bin/python" "$D/scripts/bench_decode.py" --model "$MODEL" --prompt "$PROMPT" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
    done
  done
fi
say "end $(date -u +%FT%TZ)"
