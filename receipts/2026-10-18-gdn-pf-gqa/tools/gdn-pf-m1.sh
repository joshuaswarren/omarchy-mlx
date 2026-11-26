#!/bin/bash
# gdn-pf-m1.sh MODEL TAG: STEP 2 on an M1-class laptop (16-inch M1 Max), the 9B model named by MODEL. Same design as gdn-pf.sh (M2) with a shorter numerics part.
# ONE wheel (candidate 6c863089, main 878816d6 plus the opt-in native-GQA untiled prefetch kernel), ONE boot, env only. The venv is a copy of the earlier run's venv with only the wheel replaced.
# Arms: A = default, B = MLX_OMARCHY_GDN_DECODE_TILE=0, C = MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_GQA=1.
# Part 1 (speed): bench_decode.py, 64 tokens, prompts P1 and P2, 5 rounds, order rotating ABC / BCA / CAB / ABC / BCA.
# Part 2 (numerics): gdn_logits.py, 128 greedy tokens of P1, per-step log-probabilities (full vector for the first 32 steps), arms A, B, C, C again (C2).
# Part 3 (census): the recorder with MLX_OMARCHY_TRACE_DISPATCH=1, arms A, B, C, 6 and 10 tokens of P1; per-token = difference over 4.
# Measure only. Writes only under ~/scratch/gdn/gdn-pf-TAG-1.
set -uo pipefail
MODEL=${1:?model}
TAG=${2:?tag}
Q=$HOME/scratch/gdn/gdn2
B0=$HOME/scratch/gdn/pfgqa-6c863089
VP=$B0/venv
SRC=$HOME/scratch/gdn/mlx-src
O=$HOME/scratch/gdn/gdn-pf-$TAG-1
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
WHL=$(find "$B0/wheel" -maxdepth 1 -name 'mlx_omarchy-*+6c863089-cp314-cp314-linux_aarch64.whl' | head -1)
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
[ -n "$WHL" ] || { echo "ABORT no wheel"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "GDNPFM1 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) model=$MODEL"
say "env variables GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1)"
say "libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) vulkan driver library $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$SRC/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$Q/gdn_logits.py" | cut -c1-16)"
say "patches tree $(cd "$SRC/patches" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-16) files $(find "$SRC/patches" -type f | wc -l)"
say "model snapshot $(ls "$HOME/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots") MemAvailable_kB $(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
say "venv $("$VP/bin/python" -m pip list 2>/dev/null | grep -i '^mlx' | tr -s ' ' | tr '\n' ';')"
export HF_HUB_OFFLINE=1
env_of() {
  case $1 in
    A|A2) echo "MLX_AB_UNUSED=1" ;;
    B) echo "MLX_OMARCHY_GDN_DECODE_TILE=0" ;;
    C|C2) echo "MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_GQA=1" ;;
  esac
}
prompt_of() { case $1 in P1) echo "$P1" ;; P2) echo "$P2" ;; esac; }
for i in 1 2 3 4 5; do
  for pn in P1 P2; do
    case $((i % 3)) in
      1) order="A B C" ;;
      2) order="B C A" ;;
      0) order="C A B" ;;
    esac
    for arm in $order; do
      # shellcheck disable=SC2046
      env $(env_of $arm) timeout 600 "$VP/bin/python" "$SRC/scripts/bench_decode.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
    done
  done
done
for arm in A B C C2; do
  # shellcheck disable=SC2046
  env $(env_of $arm) timeout 900 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens 128 --full-steps 32 --out "$O/num-P1-$arm" --wheel "$WHL" --scripts "$SRC/scripts" > "$O/num-P1-$arm.out" 2> "$O/num-P1-$arm.err"
  say "numerics P1 $arm rc=$? $(grep -a 'logits-run' "$O/num-P1-$arm.out" | head -1 | cut -c1-110) $(grep -a 'provenance' "$O/num-P1-$arm.out" | grep -o 'verified=[a-z]*')"
done
for arm in A B C; do
  for n in 6 10; do
    # shellcheck disable=SC2046
    env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 600 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 8 --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$SRC/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
      | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
    say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") $(grep -a 'provenance' "$O/cen-$arm-$n.out" | grep -o 'verified=[a-z]*')"
  done
done
say "end $(date -u +%FT%TZ)"
