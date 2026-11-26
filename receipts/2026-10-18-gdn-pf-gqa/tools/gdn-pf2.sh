#!/bin/bash
# gdn-pf2.sh: REBASED-WHEEL CHECK of STEP 2, a short run of gdn-pf.sh (one speed round of P1 for the digest, numerics and census, no speed claim), ticket on the 14-inch M2 Max, Qwen3.5-9B-4bit. ONE wheel (candidate 6c863089, main 878816d6 plus the opt-in native-GQA untiled prefetch kernel), ONE boot, env only.
# Arms: A = default (tiled composed kernel, native GQA), B = MLX_OMARCHY_GDN_DECODE_TILE=0 (untiled prefetch, q and k expanded by two copies),
#       C = MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_GQA=1 (untiled prefetch with native GQA, new kernel).
# Part 1 (speed): bench_decode.py, 64 tokens, 5 rounds per prompt (P1, P2), arm order rotating A B C / B C A / C A B / A B C / B C A.
# Part 2 (numerics): gdn_logits.py, 128 greedy tokens, per-step log-probabilities (full vector for the first 32), arms A, B, C, C again (C2), for prompts P1 and P2.
# Part 3 (census): the same recorder with MLX_OMARCHY_TRACE_DISPATCH=1, arms A, B, C, 6, 10 and 128 tokens of P1; per-token = difference of the 6 and 10 runs over 4.
# Measure only. Writes only under ~/scratch/gdn/gdn-pf2-m2-1. Wheel and venv: the outputs of the build ticket pfgqa2-9fde0d95.
set -uo pipefail
Q=$HOME/scratch/gdn/gdn2
B0=$HOME/scratch/gdn/pfgqa2-9fde0d95
VP=$B0/venv
SRC=/var/tmp/gdn-h8/mlx-src
O=$HOME/scratch/gdn/gdn-pf2-m2-1
MODEL=mlx-community/Qwen3.5-9B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
WHL=$(find "$B0/wheel" -maxdepth 1 -name 'mlx_omarchy-*+9fde0d95-cp314-cp314-linux_aarch64.whl' | head -1)
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
[ -n "$WHL" ] || { echo "ABORT no wheel"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "GDNPF2 $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env variables GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -d' ' -f1)"
say "libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$SRC/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$Q/gdn_logits.py" | cut -c1-16)"
say "patches tree $(cd "$SRC/patches" && find . -type f | sort | xargs sha256sum | sha256sum | cut -c1-16) files $(find "$SRC/patches" -type f | wc -l)"
say "model snapshot $(ls "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots")"
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
for i in 1; do
  for pn in P1; do
    case $((i % 3)) in
      1) order="A B C" ;;
      2) order="B C A" ;;
      0) order="C A B" ;;
    esac
    for arm in $order; do
      # shellcheck disable=SC2046
      env $(env_of $arm) timeout 400 "$VP/bin/python" "$SRC/scripts/bench_decode.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
    done
  done
done
for pn in P1; do
  for arm in A B C C2; do
    # shellcheck disable=SC2046
    env $(env_of $arm) timeout 600 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$(prompt_of $pn)" --tokens 128 --full-steps 32 --out "$O/num-$pn-$arm" --wheel "$WHL" --scripts "$SRC/scripts" > "$O/num-$pn-$arm.out" 2> "$O/num-$pn-$arm.err"
    say "numerics $pn $arm rc=$? $(grep -a 'logits-run' "$O/num-$pn-$arm.out" | head -1 | cut -c1-120) $(grep -a 'provenance' "$O/num-$pn-$arm.out" | grep -o 'verified=[a-z]*')"
  done
done
for arm in A B C; do
  for n in 6 10; do
    # shellcheck disable=SC2046
    env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 600 "$VP/bin/python" "$Q/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 32 --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$SRC/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
      | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
    say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") $(grep -a 'logits-run' "$O/cen-$arm-$n.out" | head -1 | cut -c1-90) $(grep -a 'provenance' "$O/cen-$arm-$n.out" | grep -o 'verified=[a-z]*')"
  done
done
say "end $(date -u +%FT%TZ)"
