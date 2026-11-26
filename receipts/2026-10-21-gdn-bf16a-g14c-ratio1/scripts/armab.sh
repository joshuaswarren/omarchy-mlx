#!/bin/bash
# armab.sh PART OUTDIR MODEL "PROMPT-FILE-OR-NAMES" ARM_BASE ARM_X   -- two-arm A/B of one environment switch, one wheel, fresh process per sample.
#   PART num  : 128-token greedy log-prob comparison (full vector of 32 steps), D vs X, plus a D repeat; census of kernels (10 minus 6 tokens)
#   PART time : bench_decode 64 tokens, 5 rounds x 2 prompts, order alternating
# ARM_BASE and ARM_X are NAME=VALUE environment assignments ("MLX_AB_UNUSED=1" for the unset base).
# Layout under $AB (set per host): $AB/venv-path file names are passed through env: VP (venv), WHL (wheel), SC (scripts dir with
# scripts/bench_decode.py, gdn_logits.py, gdn_numcmp.py). Not a guard ticket; run under gpu-turn.
set -uo pipefail
PART=${1:?num|time}; O=${2:?out dir}; MODEL=${3:?model}; BASE=${4:?base env}; X=${5:?x env}
: "${VP:?}" "${WHL:?}" "${SC:?}"
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "AB $*" | tee -a "$L"; }
P1=${P1_OVERRIDE:-"Explain in plain words how a bicycle works."}
P2=${P2_OVERRIDE:-"Write a short story about a lighthouse keeper."}
say "start $(date -u +%FT%TZ) part=$PART boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "model $MODEL base [$BASE] x [$X]; env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -c1-64) libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) driver $(sha256sum /usr/lib/libvulkan_<vendor>.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$SC/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$SC/gdn_logits.py" | cut -c1-16)"
export HF_HUB_OFFLINE=1
envof() { case $1 in D|D2) echo "$BASE" ;; X) echo "$X" ;; esac; }
promptof() { case $1 in P1) echo "$P1" ;; P2) echo "$P2" ;; esac; }
if [ "$PART" = num ]; then
  for run in "P1 D" "P1 X" "P1 D2" "P2 D" "P2 X"; do
    set -- $run
    # shellcheck disable=SC2046
    env $(envof "$2") timeout 1200 "$VP/bin/python" "$SC/gdn_logits.py" --model "$MODEL" --prompt "$(promptof "$1")" --tokens 128 --full-steps 32 \
      --out "$O/num-$1-$2" --wheel "$WHL" --scripts "$SC/scripts" > "$O/num-$1-$2.out" 2> "$O/num-$1-$2.err"
    say "numerics $1 $2 rc=$? $(grep -a 'logits-run' "$O/num-$1-$2.out" | head -1 | cut -c1-110) $(grep -a 'provenance' "$O/num-$1-$2.out" | grep -o 'verified=[a-z]*')"
  done
  for arm in D X; do
    for n in 6 10; do
      # shellcheck disable=SC2046
      env $(envof $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 1200 "$VP/bin/python" "$SC/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 32 \
        --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$SC/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
        | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
      say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt")"
    done
  done
  for pair in "P1 D X" "P1 D D2" "P2 D X"; do
    set -- $pair
    say "compare $1 $2 vs $3"
    python3 "$SC/gdn_numcmp.py" "$O/num-$1-$2" "$O/num-$1-$3" 2>&1 | tee -a "$L"
  done
else
  for i in 1 2 3 4 5; do
    for pn in P1 P2; do
      if [ $((i % 2)) = 1 ]; then order="D X"; else order="X D"; fi
      for arm in $order; do
        # shellcheck disable=SC2046
        env $(envof $arm) timeout 600 "$VP/bin/python" "$SC/scripts/bench_decode.py" --model "$MODEL" --prompt "$(promptof $pn)" --tokens 64 --wheel "$WHL" \
          > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
        say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-50) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*') load1=$(cut -d' ' -f1 /proc/loadavg)"
      done
    done
  done
fi
say "end $(date -u +%FT%TZ)"
