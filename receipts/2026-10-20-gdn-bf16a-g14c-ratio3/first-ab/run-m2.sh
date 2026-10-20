#!/bin/bash
# Qwen3.8-27B-4bit GDN decode kernel choice on the 14-inch M2 Max (G14C): default (D, tiled) against MLX_OMARCHY_GDN_DECODE_TILE=0 (U, untiled prefetch).
# usage: run-m2.sh census|timeP1|timeP2|D0|U0|D1|U1|D2 OUTDIR   (OUTDIR must not exist). See PREREG-g14c-27b.md.
set -uo pipefail
PART=${1:?part}
O=${2:?output dir}
D=$HOME/scratch/w7q/dense27-m2
VP=$D/venv
WHL=$(find "$HOME/.local/share/coreglass/build/dist-r3/M-f3344207" -name 'mlx_omarchy-*.whl' | head -1)
N=$D/nll-tools
MODEL=mlx-community/Qwen3.8-27B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
[ ! -e "$O" ] || { echo "ABORT: $O exists"; exit 1; }
{ [ -f "$WHL" ] && [ -x "$VP/bin/python" ]; } || { echo "ABORT: wheel or venv missing"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "G14C $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) part=$PART kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(sort -u /sys/devices/system/cpu/cpufreq/policy*/scaling_governor | tr '\n' ' ') load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -c1-64) libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16) driver $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "mlx_lm gated_delta $(sha256sum "$VP"/lib/python3.14/site-packages/mlx_lm/models/gated_delta.py | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$D/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$D/gdn_logits.py" | cut -c1-16) nll_pair $(sha256sum "$N/nll_pair.py" | cut -c1-16) nll_cmp $(sha256sum "$N/nll_cmp.py" | cut -c1-16)"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
env_of() { case $1 in D) echo "MLX_AB_UNUSED=1" ;; U) echo "MLX_OMARCHY_GDN_DECODE_TILE=0" ;; esac; }
case $PART in
  census)
    for arm in D U; do
      for n in 6 10; do
        # shellcheck disable=SC2046
        env $(env_of $arm) MLX_OMARCHY_TRACE_DISPATCH=1 timeout 900 "$VP/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 8 \
          --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$D/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
          | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
        say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") 455=$(awk '$2==455 {print $1}' "$O/hist-$arm-$n.txt") 480=$(awk '$2==480 {print $1}' "$O/hist-$arm-$n.txt") $(grep -a 'provenance' "$O/cen-$arm-$n.out" | grep -o 'verified=[a-z]*')"
      done
    done ;;
  timeP1|timeP2)
    case $PART in timeP1) pn=P1; PROMPT=$P1 ;; timeP2) pn=P2; PROMPT=$P2 ;; esac
    for i in 1 2 3 4 5; do
      if [ $((i % 2)) = 1 ]; then order="D U"; else order="U D"; fi
      for arm in $order; do
        # shellcheck disable=SC2046
        env $(env_of $arm) timeout 900 "$VP/bin/python" "$D/scripts/bench_decode.py" --model "$MODEL" --prompt "$PROMPT" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
        say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
      done
    done ;;
  D0|U0|D1|U1|D2)
    case $PART in
      D0) arm=D; first=0; count=64 ;;
      U0) arm=U; first=0; count=64 ;;
      D1) arm=D; first=64; count=64 ;;
      U1) arm=U; first=64; count=64 ;;
      D2) arm=D; first=0; count=32 ;;
    esac
    # shellcheck disable=SC2046
    env $(env_of $arm) timeout 1700 "$VP/bin/python" "$N/nll_pair.py" "$MODEL" "$N/text-AB-e7c41800.txt" "$O/$PART.npz" --prefixes $count --first $first \
      --expect-libmlx 1a2af63453eeb246 > "$O/$PART.out" 2> "$O/$PART.err"
    say "part $PART rc=$? $(grep -a 'NLL mean' "$O/$PART.out" | cut -c1-200)" ;;
  *) echo "bad part"; exit 2 ;;
esac
say "end $(date -u +%FT%TZ)"
