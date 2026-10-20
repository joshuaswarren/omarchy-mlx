#!/bin/bash
# Verify the G14C ratio-3 build (N, commit 7076d7a5) against main (O, c163e54a) on the 14-inch M2 Max (G14C) with Qwen3.8-27B-4bit.
# usage: verify-g14c.sh OUTDIR STAGE   STAGE: test | census | numP1 | numP2 | speedA (rounds 1-3) | speedB (rounds 4-5)
#  test    : the whole omarchy_gdn_gqa_bf16a_tests binary on the device, plus the new case alone
#  census  : N default dispatches GatedDeltaDecodeBF16Pf (480), N TILE=1 dispatches GatedDeltaDecodeBF16 (455)
#  numP*   : N default == O TILE=0 and N TILE=1 == O default, bit for bit (128 tokens, full log-probs of 8 steps)
#  speed*  : two distinct builds, default environment in both, 64 tokens, order alternating by round, verified build stamp on every row
set -uo pipefail
O=${1:?output dir}
STAGE=${2:?stage}
D=$HOME/scratch/w7q/dense27-m2
DIST=$HOME/.local/share/coreglass/build/dist-r3
VN=$D/venv-N
VO=$D/venv-O
WN=$(find "$DIST/N-7076d7a5" -name 'mlx_omarchy-*.whl' | head -1)
WO=$(find "$DIST/O-c163e54a" -name 'mlx_omarchy-*.whl' | head -1)
TB=$DIST/N-7076d7a5/omarchy_gdn_gqa_bf16a_tests
MODEL=mlx-community/Qwen3.8-27B-4bit
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
mkdir -p "$O"
[ ! -e "$O/stage-$STAGE.done" ] || { echo "ABORT: stage $STAGE already ran"; exit 1; }
{ [ -n "$WN" ] && [ -n "$WO" ] && [ -x "$TB" ]; } || { echo "ABORT: wheel or test binary missing"; exit 1; }
L=$O/run-$STAGE.log
say() { echo "G14R $STAGE $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "env GDN/MLX/HK/AGX present: $(env | grep -cE '^(GDN|MLX_|HK_|AGX_)')"
say "N wheel $(basename "$WN") sha256 $(sha256sum "$WN" | cut -c1-64) libmlx $(sha256sum "$VN"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
say "O wheel $(basename "$WO") sha256 $(sha256sum "$WO" | cut -c1-64) libmlx $(sha256sum "$VO"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
say "test binary sha256 $(sha256sum "$TB" | cut -c1-64) driver $(sha256sum /usr/lib/libvulkan_asahi.so | cut -c1-16)"
say "scripts bench_decode $(sha256sum "$D/scripts/bench_decode.py" | cut -c1-16) logits $(sha256sum "$D/gdn_logits.py" | cut -c1-16) numcmp $(sha256sum "$D/gdn_numcmp.py" | cut -c1-16)"
export HF_HUB_OFFLINE=1

stage_test() {
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 600 "$TB" --list-test-cases > "$O/test-list.out" 2>&1
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 timeout 600 "$TB" > "$O/test-all.out" 2>&1
  say "test all rc=$? $(grep -a 'test cases:' "$O/test-all.out" | head -1) $(grep -a 'assertions:' "$O/test-all.out" | head -1)"
}

stage_census() {
  local arm envv n
  for arm in D T; do
    case $arm in D) envv=(MLX_AB_UNUSED=1) ;; T) envv=(MLX_OMARCHY_GDN_DECODE_TILE=1) ;; esac
    for n in 6 10; do
      env "${envv[@]}" MLX_OMARCHY_TRACE_DISPATCH=1 timeout 600 "$VN/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$P1" --tokens $n --full-steps 8 \
        --out "$O/cen-$arm-$n" --wheel "$WN" --scripts "$D/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
        | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
      say "census N $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") kernels=$(wc -l < "$O/hist-$arm-$n.txt") 455=$(awk '$2==455 {print $1}' "$O/hist-$arm-$n.txt") 480=$(awk '$2==480 {print $1}' "$O/hist-$arm-$n.txt")"
    done
  done
}

numrun() { # arm(N|O) tag prompt-name env...
  local arm=$1 tag=$2 pn=$3
  shift 3
  local vp=$D/venv-$arm w=$WN p=$P1
  [ "$arm" = O ] && w=$WO
  [ "$pn" = P2 ] && p=$P2
  env "$@" timeout 900 "$vp/bin/python" "$D/gdn_logits.py" --model "$MODEL" --prompt "$p" --tokens 128 --full-steps 8 \
    --out "$O/num-$pn-$tag" --wheel "$w" --scripts "$D/scripts" > "$O/num-$pn-$tag.out" 2> "$O/num-$pn-$tag.err"
  say "numerics $pn $tag rc=$? $(grep -a 'logits-run' "$O/num-$pn-$tag.out" | head -1 | cut -c1-110) $(grep -a 'provenance' "$O/num-$pn-$tag.out" | grep -o 'verified=[a-z]*')"
}

stage_num() { # prompt-name
  local pn=$1
  numrun N Ndef "$pn" MLX_AB_UNUSED=1
  numrun O Ounti "$pn" MLX_OMARCHY_GDN_DECODE_TILE=0
  numrun N Ntile "$pn" MLX_OMARCHY_GDN_DECODE_TILE=1
  numrun O Odef "$pn" MLX_AB_UNUSED=1
  for pair in "Ndef Ounti" "Ntile Odef"; do
    set -- $pair
    say "compare $pn $1 vs $2"
    python3 "$D/gdn_numcmp.py" "$O/num-$pn-$1" "$O/num-$pn-$2" 2>&1 | tee -a "$L"
  done
}

stage_speed() { # first last
  local i pn p order arm vp w
  for i in $(seq "$1" "$2"); do
    for pn in P1 P2; do
      p=$P1
      [ $pn = P2 ] && p=$P2
      if [ $((i % 2)) = 1 ]; then order="O N"; else order="N O"; fi
      for arm in $order; do
        if [ $arm = O ]; then vp=$VO; w=$WO; else vp=$VN; w=$WN; fi
        env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 BENCH_SCRIPTS_DIR="$D/scripts" timeout 600 "$vp/bin/python" "$D/scripts/bench_decode.py" --model "$MODEL" \
          --prompt "$p" --tokens 64 --wheel "$w" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
        say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'generated_ids' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
      done
    done
  done
}

case $STAGE in
  test) stage_test ;;
  census) stage_census ;;
  numP1) stage_num P1 ;;
  numP2) stage_num P2 ;;
  speedA) stage_speed 1 3 ;;
  speedB) stage_speed 4 5 ;;
  *) echo "ABORT: bad stage $STAGE"; exit 1 ;;
esac
touch "$O/stage-$STAGE.done"
say "end $(date -u +%FT%TZ)"
