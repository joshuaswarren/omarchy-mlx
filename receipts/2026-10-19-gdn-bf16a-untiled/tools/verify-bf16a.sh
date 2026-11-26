#!/bin/bash
# Verify the PR build (wheel a2b0864) on the 16-inch M1 Max with Qwen3.6-35B-A3B-4bit:
#  1 numerics: the new default equals the old TILE=0 arm bit for bit (128 tokens, full log-probs of 32 steps), and TILE=1 equals the old default arm;
#  2 census: default dispatches GatedDeltaDecodeBF16Pf (480), TILE=1 dispatches GatedDeltaDecodeBF16 (455);
#  3 speed: 5 rounds, 2 prompts, default against TILE=1, order alternating, 64 tokens.
set -uo pipefail
D=$HOME/scratch/w7b-gqmm/verify35
B=$HOME/.local/share/coreglass/build
OLD=$HOME/scratch/w7b-gqmm/tile35/out-num-a
SRCVENV=$HOME/scratch/w7q/step1-49e6b3c1b/venv
VP=$D/venv
WHL=$(find "$B/dist-bf16a" -name "mlx_omarchy-*+a2b0864-*.whl" | head -1)
O=$D/out-1
MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
SC=$HOME/scratch/w7b-gqmm/tile35
P1="Explain in plain words how a bicycle works."
P2="Write a short story about a lighthouse keeper."
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
mkdir -p "$O"
if [ ! -e "$VP" ]; then cp -r "$SRCVENV" "$VP" && "$VP/bin/python" -m pip install -q --no-deps --force-reinstall "$WHL" || { echo "venv setup failed"; exit 1; }; fi
L=$O/run.log
say() { echo "VER $*" | tee -a "$L"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) load1=$(cut -d' ' -f1 /proc/loadavg) wheel $(basename "$WHL") sha256 $(sha256sum "$WHL" | cut -c1-64) libmlx $(sha256sum "$VP"/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
export HF_HUB_OFFLINE=1
run() { # arm prompt-name envassignment...
  local arm=$1 pn=$2; shift 2
  local p; if [ "$pn" = P1 ]; then p=$P1; else p=$P2; fi
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 "$@" timeout 1200 "$VP/bin/python" "$SC/gdn_logits.py" --model "$MODEL" --prompt "$p" \
    --tokens 128 --full-steps 32 --out "$O/num-$pn-$arm" --wheel "$WHL" --scripts "$SC/scripts" > "$O/num-$pn-$arm.out" 2> "$O/num-$pn-$arm.err"
  say "numerics $pn $arm rc=$? $(grep -a 'logits-run' "$O/num-$pn-$arm.out" | head -1 | cut -c1-110) $(grep -a 'provenance' "$O/num-$pn-$arm.out" | grep -o 'verified=[a-z]*')"
}
run D P1 MLX_AB_UNUSED=1
run T P1 MLX_OMARCHY_GDN_DECODE_TILE=1
run D P2 MLX_AB_UNUSED=1
run T P2 MLX_OMARCHY_GDN_DECODE_TILE=1
for pn in P1 P2; do
  say "compare $pn new default vs old TILE=0 arm"; python3 "$SC/gdn_numcmp.py" "$O/num-$pn-D" "$OLD/num-$pn-U" 2>&1 | tee -a "$L"
  say "compare $pn new TILE=1 vs old default arm"; python3 "$SC/gdn_numcmp.py" "$O/num-$pn-T" "$OLD/num-$pn-D" 2>&1 | tee -a "$L"
done
for arm in D T; do
  if [ $arm = D ]; then e=MLX_AB_UNUSED=1; else e=MLX_OMARCHY_GDN_DECODE_TILE=1; fi
  for n in 6 10; do
    env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 "$e" MLX_OMARCHY_TRACE_DISPATCH=1 timeout 1200 "$VP/bin/python" "$SC/gdn_logits.py" --model "$MODEL" --prompt "$P1" \
      --tokens $n --full-steps 32 --out "$O/cen-$arm-$n" --wheel "$WHL" --scripts "$SC/scripts" 2>&1 >"$O/cen-$arm-$n.out" \
      | sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' | sort -n | uniq -c | sort -k2,2n > "$O/hist-$arm-$n.txt"
    say "census $arm tokens=$n dispatches=$(awk '{s+=$1} END {print s+0}' "$O/hist-$arm-$n.txt") gdn455=$(awk '$2==455{print $1}' "$O/hist-$arm-$n.txt") gdn480=$(awk '$2==480{print $1}' "$O/hist-$arm-$n.txt")"
  done
done
for i in 1 2 3 4 5; do
  for pn in P1 P2; do
    if [ $((i % 2)) = 1 ]; then order="D T"; else order="T D"; fi
    for arm in $order; do
      if [ $arm = D ]; then e=MLX_AB_UNUSED=1; else e=MLX_OMARCHY_GDN_DECODE_TILE=1; fi
      p=$P1; [ $pn = P2 ] && p=$P2
      env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 BENCH_SCRIPTS_DIR="$SC/scripts" "$e" timeout 600 "$VP/bin/python" "$SC/scripts/bench_decode.py" --model "$MODEL" \
        --prompt "$p" --tokens 64 --wheel "$WHL" > "$O/t-$pn-$arm-$i.out" 2> "$O/t-$pn-$arm-$i.err"
      say "time $pn round $i $arm rc=$? $(grep -a 'decode ' "$O/t-$pn-$arm-$i.out" | head -1 | cut -c1-60) $(grep -a 'provenance' "$O/t-$pn-$arm-$i.out" | grep -o 'verified=[a-z]*')"
    done
  done
done
say "end $(date -u +%FT%TZ)"
