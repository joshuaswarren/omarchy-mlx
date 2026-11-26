#!/bin/bash
# Dispatch census of the ledger step path (mlx-lm stream_generate, greedy, no sampler) for the vocabulary-tail switches, one Mac, the ledger stack (venv-main wheel, driver vulkan-main).
# For each arm (base, argc, lsec, both) the step runs 8 and 12 tokens with the dispatch trace on; every DISPATCH line is counted per kernel id (load and warm-up cancel in the difference);
# the difference of the two runs divided by 4 is dispatches per decode token. Shows whether a switch changes the launches of the greedy decode. Output: counts and the per-kernel histogram difference against base.
# usage: census-tail.sh OUTDIR MODELDIR      No file outside OUTDIR changes (the line's tee sits next to it).
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
M=${2:?model dir}
B0=$HOME/.local/share/coreglass
[ -e "$O" ] && { echo "CENS abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "CENS $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) census-tail $(sha256sum "$D/census-tail.py" | cut -c1-16) model index $(sha256sum "$M/model.safetensors.index.json" | cut -c1-16) libmlx $(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
count() { # name env n
  local name=$1 extra=$2 n=$3
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$B0/mlx-lm-series-main/lib/python3.14/site-packages" MLX_OMARCHY_TRACE_DISPATCH=1 $extra \
    timeout 600 "$B0/venv-main/bin/python" "$D/census-tail.py" "$M" "$n" 2> "$O/$name-$n.err" > "$O/$name-$n.out"
}
hist() { # name n -> hist file of the traced DISPATCH lines
  sed -n 's/.*DISPATCH kernel=\([0-9]*\) .*/\1/p' "$O/$1-$2.err" | sort -n | uniq -c | sort -k2,2n > "$O/$1-$2.hist"
}
for arm in "base:MLX_AB_UNUSED=1" "argc:MLX_OMARCHY_ARGREDUCE_CHUNKED=1" "lsec:MLX_OMARCHY_LSE_CHUNKED=1" "both:MLX_OMARCHY_ARGREDUCE_CHUNKED=1 MLX_OMARCHY_LSE_CHUNKED=1"; do
  name=${arm%%:*}
  extra=${arm#*:}
  for n in 8 12; do
    count "$name" "$extra" "$n"
    hist "$name" "$n"
  done
  a=$(awk '{s+=$1} END {print s+0}' "$O/$name-8.hist")
  b=$(awk '{s+=$1} END {print s+0}' "$O/$name-12.hist")
  say "arm $name dispatches 8 tokens $a, 12 tokens $b, per decode token $(awk -v a="$a" -v b="$b" 'BEGIN {printf "%.2f", (b - a) / 4}')"
  if [ "$name" != base ]; then
    awk 'NR==FNR {x[$2]=$1; next} {y[$2]=$1} END {for (k in x) d[k]=1; for (k in y) d[k]=1; for (k in d) if ((y[k]-x[k]) != 0) printf "CENS hist %s kernel %s per token %.2f\n", "'"$name"'", k, (y[k]-x[k])/4}' \
      <(awk 'NR==FNR {a[$2]=$1; next} {b[$2]=$1} END {for (k in a) d[k]=1; for (k in b) d[k]=1; for (k in d) print b[k]-a[k], k}' "$O/base-8.hist" "$O/base-12.hist") \
      <(awk 'NR==FNR {a[$2]=$1; next} {b[$2]=$1} END {for (k in a) d[k]=1; for (k in b) d[k]=1; for (k in d) print b[k]-a[k], k}' "$O/$name-8.hist" "$O/$name-12.hist") | sort -k5,5n | tee -a "$O/run.log"
  fi
done
say "end $(date -u +%FT%TZ)"
