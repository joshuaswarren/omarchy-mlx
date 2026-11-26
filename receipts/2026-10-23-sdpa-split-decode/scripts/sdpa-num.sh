#!/bin/bash
# Model-level numerics capture for the split-K decode attention on one Mac: S=1 teacher-forced captures (tfref-tail.py: one forward per token, so the decode attention runs at every step)
# of the model in MODELDIR on the GPU, on the split-K branch tree (tree-final, ledger mlx-lm series main, driver vulkan-main).
# MODE short: 4 text segments of 160 tokens, arms base (MLX_OMARCHY_SDPA_DECODE_SPLIT=0), split (=1, so the route runs at every cache length), base2 (=0 again, the determinism and noise floor).
# MODE long: one segment of 7600 tokens of tfref-long.txt (cache lengths up to 7600, past the 7168-key point where the default route of the fused kernel hands over to the composition), arms base and split.
# tfcmp.py compares each arm with base: top-1 agreement, mean and maximum difference of the next-token log-probability, every top-1 flip with its top-2 margin. No file outside OUT changes.
# usage: sdpa-num.sh OUT MODELDIR short|long
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
M=${2:?model dir}
MODE=${3:?short|long}
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "SNUM abort $O exists"; exit 1; }
case "$MODE" in short | long) ;; *) echo "SNUM abort mode $MODE"; exit 2 ;; esac
mkdir -p "$O"
say() { echo "SNUM $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) mode $MODE boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-tail $(sha256sum "$D/tfref-tail.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) driver $(readlink "$B0/vulkan-main")"
if [ "$MODE" = short ]; then TEXT=$D/tfref-text.txt; SEG="--nseg 4 --maxtok 160"; else TEXT=$D/tfref-long.txt; SEG="--nseg 1 --maxtok 7600 --stride 0"; fi
say "text $(sha256sum "$TEXT" | cut -c1-16) segments $SEG"
run() { # name env
  local name=$1 extra=$2
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
    PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" $extra \
    timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-tail.py" --model "$M" --text "$TEXT" --device gpu --dtype native $SEG --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-90) files $(find "$O/$name" -name 'seg*.npz' 2> /dev/null | wc -l)"
}
run base "MLX_OMARCHY_SDPA_DECODE_SPLIT=0"
run split "MLX_OMARCHY_SDPA_DECODE_SPLIT=1"
[ "$MODE" = short ] && run base2 "MLX_OMARCHY_SDPA_DECODE_SPLIT=0"
if [ "$MODE" = short ]; then
  "$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/base" base2="$O/base2" split="$O/split" 2>&1 | grep -v "flip ref" | tee -a "$O/run.log"
else
  "$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/base" split="$O/split" 2>&1 | grep -v "flip ref" | tee -a "$O/run.log"
fi
say "end $(date -u +%FT%TZ)"
