#!/bin/bash
# Model-level numerics capture for the split-K decode attention on one Mac: S=1 teacher-forced captures (tfref-tail.py: one forward per token, so the decode attention runs at every step)
# of the model in MODELDIR on the GPU, on the split-K branch tree (tree-final, ledger mlx-lm series main, driver vulkan-main).
# MODE short: one text segment of 7600 tokens (tfref-long.txt, the text three times; cache lengths up to 7600, past the 7168-key point where the default route of the fused kernel hands over to the composition), arms ref (the float32 model, MLX_OMARCHY_SDPA_DECODE_SPLIT=0: attention in float32, which no bf16 route serves), base (bf16, SPLIT=0), split (bf16, =1, so the route runs at every cache length).
# tfcmp.py compares base and split with the ref arm: top-1 agreement, mean negative log-likelihood against the reference one, mean and maximum difference of the next-token log-probability, every top-1 flip with its reference top-2 gap. No file outside OUT changes.
# usage: sdpa-numref-long.sh OUT MODELDIR
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
M=${2:?model dir}
MODE=short
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "SNUM abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "SNUM $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) mode $MODE boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-tail $(sha256sum "$D/tfref-tail.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) driver $(readlink "$B0/vulkan-main")"
TEXT=$D/tfref-long.txt; SEG="--nseg 1 --maxtok 7600 --stride 0"
say "text $(sha256sum "$TEXT" | cut -c1-16) segments $SEG"
run() { # name env [dtype args]
  local name=$1 extra=$2 dt=${3:---dtype native}
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
    PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" $extra \
    timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-tail.py" --model "$M" --text "$TEXT" --device gpu $dt $SEG --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-90) files $(find "$O/$name" -name 'seg*.npz' 2> /dev/null | wc -l)"
}
run ref "MLX_OMARCHY_SDPA_DECODE_SPLIT=0" "--dtype float32"
run base "MLX_OMARCHY_SDPA_DECODE_SPLIT=0"
run split "MLX_OMARCHY_SDPA_DECODE_SPLIT=1"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/ref" base="$O/base" split="$O/split" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
