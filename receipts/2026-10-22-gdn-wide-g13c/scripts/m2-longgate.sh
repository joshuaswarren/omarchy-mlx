#!/bin/bash
# Long-context numerics capture for the decode PR on the 14-inch M2 Max, Qwen3.8-2B 4-bit: S=1 teacher-forced capture (tfref.py) of 2 text segments of 600 tokens
# (GPL-3 text at token 0 and token 1500, so the cache and the GDN state span up to 599 positions, the range of the ledger timing step) on the GPU with three arms:
# tile1 (venv-main wheel a04423e, MLX_OMARCHY_GDN_DECODE_TILE=1, the old default), new (wheel of main aaef3b6c in tree-main2, default environment: the wide kernel)
# and plain (the same wheel, MLX_OMARCHY_GDN_PF_WIDE=0). The npz files stay in the output directory; the float32 CPU reference and the comparison run on another machine.
# No file outside the output directory changes.
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
B0=$HOME/.local/share/coreglass
TREE=$D/tree-main2
M=$B0/models/qwen3_5-4bit
[ -e "$O" ] && { echo "LGATE abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "LGATE $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "tfref $(sha256sum "$D/tfref.py" | cut -c1-16) text $(sha256sum "$D/tfref-text.txt" | cut -c1-16) model $(sha256sum "$M/model.safetensors" | cut -c1-16) new libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) old libmlx $(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
run() { # name tree env
  local name=$1 tree=$2 extra=$3 pp
  pp="$B0/mlx-lm-series-main/lib/python3.14/site-packages"
  [ -n "$tree" ] && pp="$tree:$pp"
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$pp" $extra \
    timeout 900 "$B0/venv-main/bin/python" "$D/tfref.py" --model "$M" --text "$D/tfref-text.txt" --device gpu --dtype native --nseg 2 --maxtok 600 --stride 1500 --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-90) files $(find "$O/$name" -name 'seg*.npz' 2> /dev/null | wc -l)"
}
run tile1 "" "MLX_OMARCHY_GDN_DECODE_TILE=1"
run new "$TREE" "MLX_AB_UNUSED=1"
run plain "$TREE" "MLX_OMARCHY_GDN_PF_WIDE=0"
run tile1b "" "MLX_OMARCHY_GDN_DECODE_TILE=1"
say "end $(date -u +%FT%TZ)"
