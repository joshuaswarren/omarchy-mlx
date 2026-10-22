#!/bin/bash
# Long-prompt numerics capture for the decode PR on the 14-inch M2 Max, Qwen3.8-2B 4-bit: tfctx.py runs the coreglass step's repeated-sentence prompt (nominal 2048 and 4096 tokens)
# through one prefill pass, then feeds 128 tokens of a fixed continuation one per forward (teacher forcing), so the GDN decode kernel runs on a cache that holds the whole prompt.
# GPU arms: tile1 (venv-main wheel a04423e, TILE=1, the old default), new (wheel of main aaef3b6c in tree-main2, default environment: the wide kernel), plain (same wheel, PF_WIDE=0).
# The prefill is the same code in every arm. The npz files stay in the output directory; the float32 CPU references and the comparison run on another machine.
# No file outside the output directory changes.
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
B0=$HOME/.local/share/coreglass
TREE=$D/tree-main2
M=$B0/models/qwen3_5-4bit
[ -e "$O" ] && { echo "CGATE abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "CGATE $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "tfctx $(sha256sum "$D/tfctx.py" | cut -c1-16) text $(sha256sum "$D/tfref-text.txt" | cut -c1-16) model $(sha256sum "$M/model.safetensors" | cut -c1-16) new libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) old libmlx $(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
run() { # name tree env prompt
  local name=$1 tree=$2 extra=$3 prompt=$4 pp
  pp="$B0/mlx-lm-series-main/lib/python3.14/site-packages"
  [ -n "$tree" ] && pp="$tree:$pp"
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$pp" $extra \
    timeout 600 "$B0/venv-main/bin/python" "$D/tfctx.py" --model "$M" --text "$D/tfref-text.txt" --device gpu --dtype native --prompt-tokens "$prompt" --steps 128 --cont-offset 0 --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFCTX' "$O/$name.out" | cut -c1-100) files $(find "$O/$name" -name 'seg0.npz' 2> /dev/null | wc -l)"
}
for p in 2048 4096; do
  run "tile1-$p" "" "MLX_OMARCHY_GDN_DECODE_TILE=1" "$p"
  run "new-$p" "$TREE" "MLX_AB_UNUSED=1" "$p"
  run "plain-$p" "$TREE" "MLX_OMARCHY_GDN_PF_WIDE=0" "$p"
  run "tile1b-$p" "" "MLX_OMARCHY_GDN_DECODE_TILE=1" "$p"
done
say "end $(date -u +%FT%TZ)"
