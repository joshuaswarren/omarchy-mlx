#!/bin/bash
# Numerics capture for a wide-kernel default on G13C, 16-inch M1 Max, Qwen3.8-2B 4-bit: S=1 teacher-forced captures (tfref.py) on the GPU with three arms:
# tile1 (venv-main wheel a04423e, MLX_OMARCHY_GDN_DECODE_TILE=1, the current default on this part), wide (wheel of main aaef3b6c in tree-main2, TILE=0 and PF_WIDE=1)
# and plain (the same wheel, TILE=0, PF_WIDE=0). Two shapes per arm: short = 4 segments of 160 tokens (the gate shape of PR 252) and long = 2 segments of 600 tokens
# (offsets 0 and 1500 of the GPL-3 text). The npz files stay in the output directory; the float32 CPU references and the comparison run on another machine.
# No file outside the output directory changes.
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
B0=$HOME/.local/share/coreglass
TREE=$D/tree-main2
M=$B0/models/qwen3_5-4bit
[ -e "$O" ] && { echo "G16 abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "G16 $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "tfref $(sha256sum "$D/tfref.py" | cut -c1-16) text $(sha256sum "$D/tfref-text.txt" | cut -c1-16) model $(sha256sum "$M/model.safetensors" | cut -c1-16) new libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) old libmlx $(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
run() { # name tree env nseg maxtok
  local name=$1 tree=$2 extra=$3 nseg=$4 maxtok=$5 pp
  pp="$B0/mlx-lm-series-main/lib/python3.14/site-packages"
  [ -n "$tree" ] && pp="$tree:$pp"
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$pp" $extra \
    timeout 900 "$B0/venv-main/bin/python" "$D/tfref.py" --model "$M" --text "$D/tfref-text.txt" --device gpu --dtype native --nseg "$nseg" --maxtok "$maxtok" --stride 1500 --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-90) files $(find "$O/$name" -name 'seg*.npz' 2> /dev/null | wc -l)"
}
for shape in short long; do
  if [ "$shape" = short ]; then n=4; t=160; else n=2; t=600; fi
  run "tile1-$shape" "" "MLX_OMARCHY_GDN_DECODE_TILE=1" "$n" "$t"
  run "wide-$shape" "$TREE" "MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_WIDE=1" "$n" "$t"
  run "plain-$shape" "$TREE" "MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_WIDE=0" "$n" "$t"
done
say "end $(date -u +%FT%TZ)"
