#!/bin/bash
# Per-op error of the decode attention routes against float64 on one Mac (sdpa_bias.py: fused, split, float32 composition; signed error in bf16 steps), tree-final, GPU, untimed. Writes only OUT.
# usage: sdpa-bias.sh OUT TRIALS
set -uo pipefail
O=${1:?out dir}
T=${2:?trials}
D=$HOME/work
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "BIAS abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "BIASRUN $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) sdpa_bias $(sha256sum "$D/sdpa_bias.py" | cut -c1-16)"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$TREE" \
  timeout 1500 "$B0/venv-main/bin/python" "$D/sdpa_bias.py" "$T" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ) rc ${PIPESTATUS[0]}"
