#!/bin/bash
# Float32 reference arm for the 7600-token teacher-forced run on one Mac, then the comparison of two EXISTING bf16 captures (base: MLX_OMARCHY_SDPA_DECODE_SPLIT=0, split: =1; both made by sdpa-num.sh long
# on the same tree) against it. The ref arm is the float32 model on the split-K branch tree (tree-final) with SPLIT=0: attention in float32, which no bf16 route serves.
# tfcmp.py reports for base and split: top-1 agreement with the reference, mean negative log-likelihood against the reference one, mean and maximum difference of the next-token log-probability, every top-1 flip with its reference top-2 gap.
# No file outside OUT changes. usage: sdpa-reflong.sh OUT MODELDIR BASEDIR SPLITDIR
set -uo pipefail
O=${1:?out dir}
M=${2:?model dir}
BASE=${3:?base capture dir}
SPLIT=${4:?split capture dir}
D=$HOME/work
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "SREF abort $O exists"; exit 1; }
[ -f "$BASE/seg0.npz" ] && [ -f "$SPLIT/seg0.npz" ] || { echo "SREF abort captures missing"; exit 1; }
mkdir -p "$O"
say() { echo "SREF $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-tail $(sha256sum "$D/tfref-tail.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) text $(sha256sum "$D/tfref-long.txt" | cut -c1-16)"
say "base capture $(sha256sum "$BASE/seg0.npz" | cut -c1-16) split capture $(sha256sum "$SPLIT/seg0.npz" | cut -c1-16)"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
  PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" MLX_OMARCHY_SDPA_DECODE_SPLIT=0 \
  timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-tail.py" --model "$M" --text "$D/tfref-long.txt" --device gpu --dtype float32 --nseg 1 --maxtok 7600 --stride 0 --out "$O/ref" > "$O/ref.out" 2> "$O/ref.err"
say "arm ref rc=$? $(grep -a 'TFREF done' "$O/ref.out" | cut -c1-90) files $(find "$O/ref" -name 'seg*.npz' 2> /dev/null | wc -l)"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/ref" base="$BASE" split="$SPLIT" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
