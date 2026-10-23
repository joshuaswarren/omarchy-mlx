#!/bin/bash
# Second float32 reference for the 7600-token teacher-forced run on one Mac: the float32 model (tfref-attn32v.py, --dtype float32) with the single-query decode attention in the REVERSED key order (variant rev), SPLIT=0.
# tfcmp.py then compares the existing bf16 captures (base, split, rev) with this reference, to show whether the distance of an arm from a float32 reference depends on whether the arm shares the reference's summation order.
# No file outside OUT changes. usage: sdpa-ref32v.sh OUT MODELDIR BASEDIR SPLITDIR REVDIR
set -uo pipefail
O=${1:?out dir}
M=${2:?model dir}
BASE=${3:?base dir}
SPLIT=${4:?split dir}
REV=${5:?rev dir}
D=$HOME/work
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "R32V abort $O exists"; exit 1; }
for x in "$BASE" "$SPLIT" "$REV"; do [ -f "$x/seg0.npz" ] || { echo "R32V abort capture missing in $x"; exit 1; }; done
mkdir -p "$O"
say() { echo "R32V $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-attn32v $(sha256sum "$D/tfref-attn32v.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) text $(sha256sum "$D/tfref-long.txt" | cut -c1-16)"
say "captures base $(sha256sum "$BASE/seg0.npz" | cut -c1-16) split $(sha256sum "$SPLIT/seg0.npz" | cut -c1-16) rev $(sha256sum "$REV/seg0.npz" | cut -c1-16)"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
  PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" MLX_OMARCHY_SDPA_DECODE_SPLIT=0 ATTN32_VARIANT=rev \
  timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-attn32v.py" --model "$M" --text "$D/tfref-long.txt" --device gpu --dtype float32 --nseg 1 --maxtok 7600 --stride 0 --out "$O/ref32rev" > "$O/ref32rev.out" 2> "$O/ref32rev.err"
say "arm ref32rev rc=$? $(grep -a 'ATTN32' "$O/ref32rev.out" | tr '\n' ' ') $(grep -a 'TFREF done' "$O/ref32rev.out" | cut -c1-90) files $(find "$O/ref32rev" -name 'seg*.npz' 2> /dev/null | wc -l)"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/ref32rev" base="$BASE" split="$SPLIT" rev="$REV" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
