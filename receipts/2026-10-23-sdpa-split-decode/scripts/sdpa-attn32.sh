#!/bin/bash
# Third bf16 arm for the 7600-token teacher-forced run on one Mac: the model in bf16 with the single-query decode attention computed in float32 from stock ops (tfref-attn32.py: scores, softmax and P@V in float32,
# only the output cast to bf16; no kernel change, SPLIT=0), compared by tfcmp.py with the float32 reference, the old route (base) and the split route captures that already exist.
# No file outside OUT changes. usage: sdpa-attn32.sh OUT MODELDIR BASEDIR SPLITDIR REFDIR
set -uo pipefail
O=${1:?out dir}; M=${2:?model dir}; BASE=${3:?base dir}; SPLIT=${4:?split dir}; REF=${5:?ref dir}
D=$HOME/work
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "A32 abort $O exists"; exit 1; }
for x in "$BASE" "$SPLIT" "$REF"; do [ -f "$x/seg0.npz" ] || { echo "A32 abort capture missing in $x"; exit 1; }; done
mkdir -p "$O"
say() { echo "A32 $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-attn32 $(sha256sum "$D/tfref-attn32.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) text $(sha256sum "$D/tfref-long.txt" | cut -c1-16)"
say "captures base $(sha256sum "$BASE/seg0.npz" | cut -c1-16) split $(sha256sum "$SPLIT/seg0.npz" | cut -c1-16) ref $(sha256sum "$REF/seg0.npz" | cut -c1-16)"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
  PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" MLX_OMARCHY_SDPA_DECODE_SPLIT=0 \
  timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-attn32.py" --model "$M" --text "$D/tfref-long.txt" --device gpu --dtype native --nseg 1 --maxtok 7600 --stride 0 --out "$O/attn32" > "$O/attn32.out" 2> "$O/attn32.err"
say "arm attn32 rc=$? $(grep -a 'TFREF done' "$O/attn32.out" | cut -c1-90) $(grep -a 'ATTN32' "$O/attn32.out") files $(find "$O/attn32" -name 'seg*.npz' 2> /dev/null | wc -l)"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$REF" base="$BASE" split="$SPLIT" attn32="$O/attn32" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
