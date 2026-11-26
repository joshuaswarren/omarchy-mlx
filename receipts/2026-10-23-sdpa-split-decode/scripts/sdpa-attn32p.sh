#!/bin/bash
# One reorder-only bf16 arm for the 7600-token teacher-forced run on one Mac: the model in bf16 with the single-query decode attention computed in float32 from stock ops (tfref-attn32p.py), variant
# perm (the keys in a fixed random permutation per cache length: a different float32 summation order), SPLIT=0, compared by tfcmp.py with the float32 reference and the
# existing base and split captures. No file outside OUT changes. usage: sdpa-attn32p.sh OUT MODELDIR BASEDIR SPLITDIR REFDIR VARIANT
set -uo pipefail
O=${1:?out dir}
M=${2:?model dir}
BASE=${3:?base dir}
SPLIT=${4:?split dir}
REF=${5:?ref dir}
V=${6:?perm}
case "$V" in perm) ;; *) echo "A32V abort variant $V"; exit 2 ;; esac
D=$HOME/work
B0=$HOME/.local/share/coreglass
TREE=$D/tree-final
[ -e "$O" ] && { echo "A32V abort $O exists"; exit 1; }
for x in "$BASE" "$SPLIT" "$REF"; do [ -f "$x/seg0.npz" ] || { echo "A32V abort capture missing in $x"; exit 1; }; done
mkdir -p "$O"
say() { echo "A32V $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) variant $V boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-16) tfref-attn32v $(sha256sum "$D/tfref-attn32p.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) text $(sha256sum "$D/tfref-long.txt" | cut -c1-16)"
say "captures base $(sha256sum "$BASE/seg0.npz" | cut -c1-16) split $(sha256sum "$SPLIT/seg0.npz" | cut -c1-16) ref $(sha256sum "$REF/seg0.npz" | cut -c1-16)"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" \
  PYTHONPATH="$TREE:$B0/mlx-lm-series-main/lib/python3.14/site-packages" MLX_OMARCHY_SDPA_DECODE_SPLIT=0 ATTN32_VARIANT="$V" \
  timeout 3300 "$B0/venv-main/bin/python" "$D/tfref-attn32p.py" --model "$M" --text "$D/tfref-long.txt" --device gpu --dtype native --nseg 1 --maxtok 7600 --stride 0 --out "$O/$V" > "$O/$V.out" 2> "$O/$V.err"
say "arm $V rc=$? $(grep -a 'ATTN32' "$O/$V.out" | tr '\n' ' ') $(grep -a 'TFREF done' "$O/$V.out" | cut -c1-90) files $(find "$O/$V" -name 'seg*.npz' 2> /dev/null | wc -l)"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$REF" base="$BASE" split="$SPLIT" "$V=$O/$V" 2>&1 | tee -a "$O/run.log"
say "end $(date -u +%FT%TZ)"
