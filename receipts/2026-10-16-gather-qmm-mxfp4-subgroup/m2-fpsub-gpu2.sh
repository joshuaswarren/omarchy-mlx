#!/bin/bash
# m2-fpsub-gpu2.sh (repeat with the bf16 cell run twice, per-call times printed): the mxfp4 gather_qmm subgroup kernel (branch agent/w6z-gather-qmm-mxfp4-sub, commit 2f8fc4034) on the 14-inch M2 Max at the gpt-oss-20b expert shape.
# Run 1 (no trace): gptoss_fp_sub_gpu.py, numerics against fp64 and per-call GPU time for the default route and the scalar route in one process.
# Run 2 (trace on): the same script, only to read the kernel id the default route dispatches in each cell (timings from this run are not used).
# Random weights: cost and numerics only. No model, no download.
set -uo pipefail
W=${WORK:?set WORK to the working directory}
S=${PYENV:?set PYENV to a directory with venv/bin/python}
WHL=$(find "$W/fpsub-wheel" -maxdepth 1 -name 'mlx_omarchy-*+2f8fc40*-cp314-cp314-linux_aarch64.whl' | head -1)
TREE=$W/fpsub-tree
O=$W/fpsub-gpu-m2-2
[ -e "$O" ] && { echo "ABORT: $O exists"; exit 1; }
mkdir -p "$O"
L=$O/run.log
say() { echo "FPSUB $*" | tee -a "$L"; }
[ -n "$WHL" ] || { say "ABORT no wheel"; exit 1; }
WSHA=$(cut -c1-64 "$WHL.sha256")
[ "$(sha256sum "$WHL" | cut -c1-64)" = "$WSHA" ] || { say "ABORT wheel sha mismatch"; exit 1; }
say "wheel $(basename "$WHL") size $(stat -c %s "$WHL") sha256 $WSHA"
if [ ! -e "$TREE" ]; then
  mkdir -p "$TREE" && "$S/venv/bin/python" -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$WHL" "$TREE" || { say "ABORT unpack"; exit 1; }
fi
say "kernel=$(uname -r) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) governor=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "tree libmlx sha256 $(sha256sum "$TREE/mlx/lib/libmlx.so" | cut -c1-64) script $(sha256sum "$W/gptoss_fp_sub_gpu.py" | cut -c1-16)"
say "tree holds GatherQmmNbSubFp: $(grep -c -a 'GatherQmmNbSubFp' "$TREE/mlx/lib/libmlx.so") line(s) with the name"
timeout 900 env PYTHONPATH="$TREE" "$S/venv/bin/python" "$W/gptoss_fp_sub_gpu.py" > "$O/run1.out" 2> "$O/run1.err"
rc=$?
say "run1 rc=$rc load1_after=$(cut -d' ' -f1 /proc/loadavg)"
while IFS= read -r ln; do say "run1: $ln"; done < "$O/run1.out"
grep -a -v "^\[rtmod\]" "$O/run1.err" | tail -n 6 | while IFS= read -r ln; do say "run1 err: $ln"; done
MLX_OMARCHY_TRACE_DISPATCH=1 timeout 900 env PYTHONPATH="$TREE" "$S/venv/bin/python" "$W/gptoss_fp_sub_gpu.py" > "$O/run2.out" 2> "$O/run2.err"
rc2=$?
say "run2 (trace) rc=$rc2"
awk '/CELLMARK/ {cell=$0; sub(/^CELLMARK /,"",cell); on=1; next} /CELLEND/ {on=0; next} on && /DISPATCH kernel=/ {split($0,a,"kernel="); split(a[2],b," "); n[cell" id "b[1]]++} END {for (k in n) print k, n[k]}' "$O/run2.err" | sort | while IFS= read -r ln; do say "kernels: $ln"; done
say "DONE out=$O rc=$rc rc2=$rc2"
