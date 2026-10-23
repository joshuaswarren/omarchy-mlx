#!/bin/bash
# Numerics capture for the chunked vocabulary-tail kernels on one Mac, the ledger stack (venv-main wheel, driver vulkan-main): S=1 teacher-forced captures (tfref.py, 4 segments of 160 tokens)
# of the model in MODELDIR on the GPU with five arms: base (flags unset), argc (MLX_OMARCHY_ARGREDUCE_CHUNKED=1), lsec (MLX_OMARCHY_LSE_CHUNKED=1), both, and base again (determinism).
# tfref-tail.py also records the ArgMax and ArgMin index of the vocabulary row at every step, so the argc arm runs the chunked argreduce route on the model row.
# tfcmp.py compares each arm with the base capture: top-1 agreement, mean and maximum difference of the next-token log-probability, flips. tailchk.py compares the ArgMax and ArgMin indices with the base capture.
# The log-probability differs by the log-sum-exp association only (lsec arm). No file outside the output directory changes.
# usage: tail-num.sh OUTDIR MODELDIR
set -uo pipefail
D=$HOME/work
O=${1:?out dir}
M=${2:?model dir}
B0=$HOME/.local/share/coreglass
[ -e "$O" ] && { echo "TNUM abort $O exists"; exit 1; }
mkdir -p "$O"
say() { echo "TNUM $*" | tee -a "$O/run.log"; }
say "start $(date -u +%FT%TZ) boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg)"
say "tfref-tail $(sha256sum "$D/tfref-tail.py" | cut -c1-16) tfcmp $(sha256sum "$D/tfcmp.py" | cut -c1-16) tailchk $(sha256sum "$D/tailchk.py" | cut -c1-16) text $(sha256sum "$D/tfref-text.txt" | cut -c1-16) model index $(sha256sum "$M/model.safetensors.index.json" 2> /dev/null | cut -c1-16) libmlx $(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
run() { # name env
  local name=$1 extra=$2
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 HF_HUB_OFFLINE=1 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" PYTHONPATH="$B0/mlx-lm-series-main/lib/python3.14/site-packages" $extra \
    timeout 900 "$B0/venv-main/bin/python" "$D/tfref-tail.py" --model "$M" --text "$D/tfref-text.txt" --device gpu --dtype native --nseg 4 --maxtok 160 --out "$O/$name" > "$O/$name.out" 2> "$O/$name.err"
  say "arm $name rc=$? $(grep -a 'TFREF done' "$O/$name.out" | cut -c1-90) files $(find "$O/$name" -name 'seg*.npz' 2> /dev/null | wc -l)"
}
run base "MLX_AB_UNUSED=1"
run argc "MLX_OMARCHY_ARGREDUCE_CHUNKED=1"
run lsec "MLX_OMARCHY_LSE_CHUNKED=1"
run both "MLX_OMARCHY_ARGREDUCE_CHUNKED=1 MLX_OMARCHY_LSE_CHUNKED=1"
run base2 "MLX_AB_UNUSED=1"
"$B0/venv-main/bin/python" "$D/tfcmp.py" "$O/base" base2="$O/base2" argc="$O/argc" lsec="$O/lsec" both="$O/both" 2>&1 | grep -v "flip ref" | tee -a "$O/run.log"
"$B0/venv-main/bin/python" "$D/tailchk.py" "$O/base" base2="$O/base2" argc="$O/argc" lsec="$O/lsec" both="$O/both" 2>&1 | tee -a "$O/run.log"
say "tailchk rc=${PIPESTATUS[0]}"
say "end $(date -u +%FT%TZ)"
