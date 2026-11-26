#!/bin/bash
# sdpa_bench.sh OUT REPEATS MODE : decode attention microbench (sdpa_bench.py), one Mac, the ledger stack (venv-main wheel, driver vulkan-main), three processes on the value-phase branch tree: the fused kernel (SPLIT=0), the split route (SPLIT=1, PV=0) and the split route with the value-phase variant (SPLIT=1, PV=1).
# MODE timed: aborts above load 1.3 at the start; MODE smoke: no load gate (REPEATS 1). Writes only OUT.
set -uo pipefail
O=${1:?out}; R=${2:?repeats}; MODE=${3:?timed|smoke}
B0=$HOME/.local/share/coreglass
PY=$WORK/sdpa_bench_pv.py
[ -e "$O" ] && { echo "ABORT $O exists"; exit 1; }
case "$MODE" in timed | smoke) ;; *) echo "ABORT mode $MODE"; exit 2 ;; esac
mkdir -p "$O"
L=$O/run.log
say() { echo "SDPA-RUN $*" | tee -a "$L"; }
if [ "$MODE" = timed ]; then
  awk '{exit !($1<=1.3)}' /proc/loadavg || { say "ABORT load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 3; }
fi
say "start $(date -u +%FT%TZ) mode $MODE boot=$(cut -c1-8 /proc/sys/kernel/random/boot_id) kernel=$(uname -r) gov=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor) load1=$(cut -d' ' -f1 /proc/loadavg) settle=$(cat /sys/module/ane_t6021/parameters/call_settle_us 2> /dev/null || echo na)"
say "tree libmlx $(sha256sum "$WORK/tree-pv/mlx/lib/libmlx.so" | cut -c1-16)"
say "sdpa_bench_pv.py $(sha256sum "$PY" | cut -c1-16) driver $(readlink "$B0/vulkan-main") mx=$("$B0/venv-main/bin/python" -c 'import mlx.core as m; print(m.__version__)') libmlx=$(sha256sum "$B0"/venv-main/lib/python3.14/site-packages/mlx/lib/libmlx.so | cut -c1-16)"
for arm in fused split splitpv; do
  extra="MLX_AB_UNUSED=1"; [ "$arm" = fused ] && extra="MLX_OMARCHY_SDPA_DECODE_SPLIT=0"; [ "$arm" = split ] && extra="MLX_OMARCHY_SDPA_DECODE_SPLIT=1 MLX_OMARCHY_SDPA_SPLIT_PV=0"; [ "$arm" = splitpv ] && extra="MLX_OMARCHY_SDPA_DECODE_SPLIT=1 MLX_OMARCHY_SDPA_SPLIT_PV=1"
  say "arm $arm ($extra)"
  # shellcheck disable=SC2086
  env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 VK_DRIVER_FILES="$B0/vulkan-main/honeykrisp_icd.aarch64.json" $extra \
    PYTHONPATH="$WORK/tree-pv" timeout 900 "$B0/venv-main/bin/python" "$PY" "$R" 2>&1 | tee -a "$L"
done
say "end $(date -u +%FT%TZ) load1=$(cut -d' ' -f1 /proc/loadavg)"
