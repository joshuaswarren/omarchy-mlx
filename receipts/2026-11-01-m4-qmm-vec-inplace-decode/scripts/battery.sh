#!/bin/bash
# usage: battery.sh <variant> <suite>...   (suite "capsim" = all six
# omarchy_capability_sim_tests profiles). One process per suite under
# flock /tmp/m1-gpu.lock with a 1800 s timeout; after a crash or timeout
# a fresh-process device reopen decides whether to continue.
U=$HOME/gpu/upr; V=$1; shift
export VK_ICD_FILENAMES=$HOME/gpu/mesa-install/share/vulkan/icd.d/asahi_icd.aarch64.json LD_LIBRARY_PATH=$HOME/gpu/mesa-install/lib AGX_APPLE9_MAX_SHARED=32768 MLX_OMARCHY_BATCH_WORK=120000
T=$U/tests-$V; L=$U/logs/battery-$V${BATTERY_TAG:-}; mkdir -p $L
S=$L/summary.txt
errs() { journalctl -k -b --no-pager -q | grep asahi | grep -ciE "fault|error|timeout|failed"; }
echo "battery $V $(date -u +%FT%TZ) commit=$(grep "^$V " $U/commits.txt | cut -d' ' -f2) extra_env=${BATTERY_ENV:-none}" >> $S
run() {
  local name=$1; shift
  local e0=$(errs) t0=$(date +%s)
  env ${BATTERY_ENV:-} flock /tmp/m1-gpu.lock timeout 1800 "$@" > $L/$name.log 2>&1; local rc=$?
  local e1=$(errs)
  local sum=$(grep -E "^\[doctest\] (test cases|assertions)" $L/$name.log | tr '\n' ' ')
  echo "$name rc=$rc secs=$(( $(date +%s) - t0 )) new_gpu_kernel_errors=$((e1 - e0)) $sum" >> $S
  if [ $rc -gt 1 ] || [ $((e1 - e0)) -gt 0 ]; then
    EXPECT_COMMIT=$(grep "^$V " $U/commits.txt | cut -d' ' -f2) timeout 120 $U/venv-$V/bin/python $U/probes/reopen.py > $L/$name.reopen.log 2>&1
    local r=$?; echo "  reopen after $name rc=$r $(tail -1 $L/$name.reopen.log)" >> $S
    if [ $r -ne 0 ]; then echo "GPU NOT REOPENABLE - STOP" >> $S; exit 9; fi
  fi
}
for s in "$@"; do
  if [ "$s" = capsim ]; then
    for p in ${CAPSIM_PROFILES:-m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix m1-g13-legacy}; do
      MLX_OMARCHY_ALLOW_NON_APPLE=1 run capsim-$p $T/omarchy_capability_sim_tests $p
    done
  else
    run $s $T/$s
  fi
done
echo "BATTERY_DONE $(date -u +%FT%TZ)" >> $S
