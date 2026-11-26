#!/bin/bash
# Narrowed #316 (f333327: only FUSED_GEMV defaults off on G16G): release wheel p1n with the
# test targets, profiling wheel dp1n, their venvs, then the C++ suites under the GPU lock.
set -uo pipefail
U=$HOME/gpu/upr
grep -q "^p1n " $U/commits.txt || printf "p1n f333327\ndp1n f333327\n" >> $U/commits.txt
if ! grep -q BUILD_OK $U/logs/build-p1n.log 2>/dev/null; then
  bash $U/probes/build.sh p1n f333327 0 omarchy_fused_chain_tests omarchy_kv_ops_tests omarchy_capability_sim_tests || { echo "build p1n failed"; exit 3; }
fi
[ -x $U/venv-p1n/bin/python ] || bash $U/probes/mkvenv.sh p1n || exit 4
if ! grep -q BUILD_OK $U/logs/build-dp1n.log 2>/dev/null; then
  bash $U/probes/build.sh dp1n f333327 1 || { echo "build dp1n failed"; exit 3; }
fi
[ -x $U/venv-dp1n/bin/python ] || bash $U/probes/mkvenv.sh dp1n || exit 4
echo "WHEELS_READY $(date -u +%FT%TZ)"
grep -q BATTERY_DONE $U/logs/battery-p1n/summary.txt 2>/dev/null || bash $U/probes/battery.sh p1n omarchy_fused_chain_tests omarchy_kv_ops_tests capsim
echo "SETUP316N_DONE $(date -u +%FT%TZ)"
