#!/usr/bin/env bash
# run_battery_rebased.sh: run_battery.sh on the rebased test tree (scratch/wt-rebased): info, the sdpa_split suite (G15S case first), the
# standing battery (25 suites), the 8 capability-simulation profiles, device reopen; legs prefixed RB-.
set -uo pipefail
L=${LANE}
B=$L/scratch/wt-rebased/.work/mlx/build-tests; T=$B/tests/omarchy; INFO=$B/tools/mlx-omarchy-info/mlx-omarchy-info
SUM=$L/logs/rb-battery-summary.txt; : > $SUM; G=$L/scripts/gpuleg.sh
summ() { grep -E '^\[doctest\] (test cases|assertions|Status)' "$L/logs/$1.log" | tr '\n' ' '; }
leg() { local name=$1; shift; out=$("$G" "$name" "$@"); rc=$?; echo "$out | $(summ "$name")" | tee -a $SUM; return $rc; }
leg RB-H00-info-before 120 $INFO --json
leg RB-H01-g15s-case 120 $T/omarchy_sdpa_split_tests "--test-case=*M3 Pro (G15S) defaults*"
leg RB-H02-sdpa-split 1800 $T/omarchy_sdpa_split_tests || echo "NONZERO H02" | tee -a $SUM
SUITES=(runtime primitive matmul_family fast_ops kv_ops indexing_ops reduce_ops shape_ops linalg_ops copy_offset distributed compiled_tape fft_ops fft_general eig_ops take_fill conv complex_ops select_layout fast_regression scatter_determinism eq_math fused_chain error_contract ane_bundle)
PROFILES=(m1-honeykrisp-fork m1-stock-no-coopmat subgroup-size-64 small-shared-memory no-cooperative-matrix m1-g13-legacy workgroup-256 workgroup-128)
n=0; for s in "${SUITES[@]}"; do n=$((n+1)); leg $(printf 'RB-S%02d-%s' $n $s) 1800 $T/omarchy_${s}_tests || echo "NONZERO S$n" | tee -a $SUM; done
n=0; for p in "${PROFILES[@]}"; do n=$((n+1)); leg $(printf 'RB-P%02d-%s' $n $p) 1800 env MLX_OMARCHY_ALLOW_NON_APPLE=1 $T/omarchy_capability_sim_tests $p || echo "NONZERO P$n" | tee -a $SUM; done
leg RB-H40-reopen-runtime 120 $T/omarchy_runtime_tests "--test-case=a fresh process reopens the Vulkan device"
leg RB-H41-reopen-smoke 120 $INFO --trace-smoke
leg RB-H42-info-after 120 $INFO --json
echo RB_BATTERY_DONE
