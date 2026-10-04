#!/usr/bin/env bash
# Build the GDN-relevant omarchy test suites (CPU, niced, detached-safe).
set -uo pipefail
cd /var/tmp/dfuse-build
bash scripts/prepare-mlx.sh > /var/tmp/dfuse/prepare-tests.log 2>&1 || { echo PREPARE-FAILED; exit 1; }
cmake -S .work/mlx -B .work/mlx/build-tests -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release > /var/tmp/dfuse/tests-config.log 2>&1 || { echo CONFIG-FAILED; tail -5 /var/tmp/dfuse/tests-config.log; exit 1; }
nice -n 19 cmake --build .work/mlx/build-tests --target \
  omarchy_kv_ops_tests omarchy_fast_ops_tests omarchy_gdn_prefill_profile_tests \
  omarchy_gdn_fast_route_repeat_tests omarchy_gdn_maskless_correctness_tests \
  omarchy_gdn_legacy_policy_tests omarchy_capability_sim_tests \
  -j 8 > /var/tmp/dfuse/tests-build.log 2>&1
rc=$?
echo "BUILD-RC=$rc"
tail -3 /var/tmp/dfuse/tests-build.log
exit $rc
