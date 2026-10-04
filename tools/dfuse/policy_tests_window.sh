#!/usr/bin/env bash
# Runs INSIDE one gpuwin window: the GDN legacy-policy doctest plus the
# neighbouring GDN suites, each with its exit code.
# usage: policy_tests_window.sh <build-tests-dir> <label>
set -uo pipefail
B="$1"; L="$2"
O=/var/tmp/dfuse/policy-tests-$L
mkdir -p "$O"
for t in omarchy_gdn_legacy_policy_tests omarchy_gdn_fast_route_repeat_tests \
         omarchy_gdn_maskless_correctness_tests omarchy_gdn_prefill_profile_tests; do
  timeout 300 "$B/tests/omarchy/$t" > "$O/$t.log" 2>&1
  rc=$?
  echo "$L $t rc=$rc $(grep -E '^\[doctest\] (test cases|assertions)' "$O/$t.log" | tr '\n' ' ')"
  grep -h '^\[gdn_legacy_policy\]' "$O/$t.log" || true
done
