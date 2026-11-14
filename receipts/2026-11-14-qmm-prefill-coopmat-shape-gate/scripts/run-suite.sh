#!/bin/bash
# MLXCOOPFIX: run the patched omarchy C++ suites on the J616s through the mesa
# #299 overlay ICD, coopmat ON and OFF. The headline gate is
# omarchy_matmul_family_tests 47/47 (0 failed) with coopmat ON. Also the host
# selector unit test (no GPU), the runtime + copy-offset suites, and the
# capability-simulation suite. Runs in the foreground; prints a summary and
# writes results under /var/tmp/mlxcoopfix/results.
set -u
W=/var/tmp/mlxcoopfix
OV=$W/honeykrisp_icd.aarch64.json
T=$W/omx/.work/build/tests/omarchy
R=$W/results; mkdir -p $R
export MESA_SHADER_CACHE_DISABLE=1
export HF_HUB_OFFLINE=1
fails=0

line(){ printf '\n===== %s =====\n' "$*"; }
summ(){ grep -iE "test cases:|assertions:|\[doctest\].*failed|Status:" "$1" | tail -4; }
run_doctest(){ # name binary logfile [extra args...]
  local name="$1" bin="$2" logf="$3"; shift 3
  if [ ! -x "$bin" ]; then echo "MISSING $name ($bin)"; fails=$((fails+1)); return; fi
  "$bin" "$@" >"$logf" 2>&1; local ec=$?
  echo "[$name] exit=$ec"; summ "$logf"
  [ $ec -eq 0 ] || fails=$((fails+1))
}

line "device + coopmat advertisement (overlay $OV)"
VK_DRIVER_FILES=$OV ASAHI_AGX3B_COOPERATIVE_MATRIX=1 vulkaninfo 2>/dev/null | \
  grep -iE "deviceName|cooperative_matrix" | sort -u | tee $R/vkinfo-on.txt
VK_DRIVER_FILES=$OV ASAHI_AGX3B_COOPERATIVE_MATRIX=0 vulkaninfo 2>/dev/null | \
  grep -iE "cooperative_matrix" | sort -u | tee $R/vkinfo-off.txt

line "host selector unit test (no GPU, no overlay)"
run_doctest coopmat_select "$T/omarchy_coopmat_select_tests" "$R/coopmat_select.log"

for flag in 1 0; do
  tag=$([ "$flag" = 1 ] && echo on || echo off)
  export VK_DRIVER_FILES=$OV VK_ICD_FILENAMES=$OV ASAHI_AGX3B_COOPERATIVE_MATRIX=$flag
  line "matmul_family  coopmat=$tag (flag=$flag)"
  run_doctest "matmul_family/$tag" "$T/omarchy_matmul_family_tests" "$R/mf-$tag.log"
  line "runtime  coopmat=$tag"
  run_doctest "runtime/$tag" "$T/omarchy_runtime_tests" "$R/rt-$tag.log"
  line "copy_offset  coopmat=$tag"
  run_doctest "copy_offset/$tag" "$T/omarchy_copy_offset_tests" "$R/co-$tag.log"
  unset VK_DRIVER_FILES VK_ICD_FILENAMES ASAHI_AGX3B_COOPERATIVE_MATRIX
done

line "capability-simulation profiles (selector gating, GPU-agnostic)"
export VK_DRIVER_FILES=$OV VK_ICD_FILENAMES=$OV
for prof in m1-honeykrisp-fork m1-stock-no-coopmat no-cooperative-matrix; do
  run_doctest "capsim/$prof" "$T/omarchy_capability_sim_tests" "$R/capsim-$prof.log" "$prof"
done
unset VK_DRIVER_FILES VK_ICD_FILENAMES

line "SUMMARY"
mf_on=$(grep -iE "test cases:" $R/mf-on.log 2>/dev/null | tail -1)
echo "matmul_family coopmat ON: $mf_on"
echo "suite binaries with a nonzero exit: $fails"
if [ $fails -eq 0 ]; then echo "RESULT: ALL SUITES PASS"; touch $R/SUITE-PASS; else echo "RESULT: $fails SUITE(S) FAILED"; touch $R/SUITE-FAIL; fi
