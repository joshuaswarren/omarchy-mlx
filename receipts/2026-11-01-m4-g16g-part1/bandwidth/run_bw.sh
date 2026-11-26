#!/bin/bash
# Build the probes from this directory's sources on the target and run
# them. Writes bw.log (every command line beside its output and exit
# status), plus the SPIR-V load counts of each shader variant.
# Needs: gcc, glslc, python3, Vulkan headers and loader, an idle GPU.
# CFLAGS may carry the Vulkan header path (-I...); the Vulkan driver comes
# from the environment (VK_ICD_FILENAMES, LD_LIBRARY_PATH).
#
# Failure contract: the first command that fails (a compile, the shader
# generator, a probe) is logged with its status, the script stops there
# and exits with that status, and no later step runs. Earlier build
# outputs in the output directory are deleted first, so a failed rebuild
# can never leave an older executable or shader in place. A run that
# completes ends the log with DONE. No step is a shell pipeline: the
# OpLoad counts compile to an assembly file under run() first and count
# from that file second, so a compiler failure stops the run with the
# compiler's status. test_run_bw.sh checks this without a GPU. BW_SLEEP
# sets the idle pause between GPU probes (default 5 s).
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); O=${1:-$D/out}; S=${BW_SLEEP:-5}
mkdir -p "$O"; L=$O/bw.log; : > "$L"
rm -f "${O:?}/vkbw" "${O:?}/vkchain" "${O:?}/cpubw" "${O:?}"/*.spv "${O:?}"/*.spvasm "${O:?}"/icu_*.comp "${O:?}"/icl_*.comp
run() {
  echo "\$ $*" >> "$L"
  "$@" >> "$L" 2>&1
  local rc=$?
  echo "rc=$rc" >> "$L"
  if [ $rc -ne 0 ]; then
    echo "FAILED rc=$rc: $*; stopping" >> "$L"
    echo "run_bw: '$*' failed with $rc; see $L" >&2
    exit $rc
  fi
}
count_loads() {
  local n
  [ -r "$2" ] || { echo "count_loads: $2 missing"; return 1; }
  n=$(grep -c OpLoad "$2")
  echo "OpLoad count $1: ${n:-0}"
}
echo "env VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-} AGX_APPLE9_MAX_SHARED=${AGX_APPLE9_MAX_SHARED:-}" | sed "s#$HOME#~#g" >> "$L"
echo "date $(date -u +%FT%TZ) kernel $(uname -r) glslc $(glslc --version 2>/dev/null | head -1) gcc $(gcc -dumpversion 2>/dev/null)" >> "$L"
G="glslc -O --target-env=vulkan1.3"
run gcc -O2 ${CFLAGS:-} -o "$O/vkbw" "$D/vkbw.c" -lvulkan
run gcc -O2 ${CFLAGS:-} -o "$O/vkchain" "$D/vkchain.c" -lvulkan
run gcc -O2 -pthread -o "$O/cpubw" "$D/cpubw.c"
for v in 1 2 4; do run $G -DVEC=$v -DUNROLL=4 "$D/bw_read.comp" -o "$O/read_v$v.spv"; done
for w in 0 1; do run $G -DWIDE=$w "$D/bw_gemv.comp" -o "$O/gemv_w$w.spv"; done
run $G -DWIDE=0 -DALU=16 "$D/bw_gemv.comp" -o "$O/gemv_w0_alu16.spv"
run $G -DWIDE=0 -DALU=64 "$D/bw_gemv.comp" -o "$O/gemv_w0_alu64.spv"
run $G -DWIDE=0 -DEXTRA=32 "$D/bw_gemv.comp" -o "$O/gemv_w0_extra32.spv"
run sh -c "cd '$O' && python3 '$D/gen_icache.py'"
for f in "$O"/icu_*.comp "$O"/icl_*.comp; do run $G "$f" -o "${f%.comp}.spv"; done
for v in 1 2 4; do
  run $G -S -DVEC=$v -DUNROLL=4 "$D/bw_read.comp" -o "$O/read_v$v.spvasm"
  run count_loads "read_v$v" "$O/read_v$v.spvasm"
done
for w in 0 1; do
  run $G -S -DWIDE=$w "$D/bw_gemv.comp" -o "$O/gemv_w$w.spvasm"
  run count_loads "gemv_w$w" "$O/gemv_w$w.spvasm"
done
for f in "$O"/*.spv; do
  echo "spv $(basename "$f") bytes=$(wc -c < "$f" | tr -d ' ') sha256=$(sha256sum "$f" | cut -c1-16)" >> "$L"
done
B=$O/vkbw
for r in 1 2 3; do
  for v in 1 2 4; do sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/read_v$v.spv" $v 1280 -1 1024 5; done
done
for r in 1 2; do
  sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/gemv_w0.spv" 2 640 -1 1024 5
  sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/gemv_w1.spv" 4 640 -1 1024 5
  sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/gemv_w0_alu16.spv" 2 640 -1 1024 5
  sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/gemv_w0_alu64.spv" 2 640 -1 1024 5
  sleep "$S"; run flock /tmp/m1-gpu.lock "$B" "$O/gemv_w0_extra32.spv" 2 640 -1 1024 5
done
sleep "$S"; run "$O/cpubw"
for g in 4 160; do for ops in 250 1000 4000; do for kind in icu icl; do
  sleep "$S"; run flock /tmp/m1-gpu.lock "$O/vkchain" "$O/${kind}_$ops.spv" 1 $g -1 1 5 300 1
done; done; done
echo DONE >> "$L"
