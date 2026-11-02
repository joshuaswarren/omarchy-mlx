#!/bin/bash
# Fixture check for run_bw.sh's failure contract; needs no GPU, compiler
# or Vulkan. Stub tools on PATH record every call. Three cases:
#   1. the first compile fails: exit status 1, FAILED in the log, no DONE,
#      no shader compile and no GPU probe, and a stale vkbw from an earlier
#      run is gone;
#   2. the shader generator fails: no generated-shader compile, no probe;
#   3. a probe fails: the run stops at that probe;
#   4. the compiler exits 7 in the OpLoad-count step (glslc -S): the run
#      stops with status 7, logs FAILED, prints no DONE and runs no probe.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
stub() { printf '#!/bin/sh\necho "%s $*" >> "%s/calls"\n%s\n' "$1" "$T" "$2" > "$T/bin/$1"; chmod +x "$T/bin/$1"; }
setup() {
  rm -rf "$T/bin" "$T/out" "$T/calls" "$T/src"; mkdir -p "$T/bin" "$T/out" "$T/src"; : > "$T/calls"
  cp "$HERE"/run_bw.sh "$HERE"/*.c "$HERE"/*.comp "$T/src/"
  printf 'print("generated")\n' > "$T/src/gen_icache.py"
  stub gcc 'for a in "$@"; do case "$prev" in -o) : > "$a"; chmod +x "$a";; esac; prev=$a; done; exit 0'
  stub glslc 'for a in "$@"; do case "$prev" in -o) [ "$a" = - ] || : > "$a";; esac; prev=$a; done; echo "OpLoad"; exit 0'
  stub flock 'exit 0'
  stub sha256sum 'echo 0000000000000000'
}
run() { PATH="$T/bin:$PATH" BW_SLEEP=0 bash "$T/src/run_bw.sh" "$T/out" > /dev/null 2>&1; echo $?; }

setup; stub gcc 'exit 1'; echo old > "$T/out/vkbw"
rc=$(run)
check "compile failure exits 1" '[ "$rc" = 1 ]'
check "compile failure logged" 'grep -q "^FAILED rc=1" "$T/out/bw.log"'
check "no DONE after a failure" '! grep -q "^DONE" "$T/out/bw.log"'
check "no shader compiled after a failed build" '! grep -q "^glslc .*-o" "$T/calls"'
check "no GPU probe after a failed build" '! grep -q "^flock" "$T/calls"'
check "stale vkbw removed" '[ ! -e "$T/out/vkbw" ]'

setup; printf 'raise SystemExit(2)\n' > "$T/src/gen_icache.py"
rc=$(run)
check "generator failure exits 2" '[ "$rc" = 2 ]'
check "no generated shader compiled" '! grep -q "ic[ul]_" "$T/calls"'
check "no GPU probe after generator failure" '! grep -q "^flock" "$T/calls"'

setup; stub flock 'exit 3'
rc=$(run)
check "probe failure exits 3" '[ "$rc" = 3 ]'
check "stops at the first probe" '[ "$(grep -c "^flock" "$T/calls")" = 1 ]'
check "no DONE after a probe failure" '! grep -q "^DONE" "$T/out/bw.log"'

setup; stub glslc 'case " $* " in *" -S "*) exit 7;; esac; for a in "$@"; do case "$prev" in -o) : > "$a";; esac; prev=$a; done; exit 0'
rc=$(run)
check "OpLoad-count compiler failure exits 7" '[ "$rc" = 7 ]'
check "OpLoad-count compiler failure logged" 'grep -q "^FAILED rc=7" "$T/out/bw.log"'
check "no count logged after the failed compile" '! grep -q "^OpLoad count" "$T/out/bw.log"'
check "no DONE after an OpLoad-count failure" '! grep -q "^DONE" "$T/out/bw.log"'
check "no GPU probe after an OpLoad-count failure" '! grep -q "^flock" "$T/calls"'

[ $fail = 0 ] && echo "test_run_bw: OK" || { echo "test_run_bw: FAILED"; exit 1; }
