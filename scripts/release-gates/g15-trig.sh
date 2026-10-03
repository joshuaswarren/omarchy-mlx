#!/usr/bin/env bash
# Gate 15 — trig contract tests on the jw16 aarch64 build host.
#   g15-gdn... no: g15 builds and runs omarchy_eq_math_tests and
#   omarchy_trig_reduction_tests from the wheel build tree
#   (MLX_BUILD_OMARCHY=ON MLX_BUILD_TESTS=ON).
#   build stage: CPU-only, plain ssh, OUTSIDE gpuwin.
#   run stage: INSIDE a gpuwin window (the reduction paths hit the GPU).
# The trig contract (de34407c1): one honest sin/cos contract on every
# path — shared Cody-Waite with exact k*C1, NaN above 5e5. v0.7.17
# through v0.7.23 returned silently wrong values for 1e4 <= |x| < 1e7
# (max abs error up to 0.6).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g15-trig.log"
MODE="${1:-run}"
TREE="$GATE_WORKTREE/.work/mlx"
TARGETS="omarchy_eq_math_tests omarchy_trig_reduction_tests"

[[ -d "$TREE" ]] || { echo "REFUSING: build tree $TREE missing (build the wheel first)" | tee -a "$LOG"; exit 1; }
cd "$TREE"

if [[ "$MODE" == "build" ]]; then
  : > "$LOG"
  gate_begin "$LOG"
  gate_log "$LOG" "== reconfigure omarchy+tests (CPU-only) =="
  nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON . >>"$LOG" 2>&1
  RC=$?
  gate_log "$LOG" "CMAKE_CONFIGURE_EXIT $RC"
  [[ $RC -eq 0 ]] || { gate_log "$LOG" "GATE15_BUILD_EXIT 1"; exit 1; }
  gate_log "$LOG" "== build targets =="
  nice -n 19 cmake --build . --target $TARGETS >>"$LOG" 2>&1
  RC=$?
  gate_log "$LOG" "BUILD_EXIT $RC"
  MISSING=0
  for t in $TARGETS; do
    BIN=$(find "$TREE" -name "$t" -type f -executable | head -1)
    [[ -n "$BIN" ]] && gate_log "$LOG" "BIN $BIN" || { gate_log "$LOG" "BIN missing: $t"; MISSING=1; }
  done
  [[ $RC -eq 0 && $MISSING -eq 0 ]] && gate_log "$LOG" "GATE15_BUILD_EXIT 0" || gate_log "$LOG" "GATE15_BUILD_EXIT 1"
  exit $([[ $RC -eq 0 && $MISSING -eq 0 ]] && echo 0 || echo 1)
fi

# run mode
FAILED=0
for t in $TARGETS; do
  BIN=$(find "$TREE" -name "$t" -type f -executable | head -1)
  [[ -x "$BIN" ]] || { gate_log "$LOG" "REFUSING: $t not built (run the build stage first)"; FAILED=1; continue; }
  gate_log "$LOG" "BIN $BIN"
  gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"
  "$BIN" 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}
  gate_log "$LOG" "${t}_EXIT $RC"
  [[ $RC -eq 0 ]] || FAILED=1
done
gate_log "$LOG" "GATE15_EXIT $FAILED"
exit "$FAILED"
