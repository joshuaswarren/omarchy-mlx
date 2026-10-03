#!/usr/bin/env bash
# Gate 13 — GDN coopmat correctness smoke on the jw16 aarch64 build host.
#   g13-gdn-maskless.sh build   CPU-only: reconfigure the wheel build tree
#                               with MLX_BUILD_OMARCHY/TESTS=ON and build
#                               omarchy_gdn_maskless_correctness_tests
#                               (run OUTSIDE gpuwin — no GPU needed).
#   g13-gdn-maskless.sh run     run the doctest binary (INSIDE a gpuwin
#                               window: the coopmat shader needs the GPU).
# The captured-operand fp64 doctest covers T=63/64/65/96/352, repeats 1/2/3
# (fixtures overlay/tests/omarchy/fixtures/gdn_coopmat/, GDN_FIXTURE_DIR is
# a compile-time define). Independent of GdnFix's 27B end-to-end.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g13-gdn-maskless.log"
MODE="${1:-run}"
TREE="$GATE_WORKTREE/.work/mlx"

[[ -d "$TREE" ]] || { echo "REFUSING: build tree $TREE missing (build the wheel first)" | tee -a "$LOG"; exit 1; }
cd "$TREE"

if [[ "$MODE" == "build" ]]; then
  : > "$LOG"
  gate_begin "$LOG"
  gate_log "$LOG" "== reconfigure omarchy+tests (CPU-only, no GPU) =="
  nice -n 19 cmake -DMLX_BUILD_OMARCHY=ON -DMLX_BUILD_TESTS=ON . >>"$LOG" 2>&1
  RC=$?
  gate_log "$LOG" "CMAKE_CONFIGURE_EXIT $RC"
  [[ $RC -eq 0 ]] || { gate_log "$LOG" "GATE13_BUILD_EXIT 1"; exit 1; }
  gate_log "$LOG" "== build target =="
  nice -n 19 cmake --build . --target omarchy_gdn_maskless_correctness_tests >>"$LOG" 2>&1
  RC=$?
  gate_log "$LOG" "BUILD_EXIT $RC"
  BIN=$(find "$TREE" -name omarchy_gdn_maskless_correctness_tests -type f -executable | head -1)
  [[ -n "$BIN" ]] && gate_log "$LOG" "BIN $BIN"
  [[ $RC -eq 0 && -n "$BIN" ]] && gate_log "$LOG" "GATE13_BUILD_EXIT 0" || gate_log "$LOG" "GATE13_BUILD_EXIT 1"
  exit $([[ $RC -eq 0 && -n "$BIN" ]] && echo 0 || echo 1)
fi

# run mode
BIN=$(find "$TREE" -name omarchy_gdn_maskless_correctness_tests -type f -executable | head -1)
[[ -x "$BIN" ]] || { gate_log "$LOG" "REFUSING: binary not built (run the build stage first)"; exit 1; }
gate_log "$LOG" "BIN $BIN"
gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"
"$BIN" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
gate_log "$LOG" "DOCTEST_EXIT $RC"
gate_log "$LOG" "GATE13_EXIT $RC"
exit "$RC"
