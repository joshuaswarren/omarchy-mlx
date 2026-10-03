#!/usr/bin/env bash
# Gate 13 — GDN coopmat correctness smoke (runs ON the jw16 aarch64 build
# host, inside a gpuwin window): build omarchy_gdn_maskless_correctness_tests
# (the captured-operand fp64 doctest, T=63/64/65/96/352 repeats 1/2/3) from
# the wheel build tree with MLX_BUILD_TESTS=ON, then run it. Independent of
# the 27B end-to-end (GdnFix's M2 tickets).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
LOG="$LOG_DIR/g13-gdn-maskless.log"
: > "$LOG"
gate_begin "$LOG"
gate_log "$LOG" "loadavg_before=$(cat /proc/loadavg)"

TREE="$GATE_WORKTREE/.work/mlx"
[[ -d "$TREE" ]] || { gate_log "$LOG" "REFUSING: build tree $TREE missing (build the wheel first)"; exit 1; }

cd "$TREE"
gate_log "$LOG" "== reconfigure with tests =="
nice -n 19 cmake -DMLX_BUILD_TESTS=ON . >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "CMAKE_CONFIGURE_EXIT $RC"
[[ $RC -eq 0 ]] || { gate_log "$LOG" "GATE13_EXIT 1"; exit 1; }

gate_log "$LOG" "== build target =="
nice -n 19 cmake --build . --target omarchy_gdn_maskless_correctness_tests >>"$LOG" 2>&1
RC=$?
gate_log "$LOG" "BUILD_EXIT $RC"
[[ $RC -eq 0 ]] || { gate_log "$LOG" "GATE13_EXIT 1"; exit 1; }

BIN=$(find "$TREE" -name omarchy_gdn_maskless_correctness_tests -type f -executable | head -1)
[[ -x "$BIN" ]] || { gate_log "$LOG" "GATE13_EXIT 1 (binary not found)"; exit 1; }
gate_log "$LOG" "BIN $BIN"

gate_log "$LOG" "== run =="
"$BIN" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
gate_log "$LOG" "DOCTEST_EXIT $RC"
gate_log "$LOG" "GATE13_EXIT $RC"
exit "$RC"
