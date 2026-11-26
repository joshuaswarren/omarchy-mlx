#!/usr/bin/env bash
# Two-rank ring harness for the five distributed primitives.
#
# Forms a real two-rank group over loopback: this script launches the
# omarchy_two_rank_harness doctest binary twice with MLX_RANK=0 and
# MLX_RANK=1 against a generated localhost hostfile (JSON, parsed by
# ring.cpp load_nodes), waits for both ranks, and fails unless BOTH
# processes exit zero and print their TWORANK_OK rank=N size=2 marker.
# Every value check runs on both ranks; a lone single-rank run of the
# binary fails its own group-size guards instead of passing vacuously.
#
# Usage (from anywhere; target must be built first):
#   MLX_OMARCHY_ALLOW_NON_APPLE=1 \
#     overlay/tests/omarchy/distributed/run-two-rank.sh [harness-binary]
# Environment:
#   MLX_OMARCHY_WORK_DIR  build root when no binary is given
#                         (default <repo>/.work)
#   TWO_RANK_PORTS        loopback ring ports (default "55000 55001")
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
WORK_DIR="${MLX_OMARCHY_WORK_DIR:-$ROOT/.work}"
BIN="${1:-$WORK_DIR/build/tests/omarchy/omarchy_two_rank_harness}"
read -r PORT0 PORT1 <<< "${TWO_RANK_PORTS:-55000 55001}"

if [[ ! -x "$BIN" ]]; then
  echo "missing harness binary: $BIN" >&2
  echo "build it with: cmake --build $WORK_DIR/build --target omarchy_two_rank_harness" >&2
  exit 2
fi

ulimit -c 0
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tworank.XXXXXX")"
cleanup() { rm -rf "$RUN_DIR"; }
trap cleanup EXIT

printf '[["127.0.0.1:%s"], ["127.0.0.1:%s"]]\n' "$PORT0" "$PORT1" \
  > "$RUN_DIR/hosts.json"

MLX_HOSTFILE="$RUN_DIR/hosts.json" MLX_RANK=0 \
  MLX_OMARCHY_ALLOW_NON_APPLE=1 timeout 120 "$BIN" \
  > "$RUN_DIR/rank0.log" 2>&1 &
PID0=$!
MLX_HOSTFILE="$RUN_DIR/hosts.json" MLX_RANK=1 \
  MLX_OMARCHY_ALLOW_NON_APPLE=1 timeout 120 "$BIN" \
  > "$RUN_DIR/rank1.log" 2>&1 &
PID1=$!

FAIL=0
wait "$PID0" || FAIL=1
wait "$PID1" || FAIL=1
grep -q 'TWORANK_OK rank=0 size=2' "$RUN_DIR/rank0.log" || FAIL=1
grep -q 'TWORANK_OK rank=1 size=2' "$RUN_DIR/rank1.log" || FAIL=1

if [[ $FAIL -ne 0 ]]; then
  echo "two-rank harness FAILED; logs:" >&2
  cat "$RUN_DIR/rank0.log" "$RUN_DIR/rank1.log" >&2
  exit 1
fi
cat "$RUN_DIR/rank0.log" "$RUN_DIR/rank1.log"
echo "two-rank harness verified: both ranks green"
