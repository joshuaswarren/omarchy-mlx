#!/usr/bin/env bash
# Drive ONE worker (chat | decision | tts) under gdb with the count_cpu
# breakpoint. Process model:
#   1. Run control.py gpu + cpu (no warmup; baseline + positive control).
#   2. Launch gdb -batch with worker_under_count.py as the inferior.
#      worker_under_count.py starts the actual server subprocess and
#      announces WORKER_PORT=<port> via its stdout (which gdb captures
#      to its own stdout).
#   3. Tail gdb's stdout; when WORKER_PORT appears, drive the workload
#      via HTTP from this script.
#   4. Send "DONE\n" to the inferior's stdin (which gdb forwards), so
#      worker_under_count.py shuts down its server and exits. gdb then
#      prints CPU_COUNT and exits.
#   5. Parse CPU_COUNT, capture native + python stacks if nonzero.
#
# Usage: run_one_worker_gdb.sh <label> <workload_port>
#
# The script traps and kills the whole process group on exit.
set -uo pipefail
LABEL=${1:?label required}
WORKLOAD_PORT=${2:?workload port required}
OUT_DIR=${3:?out dir required}
HARNESS=<home>/agents/PairGates/worktree/harness

mkdir -p "$OUT_DIR"

echo "=== controls ===" | tee "$OUT_DIR/control.log"
for dev in gpu cpu; do
  echo "--- control-$dev ---" | tee -a "$OUT_DIR/control.log"
  CPU_COUNT_REDIRECT="> $OUT_DIR/control-$dev.out 2>&1" \
    timeout 90 gdb -batch -x "$HARNESS/count_cpu.gdb.py" \
      --args <home>/.local/share/mlx-omarchy/venv/bin/python \
      "$HARNESS/control.py" "$dev" \
    > "$OUT_DIR/control-$dev.gdb.txt" 2>&1
  grep "CPU_COUNT" "$OUT_DIR/control-$dev.gdb.txt" | tee -a "$OUT_DIR/control.log" || true
done

echo "=== worker: $LABEL ===" | tee "$OUT_DIR/worker.log"
WORKER_OUT="$OUT_DIR/worker.out"

# Run gdb with the inferior that will host the worker.
timeout 1200 gdb -batch -x "$HARNESS/count_cpu.gdb.py" \
  --args <home>/.local/share/mlx-omarchy/venv/bin/python \
  "$HARNESS/worker_under_count.py" "$LABEL" \
  > "$WORKER_OUT" 2>&1 &
GDB_PID=$!
trap 'kill -KILL -"$GDB_PID" 2>/dev/null || true' EXIT

# Wait for WORKER_PORT to appear in gdb's stdout.
WORKER_PORT=""
for i in {1..300}; do
  if grep -q "WORKER_PORT=" "$WORKER_OUT" 2>/dev/null; then
    WORKER_PORT=$(grep "WORKER_PORT=" "$WORKER_OUT" | head -1 | sed 's/.*=//')
    break
  fi
  sleep 1
done
if [ -z "$WORKER_PORT" ]; then
  echo "FAILED: worker did not announce a port" | tee -a "$OUT_DIR/worker.log"
  cat "$WORKER_OUT" | tail -50 | tee -a "$OUT_DIR/worker.log"
  kill -KILL $GDB_PID 2>/dev/null
  exit 1
fi
echo "worker port: $WORKER_PORT" | tee -a "$OUT_DIR/worker.log"

# Drive the workload via HTTP. The workload script reads WORKER_PORT
# from $1.
"$HARNESS/workload_${LABEL}.sh" "$WORKER_PORT" "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/worker.log"

# Tell the worker to exit.
echo "DONE" | gdb --batch -p "$GDB_PID" -ex "call (void)fflush(0)" -ex "detach" -ex "quit" 2>/dev/null || true
# Easier: send DONE to the inferior via the python inferior script.
# gdb forwards stdin to the inferior by default; since we ran with
# `--args`, stdin is the python process stdin. Send it DONE.
kill -INT "$GDB_PID" 2>/dev/null || true
wait "$GDB_PID" 2>/dev/null || true

# Parse CPU_COUNT from gdb output.
HIT=$(grep "CPU_COUNT" "$WORKER_OUT" | tail -1 | python3 -c "import json,sys; print(json.load(sys.stdin)['cpu_command_encoder_calls'])" 2>/dev/null || echo "?")
echo "cpu_command_encoder_calls=$HIT" | tee -a "$OUT_DIR/worker.log"

# If nonzero, capture native + python stacks (run again with bt_cpu).
if [ "$HIT" != "0" ] && [ "$HIT" != "?" ]; then
  echo "=== capturing native stacks via bt_cpu.gdb.py ===" | tee -a "$OUT_DIR/worker.log"
  timeout 1200 gdb -batch -x "$HARNESS/bt_cpu.gdb.py" \
    --args <home>/.local/share/mlx-omarchy/venv/bin/python \
    "$HARNESS/worker_under_count.py" "$LABEL" \
    > "$OUT_DIR/bt.out" 2>&1 &
  BT_PID=$!
  for i in {1..300}; do
    if grep -q "WORKER_PORT=" "$OUT_DIR/bt.out" 2>/dev/null; then
      BT_PORT=$(grep "WORKER_PORT=" "$OUT_DIR/bt.out" | head -1 | sed 's/.*=//')
      break
    fi
    sleep 1
  done
  "$HARNESS/workload_${LABEL}.sh" "$BT_PORT" "$OUT_DIR/bt" 2>&1 | tee -a "$OUT_DIR/worker.log"
  kill -INT "$BT_PID" 2>/dev/null || true
  wait "$BT_PID" 2>/dev/null || true
  grep "CPU_BT" "$OUT_DIR/bt.out" | head -20 | tee -a "$OUT_DIR/worker.log"

  echo "=== capturing python stacks via pybt_cpu.gdb.py ===" | tee -a "$OUT_DIR/worker.log"
  timeout 1200 gdb -batch -x "$HARNESS/pybt_cpu.gdb.py" \
    --args <home>/.local/share/mlx-omarchy/venv/bin/python \
    "$HARNESS/worker_under_count.py" "$LABEL" \
    > "$OUT_DIR/pybt.out" 2>&1 &
  PYBT_PID=$!
  for i in {1..300}; do
    if grep -q "WORKER_PORT=" "$OUT_DIR/pybt.out" 2>/dev/null; then
      PYBT_PORT=$(grep "WORKER_PORT=" "$OUT_DIR/pybt.out" | head -1 | sed 's/.*=//')
      break
    fi
    sleep 1
  done
  "$HARNESS/workload_${LABEL}.sh" "$PYBT_PORT" "$OUT_DIR/pybt" 2>&1 | tee -a "$OUT_DIR/worker.log"
  kill -INT "$PYBT_PID" 2>/dev/null || true
  wait "$PYBT_PID" 2>/dev/null || true
  grep -E "PYSTACK|HIT|TOTAL" "$OUT_DIR/pybt.out" | head -40 | tee -a "$OUT_DIR/worker.log"
fi

echo "DONE: $OUT_DIR"
ls -la "$OUT_DIR"