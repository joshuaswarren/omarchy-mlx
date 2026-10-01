#!/usr/bin/env bash
# Generic: run one worker process under gdb with count_cpu breakpoint,
# drive a real workload, capture dispatch count + native stack chains +
# python stacks (for nonzero hits).
#
# Usage: pairgates_run_worker.sh <worker_label> <workload_script> <out_dir>
#
# workload_script gets passed the worker pid via $PGID; it drives the
# worker (e.g. POST /v1/chat/completions).
set -uo pipefail
LABEL=${1:?label required}
WORKLOAD=${2:?workload script required}
OUT=${3:?out dir required}
mkdir -p "$OUT"

# 1. Control: mx.add on mx.gpu then mx.cpu (no CPU bundle).
echo "=== control: mx.add on mx.gpu ===" | tee "$OUT/control.log"
CPU_COUNT_REDIRECT="> $OUT/control-gpu.out 2>&1" \
  timeout 90 gdb -batch -x <home>/agents/PairGates/worktree/harness/count_cpu.gdb.py \
    --args <home>/.local/share/mlx-omarchy/venv/bin/python \
    <home>/agents/PairGates/worktree/harness/control.py gpu \
  > "$OUT/control-gpu.gdb.txt" 2>&1
cat "$OUT/control-gpu.gdb.txt" | grep "CPU_COUNT" >> "$OUT/control.log"

echo "=== control: mx.add on mx.cpu ===" | tee -a "$OUT/control.log"
CPU_COUNT_REDIRECT="> $OUT/control-cpu.out 2>&1" \
  timeout 90 gdb -batch -x <home>/agents/PairGates/worktree/harness/count_cpu.gdb.py \
    --args <home>/.local/share/mlx-omarchy/venv/bin/python \
    <home>/agents/PairGates/worktree/harness/control.py cpu \
  > "$OUT/control-cpu.gdb.txt" 2>&1
cat "$OUT/control-cpu.gdb.txt" | grep "CPU_COUNT" >> "$OUT/control.log"

# 2. Worker process under gdb (count_cpu).
echo "=== worker: $LABEL (count_cpu) ===" | tee "$OUT/worker_stats.log"
WORKER_OUT="$OUT/worker.out"
CPU_COUNT_REDIRECT="> $WORKER_OUT 2>&1" \
  timeout 600 gdb -batch -x <home>/agents/PairGates/worktree/harness/count_cpu.gdb.py \
    --args <home>/.local/share/mlx-omarchy/venv/bin/python \
    <home>/agents/PairGates/worktree/harness/worker_under_count.py "$LABEL" \
  > "$OUT/worker.gdb.txt" 2>&1 &
GDB_PID=$!
echo "gdb pid=$GDB_PID"

# Wait for worker process to come up and bind its port.
sleep 5
WORKER_PORT=$(grep -oE "WORKER_PORT=[0-9]+" "$WORKER_OUT" | head -1 | grep -oE "[0-9]+$" || true)
echo "worker port: ${WORKER_PORT:-?}"
if [ -z "$WORKER_PORT" ]; then
  echo "FAILED: worker did not bind a port"
  cat "$WORKER_OUT" | head -50
  kill -KILL $GDB_PID 2>/dev/null
  exit 1
fi

# Drive the workload.
echo "=== driving workload ===" | tee -a "$OUT/worker_stats.log"
bash "$WORKLOAD" "$WORKER_PORT" "$OUT" 2>&1 | tee -a "$OUT/worker_stats.log"

# Let the worker drain, then kill gdb (kills the inferior process group).
sleep 5
kill -INT $GDB_PID 2>/dev/null || true
wait $GDB_PID 2>/dev/null || true

# 3. If nonzero hits: native chains via bt_cpu.gdb.py with the same
# workload.
HIT_COUNT=$(grep "CPU_COUNT" "$OUT/worker.gdb.txt" | tail -1 | python3 -c "import json,sys; print(json.load(sys.stdin)['cpu_command_encoder_calls'])" 2>/dev/null || echo 0)
echo "cpu_command_encoder_calls=$HIT_COUNT" | tee -a "$OUT/worker_stats.log"

if [ "$HIT_COUNT" != "0" ] && [ "$HIT_COUNT" != "" ]; then
  echo "=== capturing native stacks via bt_cpu.gdb.py ===" | tee -a "$OUT/worker_stats.log"
  CPU_COUNT_REDIRECT="> $OUT/bt_worker.out 2>&1" \
    timeout 600 gdb -batch -x <home>/agents/PairGates/worktree/harness/bt_cpu.gdb.py \
      --args <home>/.local/share/mlx-omarchy/venv/bin/python \
      <home>/agents/PairGates/worktree/harness/worker_under_count.py "$LABEL" \
    > "$OUT/bt_worker.gdb.txt" 2>&1 &
  BT_GDB_PID=$!
  sleep 5
  BT_PORT=$(grep -oE "WORKER_PORT=[0-9]+" "$OUT/bt_worker.out" | head -1 | grep -oE "[0-9]+$" || true)
  if [ -n "$BT_PORT" ]; then
    bash "$WORKLOAD" "$BT_PORT" "$OUT/bt" 2>&1 | tee -a "$OUT/worker_stats.log" || true
    sleep 3
  fi
  kill -INT $BT_GDB_PID 2>/dev/null || true
  wait $BT_GDB_PID 2>/dev/null || true

  # 4. Python stacks via pybt_cpu.gdb.py.
  echo "=== capturing python stacks via pybt_cpu.gdb.py ===" | tee -a "$OUT/worker_stats.log"
  CPU_COUNT_REDIRECT="> $OUT/pybt_worker.out 2>&1" \
    timeout 600 gdb -batch -x <home>/agents/PairGates/worktree/harness/pybt_cpu.gdb.py \
      --args <home>/.local/share/mlx-omarchy/venv/bin/python \
      <home>/agents/PairGates/worktree/harness/worker_under_count.py "$LABEL" \
    > "$OUT/pybt_worker.gdb.txt" 2>&1 &
  PYBT_GDB_PID=$!
  sleep 5
  PYBT_PORT=$(grep -oE "WORKER_PORT=[0-9]+" "$OUT/pybt_worker.out" | head -1 | grep -oE "[0-9]+$" || true)
  if [ -n "$PYBT_PORT" ]; then
    bash "$WORKLOAD" "$PYBT_PORT" "$OUT/pybt" 2>&1 | tee -a "$OUT/worker_stats.log" || true
    sleep 3
  fi
  kill -INT $PYBT_GDB_PID 2>/dev/null || true
  wait $PYBT_GDB_PID 2>/dev/null || true
fi

echo "DONE: $OUT"
ls -la "$OUT"