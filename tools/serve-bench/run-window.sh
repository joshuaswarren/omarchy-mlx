#!/usr/bin/env bash
# GPU-window wrapper for serve_bench.py on the M1 Max Linux box.
# Protocol (matches 2026-09-19 receipts + GPUpeer deconfliction):
#   1. announce TAKE with lock inode
#   2. stop llm-inference
#   3. flock /tmp/m1-gpu.lock
#   4. run the harness
#   5. RELEASE the lock  (trap order: unlock BEFORE service restart —
#      llm-inference ExecStart uses flock --nonblock and fails while the
#      window holder still holds the lock; observed 2026-09-19)
#   6. restart llm-inference + verify /health + real completion
#   7. announce RELEASE
# Usage: run-window.sh [serve_bench.py args...]
set -u

LOCK=/tmp/m1-gpu.lock
INODE=$(stat -c %i "$LOCK" 2>/dev/null || stat -c %i /tmp)
echo "TAKE requesting t6001-test-host GPU window lock-inode=$INODE $(date -Is)"

echo "== stopping llm-inference =="
if ! sudo -n systemctl stop llm-inference; then
  echo "FATAL: cannot stop llm-inference"
  exit 9
fi
sudo -n systemctl reset-failed llm-inference 2>/dev/null

exec 9>"$LOCK"
if ! flock -w 180 9; then
  echo "FATAL: GPU lock still busy after 180s"
  restart_service
  exit 10
fi
echo "TAKE held t6001-test-host GPU window lock-inode=$INODE $(date -Is)"

PY=${SERVEBENCH_PY:-/tmp/servebench/venv/bin/python}
rc=0
"$PY" "$(dirname "$0")/serve_bench.py" "$@" || rc=$?
echo "== serve_bench rc=$rc =="

flock -u 9
echo "RELEASE t6001-test-host GPU window lock-inode=$INODE $(date -Is)"

echo "== restarting llm-inference =="
sudo -n systemctl start llm-inference 2>&1 || echo "ERROR: systemctl start failed"
for i in $(seq 1 24); do
  if curl -sf --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1; then
    echo "llm-inference /health 200 after ~$((i * 5))s"
    break
  fi
  sleep 5
done
KEY=$(sudo -n cat /etc/llm-inference/api-key 2>/dev/null)
echo "== real completion check =="
curl -s --max-time 120 http://127.0.0.1:8002/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}' \
  | head -c 400
echo
exit $rc
