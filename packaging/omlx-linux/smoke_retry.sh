#!/usr/bin/env bash
# Retry smoke: try alternative small models that may use a different fast
# path. Greedy non-stream completion; record timings and outputs.
set -uo pipefail
mkdir -p /tmp/omlx-smoke2 && cd /tmp/omlx-smoke2
exec > >(tee -a run.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }
VENV=/tmp/omlx-home/.venvs/omlx
PORT=8900
BASE=http://127.0.0.1:$PORT
export HOME=/tmp/omlx-home
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE
cleanup() { [[ -n ${SERVER_PID:-} ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$(ts)] retry smoke start (alternative small models)"
"$VENV/bin/omlx" serve --host 127.0.0.1 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!
UP=0
for i in $(seq 1 90); do
  if curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; then UP=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -30 server.log; exit 1; }
  sleep 1
done
echo "[$(ts)] server up after ${i}s"

for M in mlx-community--Qwen2.5-0.5B-Instruct-4bit mlx-community--Qwen3-4B-Instruct-2507-4bit SiddhJagani--Qwen3.8-2B-mlx-4Bit prism-ml--Ternary-Bonsai-8B-mlx-2bit; do
  echo "[$(ts)] === $M ==="
  S=$(date +%s.%N)
  curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$M\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly five words.\"}],\"max_tokens\":32,\"temperature\":0}" \
    -o "out_${M}.json"
  E=$(date +%s.%N)
  python3 - "$M" "$S" "$E" <<'PY'
import sys, json, time
m, S, E = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    d = json.load(open(f"out_{m}.json"))
except Exception as e:
    print(f"  {m}: parse error: {e}"); raise SystemExit
if "error" in d:
    print(f"  {m}: ERROR wall={E-S:.2f}s msg={d['error'].get('message','')[:120]}")
else:
    ch = d.get("choices") or [{}]
    text = ch[0].get("message", {}).get("content", "")
    usage = d.get("usage") or {}
    print(f"  {m}: OK wall={E-S:.2f}s text={text!r} usage={usage}")
PY
done

echo "[$(ts)] === retry smoke end ==="
