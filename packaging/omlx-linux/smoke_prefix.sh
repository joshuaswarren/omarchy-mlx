#!/usr/bin/env bash
# Prefix-cache TTFT pair (completes the gate ticket that hit the
# bash `wait`-includes-server hang; this script uses explicit pids).
set -uo pipefail
mkdir -p /tmp/omlx-pfx && cd /tmp/omlx-pfx
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

echo "[$(ts)] === prefix TTFT start ==="
"$VENV/bin/omlx" serve --host 127.0.0.1 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  if curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; then break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -30 server.log; exit 1; }
  sleep 1
done
echo "[$(ts)] server up after ${i}s"

python3 - <<'PY'
import json
pre = "Reference passage: " + ("Water evaporates from the surface, condenses into clouds, and returns as precipitation in a closed loop. " * 30)
open("req1.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":pre+"\nQuestion: summarize the passage in one sentence."}],"max_tokens":48,"temperature":0}))
open("req2.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":pre+"\nQuestion: list the three stages mentioned."}],"max_tokens":48,"temperature":0}))
PY

PIDS=""
for n in 1 2; do
  S=$(date +%s.%N)
  echo "$S" > start_$n
  curl -sN --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req$n.json -o pfx$n.sse &
  PIDS="$PIDS $!"
done
# EXPLICIT pids: bare `wait` would also wait for the server job.
wait $PIDS
for n in 1 2; do
  S=$(cat start_$n); E=$(date +%s.%N)
  python3 - "$n" "$S" "$E" <<'PY'
import sys, time, json
n, S, E = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
first=None; nch=0; t0=time.monotonic()
for line in open(f"pfx{n}.sse"):
    if line.startswith("data:"):
        p=line[5:].strip()
        if p=="[DONE]": break
        try: d=json.loads(p)
        except Exception: continue
        if (d.get("choices") or [{}])[0].get("delta",{}).get("content"):
            if first is None: first=time.monotonic()-t0
            nch+=1
print(f"[prefix] req{n}: ttft={first:.3f}s wall={E-S:.3f}s chunks={nch}" if first else f"[prefix] req{n}: NO TOKENS")
PY
done
grep -E "prefix_cache_lookup" server.log | tail -2
echo "[$(ts)] === prefix TTFT end ==="
