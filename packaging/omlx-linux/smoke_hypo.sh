#!/usr/bin/env bash
# Hypothesis test: 4B batched-engine cache-corruption. Run with kills
# ON first; if the [broadcast_shapes] error disappears → our patch
# series is guilty (qwen3-rope-norm likely). Then run with kills OFF
# (the default) to confirm reproducibility.
set -uo pipefail
mkdir -p /tmp/omlx-hypo && cd /tmp/omlx-hypo
exec > >(tee -a run.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }
VENV=/tmp/omlx-home/.venvs/omlx
PORT=8900
BASE=http://127.0.0.1:$PORT
export HOME=/tmp/omlx-home
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE

# kills are passed via env from the launcher; we record which set is on
RUNS=""
if [[ -n ${MLX_OMARCHY_ROPE_NORM_FUSE:-} ]]; then RUNS+=" kills-ON"; else RUNS+=" kills-OFF"; fi
echo "[$(ts)] === hypothesis test (${RUNS}) ==="

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
msgs0 = [{"role":"system","content":"You are careful. Answer with a numbered list of exactly five short sentences about the water cycle."},{"role":"user","content":"Describe the water cycle."}]
prompts = ["Describe the water cycle.", "Name three primary colors.", "What is the capital of France?", "Count from one to five."]
for i, p in enumerate(prompts):
    body = {"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":msgs0 if i==0 else [{"role":"user","content":p}],"max_tokens":96,"temperature":0}
    open(f"req_{i}.json","w").write(json.dumps(body))
PY

S=$(date +%s.%N)
for i in 0 1 2 3; do
  curl -s --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_$i.json -o res_$i.json &
done
wait
E=$(date +%s.%N)
echo "[$(ts)] 4-concurrent wall: $(echo "$E - $S" | bc)s"

python3 - <<'PY'
import json, hashlib
def text(p):
    try: d=json.load(open(p))
    except Exception: return None
    if "error" in d: return f"ERROR: {d['error'].get('message','')[:140]}"
    return d.get("choices",[{}])[0].get("message",{}).get("content","")
def digest(p):
    t = text(p) or ""
    return hashlib.sha256(t.encode()).hexdigest()[:16]
for i in range(4):
    t = text(f"res_{i}.json")
    print(f"res_{i} digest={digest(f'res_{i}.json')} text_len={len(t) if t and not t.startswith('ERROR') else 'ERR'} preview={(t or '')[:80]!r}")
# batched vs single digest
import json, hashlib
PY

echo "[$(ts)] === hypothesis end ==="
