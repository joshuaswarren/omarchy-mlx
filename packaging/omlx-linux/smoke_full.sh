#!/usr/bin/env bash
# Full numerics smoke: use a model that avoids the Qwen3.5 VLM custom-kernel
# gap (mlx-community--Qwen3-4B-Instruct-2507-4bit serves cleanly). Document
# the Qwen3.5-2B upstream gap in the receipt.
set -uo pipefail
mkdir -p /tmp/omlx-smoke3 && cd /tmp/omlx-smoke3
exec > >(tee -a run.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }
VENV=/tmp/omlx-home/.venvs/omlx
PORT=8900
BASE=http://127.0.0.1:$PORT
MODEL=mlx-community--Qwen3-4B-Instruct-2507-4bit
export HOME=/tmp/omlx-home
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE
cleanup() { [[ -n ${SERVER_PID:-} ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$(ts)] full numerics smoke start, model=$MODEL"
"$VENV/bin/omlx" serve --host 127.0.0.1 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  if curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; then break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -30 server.log; exit 1; }
  sleep 1
done
echo "[$(ts)] server up after ${i}s"

curl -s --max-time 5 "$BASE/v1/models" -o models.json

# --- non-stream chat (real completion) ---
PROMPT='Name three primary colors, comma-separated, nothing else.'
echo "[$(ts)] non-stream chat"
S=$(date +%s.%N)
curl -s --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0}" \
  -o nonstream.json
E=$(date +%s.%N); echo "[$(ts)] non-stream wall: $(echo "$E - $S" | bc)s"

# --- stream chat ---
echo "[$(ts)] stream chat"
S=$(date +%s.%N)
curl -sN --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":24,\"temperature\":0}" \
  -o stream.sse
E=$(date +%s.%N); echo "[$(ts)] stream wall: $(echo "$E - $S" | bc)s"

# --- digest equality: single vs batched-4 ---
python3 - <<'PY'
import json
P_SYS = "You are a careful assistant. Answer with a numbered list of exactly five items, each a short sentence about the water cycle."
P_USR = "Describe the water cycle."
def body(prompt_override=None):
    msgs = [{"role":"system","content":P_SYS},{"role":"user","content":prompt_override or P_USR}]
    return {"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":msgs,"max_tokens":96,"temperature":0,"stream":False}
open("req_digest.json","w").write(json.dumps(body()))
open("req_b1.json","w").write(json.dumps(body("Name three primary colors.")))
open("req_b2.json","w").write(json.dumps(body("What is the capital of France?")))
open("req_b3.json","w").write(json.dumps(body("Count from one to five.")))
PY

echo "[$(ts)] digest-single"
S=$(date +%s.%N)
curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_digest.json -o single.json
E=$(date +%s.%N); echo "[$(ts)] single wall: $(echo "$E - $S" | bc)s"

echo "[$(ts)] 4-concurrent batch"
S=$(date +%s.%N)
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_digest.json -o batch0.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b1.json -o batch1.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b2.json -o batch2.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b3.json -o batch3.json &
wait
E=$(date +%s.%N); echo "[$(ts)] batch wall: $(echo "$E - $S" | bc)s"

python3 - <<'PY'
import json, hashlib
def text_of(p):
    d = json.load(open(p))
    return d.get("choices",[{}])[0].get("message",{}).get("content","")
def digest(p):
    return hashlib.sha256(text_of(p).encode()).hexdigest()[:16]
s, b = digest("single.json"), digest("batch0.json")
print("single digest:", s)
print("batched digest:", b)
print("GREEDY-EQUAL:", s == b)
open("digest_result.txt","w").write(f"single={s}\nbatched={b}\nequal={s==b}\n")
for i in (1,2,3):
    try: print(f"batch{i} chars={len(text_of(f'batch{i}.json'))}")
    except Exception as e: print(f"batch{i} err={e}")
PY

# --- prefix reuse TTFT ---
echo "[$(ts)] prefix-reuse probe"
python3 - <<'PY'
import json
long_ctx = ("Water evaporates, condenses, and precipitates in a closed loop. " * 40) + "\n\nQuestion: "
open("req_pfx1.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":long_ctx+"Summarize in one sentence."}],"max_tokens":48,"temperature":0}))
open("req_pfx2.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":long_ctx+"List three stages."}],"max_tokens":48,"temperature":0}))
PY
for n in 1 2; do
  S=$(date +%s.%N)
  curl -sN --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_pfx$n.json -o pfx$n.sse
  E=$(date +%s.%N)
  python3 - "$n" "$S" "$E" <<'PY'
import sys, time
n, S, E = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
first = None; t0 = time.monotonic()
for line in open(f"pfx{n}.sse"):
    if line.startswith("data:"):
        p = line[5:].strip()
        if p == "[DONE]": break
        try: d = json.loads(p)
        except Exception: continue
        if (d.get("choices") or [{}])[0].get("delta", {}).get("content"):
            first = time.monotonic() - t0; break
print(f"prefix req {n}: ttft={first:.3f}s wall={E-S:.3f}s" if first else f"prefix req {n}: NO TOKENS")
PY
done

echo "[$(ts)] === smoke end ==="
