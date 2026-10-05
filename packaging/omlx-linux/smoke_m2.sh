#!/usr/bin/env bash
# oMLX-on-Linux server smoke — runs INSIDE a gpu-turn ticket on the M2.
# Receipts land in /tmp/omlx-smoke/ with wall-clock timestamps everywhere.
set -uo pipefail
mkdir -p /tmp/omlx-smoke && cd /tmp/omlx-smoke
exec > >(tee -a run.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }

VENV=/tmp/omlx-home/.venvs/omlx
PORT=8900
MODEL=mlx-community--Qwen3.5-2B-MLX-4bit
BASE=http://127.0.0.1:$PORT
export HOME=/tmp/omlx-home
# Reuse the pre-cached weights from the real user cache; the throwaway
# install HOME must not trigger a 2 GB wifi re-download.
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE

echo "[$(ts)] === omlx-linux server smoke start ==="
echo "[$(ts)] provenance: $($VENV/bin/python -m pip show omlx mlx-lm mlx-omarchy 2>/dev/null | grep -E '^(Name|Version):' | tr '\n' ' ')"

cleanup() { [[ -n ${SERVER_PID:-} ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

# --- start server ---
echo "[$(ts)] starting server: $VENV/bin/omlx serve --host 127.0.0.1 --port $PORT --model $MODEL"
"$VENV/bin/omlx" serve --host 127.0.0.1 --port "$PORT" --model "$MODEL" > server.log 2>&1 &
SERVER_PID=$!
UP=0
for i in $(seq 1 90); do
  if curl -s --max-time 2 "$BASE/health" >health.json 2>/dev/null; then UP=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -30 server.log; exit 1; }
  sleep 1
done
if (( ! UP )); then echo "[$(ts)] server never became healthy"; tail -30 server.log; exit 1; fi
echo "[$(ts)] server healthy after ~${i}s"
echo "--- /v1/models ---"
curl -s --max-time 5 "$BASE/v1/models" -o models.json && head -c 400 models.json && echo

# --- non-stream chat ---
PROMPT='Explain in one short paragraph why the sky is blue.'
echo "[$(ts)] non-stream chat (max_tokens 64, temp 0)"
S=$(date +%s.%N)
curl -s --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":64,\"temperature\":0}" \
  -o nonstream.json
E=$(date +%s.%N)
echo "[$(ts)] non-stream wall: $(echo "$E - $S" | bc)s"
python3 - <<'PY'
import json
d = json.load(open("nonstream.json"))
print("nonstream content:", repr(d["choices"][0]["message"]["content"][:200]))
print("usage:", d.get("usage"))
PY

# --- stream chat (TTFT + tok/s) ---
echo "[$(ts)] stream chat (max_tokens 80, temp 0)"
S=$(date +%s.%N)
curl -sN --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":80,\"temperature\":0}" \
  -o stream.sse
E=$(date +%s.%N)
echo "[$(ts)] stream wall: $(echo "$E - $S" | bc)s"
python3 - <<'PY'
import json, time
first, n = None, 0
t0 = time.monotonic()
text = []
for line in open("stream.sse"):
    line = line.strip()
    if not line.startswith("data:"): continue
    payload = line[5:].strip()
    if payload == "[DONE]": break
    try: d = json.loads(payload)
    except Exception: continue
    delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
    if delta:
        if first is None: first = time.monotonic() - t0
        n += 1
        text.append(delta)
total = time.monotonic() - t0
print(f"stream chunks={n} ttft={first:.3f}s total={total:.3f}s tok/s={n/total:.1f}" if first else "stream: no chunks!")
print("stream text:", repr("".join(text)[:200]))
open("stream_metrics.txt","w").write(f"ttft_s={first}\nchunks={n}\ntotal_s={total}\n")
PY

# --- digest equality: single vs batched-4 ---
python3 - <<'PY'
import json, hashlib
P_SYS = "You are a careful assistant. Answer with a numbered list of exactly five items, each a short sentence about the water cycle."
P_USR = "Describe the water cycle."
def body(prompt_override=None):
    msgs = [{"role":"system","content":P_SYS},{"role":"user","content":prompt_override or P_USR}]
    return {"model":"mlx-community--Qwen3.5-2B-MLX-4bit","messages":msgs,"max_tokens":96,"temperature":0}
open("req_digest.json","w").write(json.dumps(body()))
open("req_b1.json","w").write(json.dumps(body("Name three primary colors.")))
open("req_b2.json","w").write(json.dumps(body("What is the capital of France?")))
open("req_b3.json","w").write(json.dumps(body("Count from one to five.")))
PY

echo "[$(ts)] digest-single request"
S=$(date +%s.%N)
curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d @req_digest.json -o single.json
E=$(date +%s.%N); echo "[$(ts)] single wall: $(echo "$E - $S" | bc)s"

echo "[$(ts)] 4-concurrent batch (digest prompt + 3 others)"
S=$(date +%s.%N)
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_digest.json -o batch0.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b1.json -o batch1.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b2.json -o batch2.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b3.json -o batch3.json &
wait
E=$(date +%s.%N); echo "[$(ts)] batch wall (4 concurrent): $(echo "$E - $S" | bc)s"

python3 - <<'PY'
import json, hashlib
def text_of(p):
    d = json.load(open(p))
    return d["choices"][0]["message"]["content"]
def digest(p):
    return hashlib.sha256(text_of(p).encode()).hexdigest()[:16]
s, b = digest("single.json"), digest("batch0.json")
print("single digest:", s)
print("batched digest:", b)
print("GREEDY-EQUAL:", s == b)
open("digest_result.txt","w").write(f"single={s}\nbatched={b}\nequal={s==b}\n")
for i in (1,2,3):
    try: print(f"batch{i} ok, {len(text_of(f'batch{i}.json'))} chars")
    except Exception as e: print(f"batch{i} FAILED: {e}")
PY

# --- prefix reuse TTFT (deferred if past 14 min of ticket) ---
ELAPSED=$((SECONDS / 60))
if (( ELAPSED < 14 )); then
  echo "[$(ts)] prefix-reuse probe (two requests, same long prefix)"
  python3 - <<'PY'
import json
long_ctx = ("The following reference passage is provided for context.\n\n" + ("Water evaporates, condenses, and precipitates in a closed loop. " * 40)) + "\n\nQuestion: "
open("req_pfx1.json","w").write(json.dumps({"model":"mlx-community--Qwen3.5-2B-MLX-4bit","stream":True,"messages":[{"role":"user","content":long_ctx+"Summarize the passage in one sentence."}],"max_tokens":48,"temperature":0}))
open("req_pfx2.json","w").write(json.dumps({"model":"mlx-community--Qwen3.5-2B-MLX-4bit","stream":True,"messages":[{"role":"user","content":long_ctx+"List the three stages mentioned."}],"max_tokens":48,"temperature":0}))
PY
  for n in 1 2; do
    S=$(date +%s.%N)
    curl -sN --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_pfx$n.json -o pfx$n.sse
    E=$(date +%s.%N)
    python3 - "$n" "$S" "$E" <<'PY'
import sys, json, time
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
else
  echo "[$(ts)] prefix probe DEFERRED (ticket at ${ELAPSED}m) — recorded honestly as not-run"
fi

echo "[$(ts)] === smoke end ==="
tail -5 server.log
