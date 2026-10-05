#!/usr/bin/env bash
# Numerics-gate closure: greedy batched-vs-single digest equality on
# Qwen3-4B-Instruct-2507-4bit with the CURRENT interim fences, plus the
# long-shared-prefix TTFT pair. Correctness only; exits immediately.
set -uo pipefail
mkdir -p /tmp/omlx-gate && cd /tmp/omlx-gate
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

echo "[$(ts)] === numerics gate start (interim fences ON) ==="
echo "[$(ts)] provenance: $($VENV/bin/python -m pip show omlx mlx-lm mlx-omarchy 2>/dev/null | grep -E '^(Name|Version):' | tr '\n' ' ')"
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
P_SYS = "You are careful. Answer with a numbered list of exactly five short sentences about the water cycle."
P_USR = "Describe the water cycle."
def body():
    return {"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit",
            "messages":[{"role":"system","content":P_SYS},{"role":"user","content":P_USR}],
            "max_tokens":96,"temperature":0,"stream":False}
open("req_digest.json","w").write(json.dumps(body()))
open("req_b1.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":96,"temperature":0,"stream":False}))
open("req_b2.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":96,"temperature":0,"stream":False}))
open("req_b3.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":96,"temperature":0,"stream":False}))
# long shared prefix (two streaming requests, same prefix, different ask)
pre = "Reference passage: " + ("Water evaporates from the surface, condenses into clouds, and returns as precipitation in a closed loop. " * 30)
open("req_pfx1.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":pre+"\nQuestion: summarize the passage in one sentence."}],"max_tokens":48,"temperature":0}))
open("req_pfx2.json","w").write(json.dumps({"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","stream":True,"messages":[{"role":"user","content":pre+"\nQuestion: list the three stages mentioned."}],"max_tokens":48,"temperature":0}))
PY

# --- THE GATE: single vs batched-4 digest equality ---
echo "[$(ts)] [gate] single digest request"
S=$(date +%s.%N)
curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_digest.json -o single.json
E=$(date +%s.%N); echo "[$(ts)] [gate] single wall: $(echo "$E - $S" | bc)s"

echo "[$(ts)] [gate] 4-concurrent batch (digest prompt = slot 0)"
S=$(date +%s.%N)
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_digest.json -o batch0.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b1.json -o batch1.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b2.json -o batch2.json &
curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_b3.json -o batch3.json &
wait
E=$(date +%s.%N); echo "[$(ts)] [gate] batch wall: $(echo "$E - $S" | bc)s"

python3 - <<'PY'
import json, hashlib
def get(p):
    try: d = json.load(open(p))
    except Exception as e: return {"err": f"parse: {e}"}
    if "error" in d: return {"err": d["error"].get("message","")[:120]}
    ch = d.get("choices") or [{}]
    return {"text": ch[0].get("message",{}).get("content",""), "usage": d.get("usage")}
s, b = get("single.json"), get("batch0.json")
def dg(r): return hashlib.sha256(r.get("text","").encode()).hexdigest()[:16] if "text" in r else None
print("[gate] single:", "ERR "+s["err"] if "err" in s else f"sha={dg(s)} len={len(s['text'])} tok/s={s['usage'].get('generation_tokens_per_second')}")
print("[gate] batch0:", "ERR "+b["err"] if "err" in b else f"sha={dg(b)} len={len(b['text'])} tok/s={b['usage'].get('generation_tokens_per_second')}")
eq = ("text" in s and "text" in b and s["text"] == b["text"])
print("[gate] GREEDY_BATCHED_EQ_SINGLE:", eq)
open("gate_result.txt","w").write(f"single_sha={dg(s)}\nbatch0_sha={dg(b)}\nequal={eq}\n")
for i in (1,2,3):
    r = get(f"batch{i}.json")
    print(f"[gate] batch{i}:", "ERR "+r["err"] if "err" in r else f"len={len(r['text'])} preview={r['text'][:50]!r}")
PY

# --- prefix-cache TTFT pair ---
echo "[$(ts)] [prefix] long-shared-prefix TTFT pair"
for n in 1 2; do
  S=$(date +%s.%N)
  curl -sN --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req_pfx$n.json -o pfx$n.sse
  E=$(date +%s.%N)
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

echo "[$(ts)] === numerics gate end ==="
