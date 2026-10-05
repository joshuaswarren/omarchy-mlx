#!/usr/bin/env bash
# Repro Qwen3.5-2B custom-kernel failure to capture the .comp + error.
# Then run kill-switch test: with MLX_OMARCHY_ROPE_NORM_FUSE=0 and
# MLX_OMARCHY_GDN_RAW_REPEAT=0, run 4-concurrent on Qwen3-4B and
# see if the [broadcast_shapes] (2,8,1,64) vs (2,1,1,128) error
# disappears.
set -uo pipefail
mkdir -p /tmp/omlx-repro && cd /tmp/omlx-repro
exec > >(tee -a run.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }
VENV=/tmp/omlx-home/.venvs/omlx
PORT=8900
BASE=http://127.0.0.1:$PORT
export HOME=/tmp/omlx-home
HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"
export HF_HUB_CACHE
# kills ON for the Qwen3.5-2B repro (per Main's hypothesis: the
# qwen3-rope-norm patcher is a B==1 fold; this only fails on the
# 2B custom-kernel path, but turning it off lets the upstream
# composed path run, so we can prove the failure is NOT ours).
# For the 4B batch test we run BOTH ways.
export MLX_OMARCHY_ROPE_NORM_FUSE=0
export MLX_OMARCHY_GDN_RAW_REPEAT=0
cleanup() { [[ -n ${SERVER_PID:-} ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$(ts)] === repro start (kills ON) ==="
"$VENV/bin/omlx" serve --host 127.0.0.1 --port "$PORT" > server.log 2>&1 &
SERVER_PID=$!
for i in $(seq 1 90); do
  if curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1; then break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(ts)] SERVER DIED"; tail -30 server.log; exit 1; }
  sleep 1
done
echo "[$(ts)] server up after ${i}s"

echo "[$(ts)] Qwen3.5-2B generation (capture .comp)"
S=$(date +%s.%N)
curl -s --max-time 30 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"mlx-community--Qwen3.5-2B-MLX-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' \
  -o qwen2b_kills.json
E=$(date +%s.%N)
echo "[$(ts)] Qwen3.5-2B wall (kills ON): $(echo "$E - $S" | bc)s"
cat qwen2b_kills.json | head -c 300; echo
ls -la /tmp/mlx-omarchy-custom-*.comp 2>/dev/null
COMPF=$(ls -t /tmp/mlx-omarchy-custom-*.comp 2>/dev/null | head -1)
if [[ -n $COMPF ]]; then cp "$COMPF" /tmp/omlx-repro/; echo "captured: $COMPF -> /tmp/omlx-repro/$(basename $COMPF)"; fi

# 4-concurrent on Qwen3-4B with kills ON
echo "[$(ts)] 4-concurrent on Qwen3-4B (kills ON)"
for i in 0 1 2 3; do
  python3 -c "import json; json.dump({'model':'mlx-community--Qwen3-4B-Instruct-2507-4bit','messages':[{'role':'system','content':'You are careful. Answer with a numbered list of exactly five short sentences about the water cycle.'},{'role':'user','content':'Describe the water cycle.'}] if $i==0 else [{'role':'user','content':$repr_msg}],'max_tokens':96,'temperature':0}, open('req$i.json','w'))" || true
done
# rebuild four requests via shell (avoid python -c quoting hell)
python3 - <<'PY'
import json
msgs = [{"role":"system","content":"You are careful. Answer with a numbered list of exactly five short sentences about the water cycle."},{"role":"user","content":"Describe the water cycle."}]
prompts = ["Describe the water cycle.", "Name three primary colors.", "What is the capital of France?", "Count from one to five."]
for i, p in enumerate(prompts):
    body = {"model":"mlx-community--Qwen3-4B-Instruct-2507-4bit","messages":msgs if i==0 else [{"role":"user","content":p}],"max_tokens":96,"temperature":0}
    open(f"req4b_{i}.json","w").write(json.dumps(body))
PY
S=$(date +%s.%N)
for i in 0 1 2 3; do
  curl -s --max-time 60 "$BASE/v1/chat/completions" -H 'content-type: application/json' -d @req4b_$i.json -o 4b_kills_$i.json &
done
wait
E=$(date +%s.%N)
echo "[$(ts)] 4-concurrent wall (kills ON): $(echo "$E - $S" | bc)s"
python3 - <<'PY'
import json, hashlib
def text(p):
    try: d=json.load(open(p))
    except Exception: return None
    if "error" in d: return f"ERROR: {d['error'].get('message','')[:80]}"
    return d.get("choices",[{}])[0].get("message",{}).get("content","")
def digest(p):
    t = text(p) or ""
    return hashlib.sha256(t.encode()).hexdigest()[:16]
for i in range(4):
    t = text(f"4b_kills_{i}.json")
    print(f"4b_kills_{i} digest={digest(f'4b_kills_{i}.json')} text_len={len(t) if t and not t.startswith('ERROR') else 'ERR'} preview={(t or '')[:60]!r}")
PY

echo "[$(ts)] === repro end ==="
