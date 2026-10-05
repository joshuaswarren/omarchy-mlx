#!/usr/bin/env bash
# api_parity.sh — oMLX API-surface parity harness: rows A1 A2 A3 A12 A13 A18 A31.
#
# Real-GPU receipt runs happen on M2/jwm1 inside a gpu-turn ticket. On the dev box
# use test_api_protocol.py instead (no mlx wheel on x86; those are supporting
# evidence only).
#
# Usage:
#   api_parity.sh --list
#   api_parity.sh [--start --venv /path/to/venv] [--base-url URL] [--model ID]
#                 [--embed-model ID] [--rerank-model ID] [--hf-cache PATH]
#                 [--out DIR] [--expect-context N] [section ...]
#
# Sections: a1 a2 a3 a12 a13 a18 a31 (default: all; a12 skips cleanly without models).
# Exit: 0 iff no FAIL. Every check prints PASS/FAIL/SKIP and leaves evidence files.
set -uo pipefail

BASE=http://127.0.0.1:8900
MODEL=mlx-community--Qwen3-4B-Instruct-2507-4bit
EMBED_MODEL=
RERANK_MODEL=
VENV=
START=0
HFCACHE=${HF_HUB_CACHE:-}
EXPECT_CONTEXT=
OUT=/tmp/omlx-api-parity
SECTIONS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) echo "a1: /v1/models + chat completions (stream+non-stream) + legacy completions"
            echo "a2: Anthropic /v1/messages (stream+non-stream) + count_tokens + adaptive thinking"
            echo "a3: stream include_usage + SSE keep-alive + real-context reporting"
            echo "a12: /v1/embeddings + /v1/rerank (needs --embed-model/--rerank-model)"
            echo "a13: model alias + profile create/apply/expose (admin API)"
            echo "a18: tool calling + json_schema structured output + MCP echo round-trip"
            echo "a31: web search test endpoint + chat search + /api/usage history"
            exit 0 ;;
    --base-url) BASE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    --rerank-model) RERANK_MODEL="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --start) START=1; shift ;;
    --hf-cache) HFCACHE="$2"; shift 2 ;;
    --expect-context) EXPECT_CONTEXT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) SECTIONS="$SECTIONS $1"; shift ;;
  esac
done
SECTIONS="${SECTIONS:- all}"

mkdir -p "$OUT" && cd "$OUT" || exit 1
exec > >(tee -a parity.log) 2>&1
ts() { date -u +%H:%M:%S.%3NZ; }
PASS=0; FAIL=0; SKIP=0
ok()   { PASS=$((PASS+1)); echo "[$(ts)] PASS: $1"; }
bad()  { FAIL=$((FAIL+1)); echo "[$(ts)] FAIL: $1"; }
skip() { SKIP=$((SKIP+1)); echo "[$(ts)] SKIP: $1 (${2:-no reason given})"; }

SERVER_PID=
cleanup() { [[ -n $SERVER_PID ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$(ts)] === api_parity start | base=$BASE model=$MODEL out=$OUT ==="

if (( START )); then
  [[ -n $VENV && -x $VENV/bin/omlx ]] || { echo "--start needs --venv with omlx installed"; exit 2; }
  export HOME="${HOME:-/tmp/omlx-home}"
  # MCP config for a18 (echo server lives next to this script)
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PYBIN="$VENV/bin/python"
  cat > mcp_parity.json <<EOF
{"servers": {"echo": {"transport": "stdio", "command": "$PYBIN",
  "args": ["$HERE/mcp_echo_server.py"], "enabled": true, "timeout": 30}}}
EOF
  echo "[$(ts)] starting server with --mcp-config (PID follows)"
  "$VENV/bin/omlx" serve --host 127.0.0.1 --port "${BASE##*:}" --model "$MODEL" \
    --mcp-config "$OUT/mcp_parity.json" > server.log 2>&1 &
  SERVER_PID=$!
  UP=0
  for i in $(seq 1 120); do
    curl -s --max-time 2 "$BASE/health" >/dev/null 2>&1 && { UP=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "SERVER DIED"; tail -30 server.log; exit 1; }
    sleep 1
  done
  (( UP )) || { echo "server never became healthy"; tail -30 server.log; exit 1; }
  echo "[$(ts)] server healthy (pid $SERVER_PID)"
fi

# ---- shared: models.json + health + provenance ----
curl -s --max-time 5 "$BASE/health" -o health.json || true
curl -s --max-time 5 "$BASE/v1/models" -o models.json || true
curl -s --max-time 5 "$BASE/api/status" -o status.json || true
python3 - <<'PY'
import json
try:
    d = json.load(open("models.json"))
    ids = [m.get("id") for m in d.get("data", [])]
    print("model ids:", ids)
    m = [x for x in d.get("data", []) if x.get("id")][0]
    print("first entry keys:", sorted(m.keys()))
except Exception as e:
    print("models.json unusable:", e)
PY
has_model() {  # $1 = id to find in models.json
  python3 - "$1" <<'PY'
import json, sys
want = sys.argv[1]
try:
    d = json.load(open("models.json"))
except Exception:
    print("no"); raise SystemExit
ids = [m.get("id", "") for m in d.get("data", [])]
print("yes" if want in ids else "no")
PY
}

# ============================== A1 ==============================
sec_a1() {
  echo "[$(ts)] --- A1: OpenAI-compatible server ---"
  grep -q '"object"' models.json 2>/dev/null && ok "A1 GET /v1/models 200 list" \
    || bad "A1 GET /v1/models 200 list"
  if [[ $(has_model "$MODEL") == yes ]]; then ok "A1 models lists $MODEL"; else bad "A1 models lists $MODEL"; fi

  curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name the three primary colors.\"}],\"max_tokens\":64,\"temperature\":0}" \
    -o a1_chat.json
  python3 - <<'PY'
import json
try:
    d = json.load(open("a1_chat.json"))
    txt = d["choices"][0]["message"]["content"]
    u = d.get("usage") or {}
    assert txt and len(txt) > 3, f"empty content: {txt!r}"
    assert u.get("completion_tokens", 0) > 0, f"usage missing: {u}"
    print("PASS-DETAIL a1 chat:", txt[:80].replace("\n", " "), "| usage:", u)
except Exception as e:
    print("FAIL-DETAIL a1 chat:", e); raise SystemExit(1)
PY
  [[ $? -eq 0 ]] && ok "A1 chat/completions non-stream real text + usage" || bad "A1 chat/completions non-stream real text + usage"

  curl -sN --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 20.\"}],\"max_tokens\":96,\"temperature\":0}" \
    -o a1_stream.sse
  python3 - <<'PY'
import json
chunks, text = 0, []
for line in open("a1_stream.sse"):
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    try:
        d = json.loads(line[6:])
    except Exception:
        continue
    if d.get("object") == "chat.completion.chunk":
        chunks += 1
        for ch in d.get("choices", []):
            delta = ch.get("delta", {}).get("content")
            if delta:
                text.append(delta)
assert chunks >= 3, f"only {chunks} chunks"
assert "".join(text).strip(), "no streamed content"
print("PASS-DETAIL a1 stream: chunks=%d text=%r" % (chunks, "".join(text)[:60]))
PY
  [[ $? -eq 0 ]] && ok "A1 chat/completions stream real SSE chunks" || bad "A1 chat/completions stream real SSE chunks"

  curl -s --max-time 90 "$BASE/v1/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"The three primary colors are\",\"max_tokens\":32,\"temperature\":0}" \
    -o a1_completion.json
  python3 - <<'PY'
import json
d = json.load(open("a1_completion.json"))
txt = d["choices"][0].get("text", "")
assert txt.strip(), f"empty completion text: {d}"
print("PASS-DETAIL a1 legacy completions:", repr(txt[:60]))
PY
  [[ $? -eq 0 ]] && ok "A1 /v1/completions legacy text" || bad "A1 /v1/completions legacy text"
}

# ============================== A2 ==============================
sec_a2() {
  echo "[$(ts)] --- A2: Anthropic Messages + adaptive thinking ---"
  curl -s --max-time 90 "$BASE/v1/messages" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":128,\"messages\":[{\"role\":\"user\",\"content\":\"Name the three primary colors.\"}]}" \
    -o a2_messages.json
  python3 - <<'PY'
import json
d = json.load(open("a2_messages.json"))
assert d.get("type") == "message" or "content" in d, f"not a message: {list(d)}"
blocks = d.get("content") or []
assert any(b.get("type") == "text" and b.get("text", "").strip() for b in blocks), f"no text block: {blocks}"
u = d.get("usage") or {}
assert isinstance(u.get("input_tokens"), int) and u["input_tokens"] > 0, f"input_tokens: {u}"
assert isinstance(u.get("output_tokens"), int) and u["output_tokens"] > 0, f"output_tokens: {u}"
assert d.get("stop_reason"), "no stop_reason"
print("PASS-DETAIL a2 messages:", blocks[0].get("text", "")[:60], "| usage:", u, "| stop:", d.get("stop_reason"))
PY
  [[ $? -eq 0 ]] && ok "A2 /v1/messages non-stream Anthropic shape + usage" || bad "A2 /v1/messages non-stream Anthropic shape + usage"

  curl -sN --max-time 120 "$BASE/v1/messages" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":96,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 10.\"}]}" \
    -o a2_stream.sse
  python3 - <<'PY'
import json
events, input_tok, output_tok = [], None, None
for line in open("a2_stream.sse"):
    line = line.strip()
    if line.startswith("event: "):
        events.append(line[7:])
    elif line.startswith("data: ") and line != "data: [DONE]":
        try:
            d = json.loads(line[6:])
        except Exception:
            continue
        if d.get("type") == "message_start":
            input_tok = (d.get("message", {}).get("usage") or {}).get("input_tokens")
        if d.get("type") == "message_delta":
            output_tok = (d.get("usage") or {}).get("output_tokens")
need = {"message_start", "content_block_delta", "message_delta", "message_stop"}
missing = need - set(events)
assert not missing, f"missing events: {missing}"
assert input_tok and input_tok > 0, f"message_start input_tokens={input_tok}"
assert output_tok and output_tok > 0, f"message_delta output_tokens={output_tok}"
print("PASS-DETAIL a2 stream: input=%s output=%s events=%d" % (input_tok, output_tok, len(events)))
PY
  [[ $? -eq 0 ]] && ok "A2 messages stream lifecycle + usage in start/delta" || bad "A2 messages stream lifecycle + usage in start/delta"

  curl -s --max-time 30 "$BASE/v1/messages/count_tokens" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello there\"}]}" -o a2_count.json
  python3 -c '
import json; d = json.load(open("a2_count.json"))
assert isinstance(d.get("input_tokens"), int), d
print("PASS-DETAIL a2 count_tokens:", d["input_tokens"])'
  [[ $? -eq 0 ]] && ok "A2 count_tokens integer" || bad "A2 count_tokens integer"

  curl -s --max-time 90 "$BASE/v1/messages" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"max_tokens\":512,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":256},\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23?\"}]}" \
    -o a2_thinking.json
  python3 - <<'PY'
import json
d = json.load(open("a2_thinking.json"))
blocks = d.get("content") or []
assert any(b.get("type") == "text" for b in blocks) or any(b.get("type") == "thinking" for b in blocks), f"bad blocks: {blocks}"
has = any(b.get("type") == "thinking" and b.get("thinking") for b in blocks)
print("PASS-DETAIL a2 thinking: thinking_block_present=%s types=%s" % (has, [b.get("type") for b in blocks]))
PY
  # Non-thinking model (Qwen3-4B-Instruct-2507): structure validity is the gate;
  # thinking-block presence is recorded, not required (pre-registered).
  [[ $? -eq 0 ]] && ok "A2 thinking request accepted, valid structure" || bad "A2 thinking request accepted, valid structure"
}

# ============================== A3 ==============================
sec_a3() {
  echo "[$(ts)] --- A3: streaming usage, SSE keep-alive, context reporting ---"
  curl -sN --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[{\"role\":\"user\",\"content\":\"Write ten words about the sea.\"}],\"max_tokens\":64,\"temperature\":0}" \
    -o a3_usage.sse
  python3 - <<'PY'
import json
last_usage, n = None, 0
for line in open("a3_usage.sse"):
    line = line.strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    try:
        d = json.loads(line[6:])
    except Exception:
        continue
    n += 1
    if d.get("usage"):
        last_usage = d["usage"]
assert last_usage, "no usage chunk with include_usage=true"
assert last_usage.get("prompt_tokens", 0) > 0 and last_usage.get("completion_tokens", 0) > 0, last_usage
print("PASS-DETAIL a3 include_usage after %d chunks: %s" % (n, last_usage))
PY
  [[ $? -eq 0 ]] && ok "A3 stream_options.include_usage final usage chunk" || bad "A3 stream_options.include_usage final usage chunk"

  # Keep-alive: long generation keeps the stream open; look for keep-alive
  # comments/pings before the last chunk. (server.py:2463/2491)
  curl -sN --max-time 180 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed 300-word story about a lighthouse keeper.\"}],\"max_tokens\":420,\"temperature\":0}" \
    -o a3_keepalive.sse
  python3 - <<'PY'
lines = open("a3_keepalive.sse").read().splitlines()
ka = [i for i, l in enumerate(lines) if l.startswith(": keep-alive")]
pings = [i for i, l in enumerate(lines) if l.startswith("event: ping")]
data = [i for i, l in enumerate(lines) if l.startswith("data: ") and l != "data: [DONE]"]
assert data, "no data chunks at all"
last_data = data[-1]
ka_before = [i for i in ka if i < last_data]
assert ka_before or pings, f"no keep-alive comments/pings (lines={len(lines)})"
print("PASS-DETAIL a3 keepalive: %d comments, %d pings before last data chunk" % (len(ka_before), len(pings)))
PY
  [[ $? -eq 0 ]] && ok "A3 SSE keep-alive observed in long stream" || bad "A3 SSE keep-alive observed in long stream"

  # Claude Code context reporting: /v1/models max_model_len must be the REAL
  # context window (README:202-205), compared with the checkpoint config.
  REAL=""; RC=0
  if [[ -n $EXPECT_CONTEXT ]]; then
    REAL=$EXPECT_CONTEXT
  else
    REAL=$(HFCACHE="$HFCACHE" WANTCFG="$MODEL" python3 - <<'PY'
import glob, json, os
pat = os.path.join((os.environ.get("HFCACHE") or ""), "*", "snapshots", "*", "config.json")
hits = []
for p in glob.glob(pat):
    try:
        cfg = json.load(open(p))
    except Exception:
        continue
    name = p.split("/snapshots/")[0].rsplit("/", 1)[-1]
    hits.append((name, cfg))
want = os.environ.get("WANTCFG", "").replace("--", "/").lower()
best = None
for name, cfg in hits:
    if want and want in name.lower():
        best = cfg
        break
if best is None and len(hits) == 1:
    best = hits[0][1]
if best is None:
    raise SystemExit(2)
print(best.get("max_position_embeddings"))
PY
)
    RC=$?
  fi
  if [[ $RC -eq 2 ]]; then
    skip "A3 real-context reporting" "checkpoint config.json not found; pass --expect-context N to enable"
  else
    MODELLEN=$(python3 - "$MODEL" <<'PY'
import json, sys
want = sys.argv[1]
d = json.load(open("models.json"))
for m in d.get("data", []):
    if m.get("id") == want:
        print(m.get("max_model_len") or "")
        break
PY
)
    if [[ -n $REAL && -n $MODELLEN && $MODELLEN == "$REAL" ]]; then
      ok "A3 max_model_len == real context window ($MODELLEN)"
    else
      bad "A3 max_model_len ($MODELLEN) == real context window ($REAL)"
    fi
  fi
}

# ============================== A12 ==============================
sec_a12() {
  echo "[$(ts)] --- A12: native embeddings + rerank ---"
  if [[ -z $EMBED_MODEL ]]; then skip "A12 embeddings" "no --embed-model (models not approved/cached yet)"
  else
    if [[ $(has_model "$EMBED_MODEL") != yes ]]; then
      skip "A12 embeddings" "$EMBED_MODEL not in /v1/models"
    else
      curl -s --max-time 60 "$BASE/v1/embeddings" -H 'content-type: application/json' \
        -d "{\"model\":\"$EMBED_MODEL\",\"input\":[\"The cat sat on the mat.\",\"A feline rested on the rug.\",\"Quantum computing uses qubits.\"]}" \
        -o a12_embed.json
      python3 - <<'PY'
import json, math
d = json.load(open("a12_embed.json"))
vecs = [x["embedding"] for x in d["data"]]
assert len(vecs) == 3 and all(len(v) > 16 for v in vecs), f"bad vectors: {[len(v) for v in vecs]}"
def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))
    return num / den
sim = cos(vecs[0], vecs[1]); dis = cos(vecs[0], vecs[2])
assert sim > dis, f"similarity ordering broken: sim={sim:.3f} dis={dis:.3f}"
print("PASS-DETAIL a12 embeddings: dim=%d cos(sim)=%.3f > cos(dis)=%.3f" % (len(vecs[0]), sim, dis))
PY
      [[ $? -eq 0 ]] && ok "A12 embeddings vectors + semantic ordering" || bad "A12 embeddings vectors + semantic ordering"
    fi
  fi
  if [[ -z $RERANK_MODEL ]]; then skip "A12 rerank" "no --rerank-model (models not approved/cached yet)"
  else
    if [[ $(has_model "$RERANK_MODEL") != yes ]]; then
      skip "A12 rerank" "$RERANK_MODEL not in /v1/models"
    else
      curl -s --max-time 90 "$BASE/v1/rerank" -H 'content-type: application/json' \
        -d "{\"model\":\"$RERANK_MODEL\",\"query\":\"What is the capital of France?\",\"documents\":[\"Paris is the capital of France.\",\"Berlin is the capital of Germany.\"],\"top_n\":2}" \
        -o a12_rerank.json
      python3 - <<'PY'
import json
d = json.load(open("a12_rerank.json"))
res = d.get("results") or d.get("data") or []
assert len(res) == 2, f"expected 2 results: {d}"
scores = [r["relevance_score"] for r in res]
assert scores[0] > scores[1], f"relevant doc not ranked first: {scores}"
assert res[0].get("index") == 0, f"top index should be the Paris doc: {res[0]}"
print("PASS-DETAIL a12 rerank: scores=%s top_index=%s" % (scores, res[0].get("index")))
PY
      [[ $? -eq 0 ]] && ok "A12 rerank ranks relevant doc first" || bad "A12 rerank ranks relevant doc first"
    fi
  fi
}

# ============================== A13 ==============================
sec_a13() {
  echo "[$(ts)] --- A13: model profiles, per-model settings, alias, type override ---"
  ALIAS=parity-alias-model
  code=$(curl -s -o a13_alias.json -w '%{http_code}' --max-time 15 -X PUT \
    "$BASE/api/models/$MODEL/settings" -H 'content-type: application/json' \
    -d "{\"model_alias\":\"$ALIAS\"}")
  if [[ $code == 401 ]]; then skip "A13 admin API" "401 — server has an API key; rerun loopback without auth"; return; fi
  [[ $code == 200 ]] && ok "A13 PUT settings model_alias 200" || bad "A13 PUT settings model_alias 200 (http $code)"

  curl -s --max-time 5 "$BASE/v1/models" -o a13_models.json
  if python3 -c "import json,sys; sys.exit(0 if '$ALIAS' in [m['id'] for m in json.load(open('a13_models.json'))['data']] else 1)"; then
    ok "A13 /v1/models advertises alias $ALIAS"
  else bad "A13 /v1/models advertises alias $ALIAS"; fi

  curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":16,\"temperature\":0}" \
    -o a13_alias_chat.json
  python3 -c '
import json; d = json.load(open("a13_alias_chat.json"))
assert d["choices"][0]["message"]["content"].strip(), d' \
    && ok "A13 chat request by alias serves" || bad "A13 chat request by alias serves"

  code=$(curl -s -o a13_profile.json -w '%{http_code}' --max-time 15 -X POST \
    "$BASE/api/models/$MODEL/profiles" -H 'content-type: application/json' \
    -d '{"name":"parity","display_name":"Parity","settings":{"temperature":0.5},"expose_as_model":true}')
  [[ $code == 200 ]] && ok "A13 create profile expose_as_model 200" || bad "A13 create profile expose_as_model 200 (http $code)"

  curl -s --max-time 5 "$BASE/v1/models" -o a13_models2.json
  PID=""
  python3 - <<'PY'
import json
d = json.load(open("a13_models2.json"))
ids = [m["id"] for m in d["data"]]
cands = [i for i in ids if i.endswith(":parity")]
open("a13_profile_id.txt", "w").write(cands[0] if cands else "")
print("profile ids in /v1/models:", cands)
PY
  PID=$(cat a13_profile_id.txt 2>/dev/null)
  [[ -n $PID ]] && ok "A13 profile exposed as model ($PID)" || bad "A13 profile exposed as model"

  if [[ -n $PID ]]; then
    curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$PID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":16,\"temperature\":0}" \
      -o a13_profile_chat.json
    python3 -c '
import json; d = json.load(open("a13_profile_chat.json"))
assert d["choices"][0]["message"]["content"].strip(), d' \
      && ok "A13 chat on <model>:<profile> serves on shared engine" || bad "A13 chat on <model>:<profile> serves on shared engine"
  fi

  code=$(curl -s -o a13_apply.json -w '%{http_code}' --max-time 15 -X POST \
    "$BASE/api/models/$MODEL/profiles/parity/apply")
  python3 -c '
import json, sys
d = json.load(open("a13_apply.json"))
s = (d.get("settings") or {})
sys.exit(0 if s.get("temperature") == 0.5 else 1)' \
    && ok "A13 profile apply returns overlaid settings (temperature=0.5)" \
    || bad "A13 profile apply returns overlaid settings (http $code)"

  # type override round-trip (setting visible, then reset)
  code=$(curl -s -o a13_type.json -w '%{http_code}' --max-time 15 -X PUT \
    "$BASE/api/models/$MODEL/settings" -H 'content-type: application/json' \
    -d '{"model_type_override":"llm"}')
  python3 -c '
import json, sys
d = json.load(open("a13_type.json"))
st = d.get("settings") or d
sys.exit(0 if (st.get("model_type_override") == "llm") else 1)' \
    && ok "A13 model_type_override accepted+persisted" \
    || bad "A13 model_type_override accepted+persisted (http $code)"

  # cleanup: restore defaults so repeated runs start clean
  curl -s --max-time 15 -X POST "$BASE/api/models/$MODEL/settings/reset" -o /dev/null
  curl -s --max-time 15 -X DELETE "$BASE/api/models/$MODEL/profiles/parity" -o /dev/null
  echo "[$(ts)] a13 cleanup: settings reset + profile deleted"
}

# ============================== A18 ==============================
sec_a18() {
  echo "[$(ts)] --- A18: tool calling, structured output, MCP ---"
  curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris right now? Use the provided tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}" \
    -o a18_tools.json
  python3 - <<'PY'
import json
d = json.load(open("a18_tools.json"))
msg = d["choices"][0]["message"]
tcs = msg.get("tool_calls") or []
assert tcs, f"no tool_calls: {msg}"
tc = tcs[0]["function"]
assert tc["name"] == "get_weather", tc
args = json.loads(tc["arguments"])
assert isinstance(args, dict) and "paris" in args.get("city", "").lower(), args
print("PASS-DETAIL a18 tools:", tc["name"], args)
PY
  [[ $? -eq 0 ]] && ok "A18 tool call parsed (name + JSON args)" || bad "A18 tool call parsed (name + JSON args)"

  curl -s --max-time 90 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Return a person named Ada aged 36.\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"person\",\"schema\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}},\"required\":[\"name\",\"age\"],\"additionalProperties\":false}}}}" \
    -o a18_schema.json
  python3 - <<'PY'
import json
d = json.load(open("a18_schema.json"))
txt = d["choices"][0]["message"]["content"].strip()
obj = json.loads(txt)
assert obj.get("name") == "Ada" and obj.get("age") == 36, obj
print("PASS-DETAIL a18 json_schema:", obj)
PY
  [[ $? -eq 0 ]] && ok "A18 grammar/structured output conforms to schema" || bad "A18 grammar/structured output conforms to schema"

  # MCP: only meaningful when this script started the server with --mcp-config
  if [[ $START == 1 ]]; then
    curl -s --max-time 120 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Call the echo tool with the exact text parity-ok and then reply with exactly what the tool returned.\"}]}" \
      -o a18_mcp.json
    python3 - <<'PY'
import json
d = json.load(open("a18_mcp.json"))
txt = d["choices"][0]["message"]["content"] or ""
assert "parity-ok" in txt, f"echo result missing from response: {txt[:200]!r}"
print("PASS-DETAIL a18 mcp echo:", txt[:120].replace(chr(10), " "))
PY
    [[ $? -eq 0 ]] && ok "A18 MCP echo tool round-trip via chat" || bad "A18 MCP echo tool round-trip via chat"
  else
    skip "A18 MCP round-trip" "server not started by this script; rerun with --start"
  fi
}

# ============================== A31 ==============================
sec_a31() {
  echo "[$(ts)] --- A31: web search tool + usage history ---"
  code=$(curl -s -o a31_searchtest.json -w '%{http_code}' --max-time 60 -X POST \
    "$BASE/api/web-search/test" -H 'content-type: application/json' \
    -d '{"provider":"ddgs","max_results":3}')
  python3 - <<'PY'
import json
d = json.load(open("a31_searchtest.json"))
ok_ = d.get("ok")
res = d.get("results") or []
print("PASS-DETAIL a31 web-search/test: ok=%s n=%d err=%s" % (ok_, len(res), (d.get("error") or {}).get("message", "")))
raise SystemExit(0 if (ok_ and res) else 1)
PY
  [[ $? -eq 0 ]] && ok "A31 server-side web search returns real results (http $code)" \
    || bad "A31 server-side web search returns real results (http $code, body in a31_searchtest.json)"

  curl -s --max-time 150 "$BASE/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Use the web_search tool with the query \\\"Albert Einstein year of birth\\\" and report the year the tool gives.\"}]}" \
    -o a31_chat.json
  python3 - <<'PY'
import json, re
d = json.load(open("a31_chat.json"))
txt = d["choices"][0]["message"]["content"] or ""
m = re.search(r"\b(18|19|20)\d{2}\b", txt)
assert m, f"no year in response: {txt[:200]!r}"
print("PASS-DETAIL a31 chat search: year=%s in %r" % (m.group(0), txt[:100]))
PY
  [[ $? -eq 0 ]] && ok "A31 chat triggers server-side search path" || bad "A31 chat triggers server-side search path (body in a31_chat.json)"

  curl -s --max-time 15 "$BASE/api/usage?range=today&include_details=true" -o a31_usage.json
  python3 - <<'PY'
import json
d = json.load(open("a31_usage.json"))
models = d.get("models") or []
if isinstance(models, dict):
    n = len(models)
else:
    n = len(models)
totals = d.get("totals")
assert n > 0, f"usage history has no model rows: {sorted(d)} -> {str(models)[:200]}"
print("PASS-DETAIL a31 usage history: model_rows=%d totals=%s" % (n, totals))
PY
  [[ $? -eq 0 ]] && ok "A31 usage history records today's requests" || bad "A31 usage history records today's requests"
}

# ============================== driver ==============================
for s in $SECTIONS; do
  case "$s" in
    all) sec_a1; sec_a2; sec_a3; sec_a12; sec_a13; sec_a18; sec_a31 ;;
    a1|a2|a3|a12|a13|a18|a31) "sec_$s" ;;
    *) echo "unknown section: $s"; exit 2 ;;
  esac
done

echo "[$(ts)] === SUMMARY: PASS=$PASS FAIL=$FAIL SKIP=$SKIP ==="
[[ $FAIL -eq 0 ]] || exit 1
