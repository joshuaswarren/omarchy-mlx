#!/usr/bin/env bash
# Chat worker workload: warm-up + a streaming chat turn + a ~2k-token
# turn + a card turn.
set -uo pipefail
PORT=${1:?port required}
OUT=${2:?out dir required}
mkdir -p "$OUT"
URL="http://127.0.0.1:$PORT/v1/chat/completions"

# Warm-up (forces model materialization).
echo "=== warm-up ===" | tee "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-2b-4bit","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
  > "$OUT/warmup.json"
sleep 1

# Plain chat turn.
echo "=== plain chat ===" | tee -a "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-2b-4bit","messages":[{"role":"user","content":"Say hello in five words or fewer."}],"max_tokens":32,"stream":false}' \
  > "$OUT/plain.json"

# Long-context turn (~2k tokens; ~8k chars).
FILLER=$(printf 'The quick brown fox jumps over the lazy dog. %.0s' {1..30})
PROMPT=$(printf 'Summarize the following in one sentence:\n\n%s\n%.0s' "$FILLER" {1..100})
echo "=== long context (prompt chars=${#PROMPT}) ===" | tee -a "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys; print(json.dumps({'model':'qwen3.8-2b-4bit','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':256}))" "$PROMPT")" \
  > "$OUT/long.json"

# Card turn.
echo "=== card turn ===" | tee -a "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-2b-4bit","messages":[{"role":"user","content":"Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M."}],"max_tokens":256,"stream":false}' \
  > "$OUT/card.json"

echo "=== chat workload complete ===" | tee -a "$OUT/workload.log"