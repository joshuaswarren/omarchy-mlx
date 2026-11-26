#!/usr/bin/env bash
# Decision (Laya) workload: one /v1/decisions compare call.
set -uo pipefail
PORT=${1:?port required}
OUT=${2:?out dir required}
mkdir -p "$OUT"
URL="http://127.0.0.1:$PORT/v1/decisions"

# Decision compare (3 options, 2 criteria).
echo "=== /v1/decisions compare ===" | tee "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"questions":[{"id":"q1","question":"Pick the bitter option: espresso | tea | juice","options":[{"id":"a","label":"espresso"},{"id":"b","label":"tea"},{"id":"c","label":"juice"}],"criteria":"bitterness"}]}' \
  > "$OUT/decision.json"

# Second call to ensure a sustained decision path.
echo "=== /v1/decisions classification ===" | tee -a "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"questions":[{"id":"q1","question":"Score the urgency of these emails:","options":[{"id":"a","label":"Important"},{"id":"b","label":"Normal"},{"id":"c","label":"Low"}],"criteria":"i need to respond today"}]}' \
  > "$OUT/decision2.json"

echo "=== decision workload complete ===" | tee -a "$OUT/workload.log"