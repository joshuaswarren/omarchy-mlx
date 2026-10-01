#!/usr/bin/env bash
# TTS workload: synthesize 2 sentences.
set -uo pipefail
PORT=${1:?port required}
OUT=${2:?out dir required}
mkdir -p "$OUT"
URL="http://127.0.0.1:$PORT/v1/audio/speech"

# OpenAI-compatible TTS endpoint.
echo "=== TTS sentence 1 ===" | tee "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"Hello, this is a test sentence.","voice":"aiden"}' \
  > "$OUT/tts1.opus"
echo "wrote $OUT/tts1.opus ($(stat -c %s $OUT/tts1.opus 2>/dev/null) bytes)" | tee -a "$OUT/workload.log"

echo "=== TTS sentence 2 ===" | tee -a "$OUT/workload.log"
curl -s -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"The quick brown fox jumps over the lazy dog.","voice":"aiden"}' \
  > "$OUT/tts2.opus"
echo "wrote $OUT/tts2.opus ($(stat -c %s $OUT/tts2.opus 2>/dev/null) bytes)" | tee -a "$OUT/workload.log"

echo "=== TTS workload complete ===" | tee -a "$OUT/workload.log"