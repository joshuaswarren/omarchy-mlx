#!/usr/bin/env bash
# Frame-pacing proxy matrix for issue #19 (M2, one gpu-turn ticket).
#   case 1: probe alone (idle GPU)
#   case 2: probe + MLX decode, cap OFF   (MLX_OMARCHY_BATCH_WORK=0)
#   case 3: probe + MLX decode, cap ON    (wheel default 40000)
# The decode runs ~20 s in the background; the probe samples 12 s inside
# that window after a 5 s warmup. Prints one line per case.
set -u
PY=/tmp/sc-venvB/bin/python
MODEL=$(echo ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/*)
PROBE=/tmp/submit-latency

driver() {  # $1 = BATCH_WORK value
  $PY - "$MODEL" "$1" <<'EOF'
import sys, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
model, tok = load(sys.argv[1])
for _ in stream_generate(model, tok, prompt="hi", max_tokens=4):
    pass
print("decode-ready", flush=True)
for r in stream_generate(model, tok, prompt="Count from one to a thousand slowly, with details." * 3, max_tokens=512):
    pass
EOF
}

echo "=== case1 probe-alone ==="
$PROBE 12

echo "=== case2 decode-cap-OFF + probe ==="
MLX_OMARCHY_BATCH_WORK=0 driver 40000 > /tmp/sc-decode-off.log 2>&1 &
DPID=$!
# wait for decode-ready marker (model loaded, warmup done)
for i in $(seq 1 120); do grep -q decode-ready /tmp/sc-decode-off.log 2>/dev/null && break; sleep 0.5; done
sleep 2
$PROBE 12
wait $DPID
echo "decode-off tail: $(tail -c 120 /tmp/sc-decode-off.log | tr '\n' ' ')"

echo "=== case3 decode-cap-ON + probe ==="
driver x > /tmp/sc-decode-on.log 2>&1 &
DPID=$!
for i in $(seq 1 120); do grep -q decode-ready /tmp/sc-decode-on.log 2>/dev/null && break; sleep 0.5; done
sleep 2
$PROBE 12
wait $DPID
echo "decode-on tail: $(tail -c 120 /tmp/sc-decode-on.log | tr '\n' ' ')"
