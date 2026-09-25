#!/usr/bin/env bash
# Managed-CLI Q4G64 v5-equivalent run on the 35B-distill local conversion.
# Stops llm-inference, flocks inode 12, launches the managed CLI serving
# $HOME/models/qwen3.8-35b-a3b-distill-q4-g64 as a local dir
# target (off-catalog local variant with --weights-gib), issues a pinned
# 60/128 streamed request, captures PROBE ids, and restores.
set -uo pipefail
ROOT=/var/tmp/agent-catalog-2026-09-21
VENV=$HOME/bonsai2-window-venv
PY=$VENV/bin/python
R=$ROOT/receipts/2026-09-21-35b-q4g64-v5
mkdir -p $R
cd $ROOT
export PYTHONPATH=$ROOT/serve:$ROOT/serve/mlx_omarchy_laya:$ROOT/serve/mlx_omarchy_bonsai2
export MLX_OMARCHY_SERVE_IDS_PROBE=1

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a $R/gates.log; }
trap '
    rc=$?
    pkill -TERM -f "_mlxlm_server\|port 8093" 2>/dev/null
    sleep 6
    flock -u 9 2>/dev/null || true
    sudo -n systemctl start llm-inference 2>&1 | tee -a $R/gates.log
    for i in $(seq 1 24); do
        curl -sf --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1 && break
        sleep 5
    done
    curl --max-time 20 -s -H "authorization: Bearer $(cat /etc/llm-inference/api-key)" -H "content-type: application/json" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly Paris.\"}],\"max_tokens\":8,\"temperature\":0}" http://127.0.0.1:8002/v1/chat/completions > $R/auth-completion.json
    grep -q "\"choices\"" $R/auth-completion.json && log "auth completion: VALID" || log "auth completion: MISSING"
    log "EXIT trap rc=$rc"
    exit $rc
' EXIT INT TERM

log "RUNID serving-v5i-q4g64-20260921T START $(date -Is)"
sudo -n systemctl stop llm-inference || { log "stop failed"; exit 1; }
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { log "lock busy"; exit 1; }
log "lock acquired"

TARGET=$HOME/models/qwen3.8-35b-a3b-distill-q4-g64
# Total safetensors size: 6 shards ~= 19.1 GB
WEIGHTS_GIB=19.0
$PY -m mlx_omarchy_serve serve $TARGET \
    --weights-gib $WEIGHTS_GIB \
    --context 4096 --server mlx-lm \
    --host 127.0.0.1 --port 8093 --yes \
    >$R/serve.log 2>&1 &
SRV=$!
log "35B Q4G64 managed CLI pid=$SRV"
ok=0
for i in $(seq 1 90); do
    sleep 5
    h=$(curl --max-time 3 -s "http://127.0.0.1:8093/health" 2>/dev/null || true)
    if echo "$h" | grep -q '"ok"'; then ok=1; log "ready ~$((i*5))s"; break; fi
done
[ $ok -eq 0 ] && { log "NOT_READY (90x5s)"; kill -TERM $SRV; exit 1; }

listen_pid=$(ss -tlnp 2>/dev/null | grep ":8093 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
log "AT_READY pid=$listen_pid smaps: $(grep -E '^(Rss|Pss):' /proc/$listen_pid/smaps_rollup 2>/dev/null | tr '\n' ' ')"

# Pinned 60/128 streamed request
printf '{"model":"%s","messages":[{"role":"user","content":"Explain why the sky is blue."}],"max_tokens":128,"temperature":0,"stream":true}' "$TARGET" >/tmp/35b-body.json
$PY $ROOT/scripts-local/ttft_probe.py 127.0.0.1 8093 /tmp/35b-body.json >$R/pinned-stream.json 2>&1
log "pinned: $(tr -d '\n ' <$R/pinned-stream.json | head -c 250)"

# Non-streamed pinned for ids capture (the PROBE hook needs the route)
curl -s --max-time 300 -H "content-type: application/json" \
    -d "{\"model\":\"$TARGET\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the sky is blue.\"}],\"max_tokens\":128,\"temperature\":0,\"stream\":false}" \
    "http://127.0.0.1:8093/v1/chat/completions" >$R/managed-128.json
log "non-stream pinned done"

# POST_GEN memory
log "POST_GEN: $(grep -E '^(Rss|Pss):' /proc/$listen_pid/smaps_rollup 2>/dev/null | tr '\n' ' ')"

# PROBE events (should capture if the hook works through the managed path)
sleep 2
grep "^PROBE:" $R/serve.log > $R/probe-events.jsonl || true
log "probe events: $(wc -l < $R/probe-events.jsonl)"

kill -TERM $SRV 2>/dev/null
sleep 8
log "RUNID DONE $(date -Is)"
