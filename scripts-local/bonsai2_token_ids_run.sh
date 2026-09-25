#!/usr/bin/env bash
# Managed-CLI bonsai2 token-array capture (runid per arbiter grant).
# Stops llm-inference, flocks inode 12, launches the bonsai2 managed CLI
# (mlx_omarchy_serve._bonsai2.server with --managed and PROBE env), issues
# the pinned 60/128 prompt streaming, parses the PROBE: events, restores.
set -uo pipefail
ROOT=/var/tmp/agent-catalog-2026-09-21
VENV=$HOME/bonsai2-window-venv
PY=$VENV/bin/python
R=$ROOT/receipts/2026-09-21-bonsai2-token-ids
mkdir -p $R
cd $ROOT
export PYTHONPATH=$ROOT/serve:$ROOT/serve/mlx_omarchy_laya:$ROOT/serve/mlx_omarchy_bonsai2
export MLX_OMARCHY_SERVE_IDS_PROBE=1

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a $R/gates.log; }
trap '
    rc=$?
    pkill -TERM -f "_bonsai2.server\|port 8094" 2>/dev/null
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

log "RUNID START $(date -Is)"
sudo -n systemctl stop llm-inference || { log "stop failed"; exit 1; }
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { log "lock busy"; exit 1; }
log "lock acquired"

SNAP=$HOME/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b415992eaa2ae9dd7b573706494d6bbf787
# Managed CLI launch (the same path the catalog serves)
$PY -m mlx_omarchy_serve serve bonsai-2-27b-mlx-2bit \
    --context 8192 --server module \
    --host 127.0.0.1 --port 8094 --yes \
    >$R/serve.log 2>&1 &
SRV=$!
log "bonsai2 managed CLI pid=$SRV"
ok=0
for i in $(seq 1 60); do
    sleep 5
    h=$(curl --max-time 3 -s "http://127.0.0.1:8094/health" 2>/dev/null || true)
    if echo "$h" | grep -q '"ok"'; then ok=1; log "ready ~$((i*5))s: $h"; break; fi
done
[ $ok -eq 0 ] && { log "NOT_READY"; kill -TERM $SRV; exit 1; }

# pids by port for memory accounting
listen_pid=$(ss -tlnp 2>/dev/null | grep ":8094 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
log "AT_READY: $(grep -E '^(Rss|Pss):' /proc/$listen_pid/smaps_rollup 2>/dev/null | tr '\n' ' ')"

# pinned 60/128 streamed request (mirrors the qwen v5f script for parity)
curl -s -N --max-time 300 -H "content-type: application/json" \
    -d "{\"model\":\"bonsai-2-27b-mlx-2bit\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the sky is blue.\"}],\"max_tokens\":128,\"temperature\":0,\"stream\":true}" \
    "http://127.0.0.1:8094/v1/chat/completions" >$R/pinned-stream.jsonl
log "pinned request done"

# post-gen memory
log "POST_GEN: $(grep -E '^(Rss|Pss):' /proc/$listen_pid/smaps_rollup 2>/dev/null | tr '\n' ' ')"

# parse PROBE: events (bonsai2 hook emits one per request)
sleep 2
grep "^PROBE:" $R/serve.log > $R/probe-events.jsonl || true
log "probe events: $(wc -l < $R/probe-events.jsonl)"

kill -TERM $SRV 2>/dev/null
sleep 8
log "RUNID DONE $(date -Is)"
