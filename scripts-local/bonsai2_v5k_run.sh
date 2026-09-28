#!/usr/bin/env bash
# Bonsai2 server v5-equivalent run with the per-call fix.
# Captures the bonsai2 managed-CLI token array (with the per-call
# fix that rebinds mlx_lm.generate to the wrap). The result is the
# reference for the 64 GiB-class recommended:true gate.
set -uo pipefail
ROOT=/var/tmp/agent-catalog-2026-09-21
VENV=$HOME/bonsai2-window-venv
PY=$VENV/bin/python
R=$ROOT/receipts/2026-09-21-managed-gates-v5-bonsai2
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
    log "EXIT trap rc=$rc"
    exit $rc
' EXIT INT TERM

log "RUNID serving-v5k-bonsai2-20260921T START $(date -Is)"
sudo -n systemctl stop llm-inference || { log "stop failed"; exit 1; }
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { log "lock busy"; exit 1; }
log "lock acquired"

# Same prompt + same temperature + same max_tokens as the direct-V5 attempt.
PROMPT_JSON='{"model":"bonsai-2-27b-mlx-2bit","messages":[{"role":"user","content":"Explain why the sky is blue."}],"max_tokens":128,"temperature":0,"stream":true}'
curl -s -N --max-time 300 -H "content-type: application/json" \
    -d "$PROMPT_JSON" \
    "http://127.0.0.1:8094/v1/chat/completions" > $R/pinned-stream.jsonl
sleep 3
grep "^PROBE:" $R/serve.log 2>/dev/null || true
grep "^PROBE:" /var/tmp/agent-catalog-2026-09-21/receipts/2026-09-21-managed-gates-v5-bonsai2/serve.log 2>/dev/null > $R/probe-events.jsonl || true
log "probe events captured: $(wc -l < $R/probe-events.jsonl)"
python3 - <<PYEOF
import json
e = None
for line in open("/var/tmp/agent-catalog-2026-09-21/receipts/2026-09-21-managed-gates-v5-bonsai2/probe-events.jsonl"):
    if "PROBE: " not in line: continue
    try: ev = json.loads(line.split("PROBE: ", 1)[1])
    except: continue
    if ev.get("event") == "generation" and ev.get("n") == 128:
        e = ev
        break
if e:
    print("n128 sha16:", e["ids_sha16"])
    with open("/tmp/bonsai2_v5k_ids.json", "w") as f:
        json.dump(e["ids"], f)
else:
    print("NO 128-token event")
PYEOF
pkill -TERM -f "_bonsai2.server" 2>/dev/null || true
sleep 8
flock -u 9
sudo -n systemctl start llm-inference
for i in $(seq 1 24); do
    sleep 5
    curl -sf --max-time 5 http://127.0.0.1:8002/health >/dev/null 2>&1 && break
done
log "8002 auth: $(curl --max-time 3 -s http://127.0.0.1:8002/health)"
log "RUNID DONE $(date -Is)"
