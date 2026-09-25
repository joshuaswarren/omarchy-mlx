#!/usr/bin/env bash
# Memory ownership accounting for the qwen3.8-27B-4bit managed route.
# Stops llm-inference, flocks inode 12, launches the managed CLI shim
# directly (not the catalog CLI) with PROBE env + pinned 60/128 streamed
# request, samples per-PID GPU + per-mapping smaps_rollup of the
# _mlxlm_server child before/after the pinned generation, and restores.
set -uo pipefail
ROOT=/var/tmp/agent-catalog-2026-09-21
VENV=$HOME/bonsai2-window-venv
PY=$VENV/bin/python
R=$ROOT/receipts/2026-09-21-memaccount
mkdir -p $R
cd $ROOT
export PYTHONPATH=$ROOT/serve
export MLX_OMARCHY_SERVE_IDS_PROBE=1
export MLX_OMARCHY_SERVE_CONTEXT_LIMIT=4096

ts() { date '+%H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a $R/gates.log; }
trap '
    rc=$?
    pkill -TERM -f "_mlxlm_server\|port 8091" 2>/dev/null
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

log "RUNID serving-v5h-memaccount-20260921T START $(date -Is)"
sudo -n systemctl stop llm-inference || { log "stop failed"; exit 1; }
sleep 2
exec 9>/tmp/m1-gpu.lock
flock -n 9 || { log "lock busy"; exit 1; }
log "lock acquired"

SNAP=$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818

# Launch the shim directly (same path the managed CLI uses)
$PY -m mlx_omarchy_serve._mlxlm_server --model $SNAP \
    --host 127.0.0.1 --port 8091 --max-tokens 512 \
    >$R/serve.log 2>&1 &
SRV=$!
ok=0
for i in $(seq 1 40); do
    sleep 5
    curl --max-time 3 -s http://127.0.0.1:8091/health 2>/dev/null | grep -q '"ok"' && { ok=1; log "ready ~$((i*5))s"; break; }
done
[ $ok -eq 0 ] && { log "NOT_READY"; kill -TERM $SRV; exit 1; }

# PID-by-port for the listening server process (the _mlxlm_server child)
LISTEN=$(ss -tlnp 2>/dev/null | grep ":8091 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
log "AT_READY listening_pid=$LISTEN"
log "AT_READY smaps (Rss/Pss): $(grep -E '^(Rss|Pss):' /proc/$LISTEN/smaps_rollup 2>/dev/null | tr '\n' ' ')"
# Per-mapping: count file-backed vs anonymous, find the top consumers
log "AT_READY mappings: total=$(wc -l < /proc/$LISTEN/smaps) anonymous=$(awk '$5=="anonymous" {n++} END {print n+0}' /proc/$LISTEN/smaps) file=$(awk '$5~/\.bin$|\.dylib$|mlx_|qwen|model/ {n++} END {print n+0}' /proc/$LISTEN/smaps)"
log "AT_READY gpu info: $(cat /proc/$LISTEN/status 2>/dev/null | grep -E 'VmRSS|VmSize|VmPeak|CapEff' | tr '\n' ' ')"
# mlx_lm server often spawns workers — list the process tree
log "AT_READY process tree: $(ps -o pid,ppid,comm --no-headers --ppid $LISTEN 2>/dev/null | tr '\n' '|' )"
log "AT_READY gpu allocations (Asahi allocator): $(find /proc/dri -name 'memory' -path '*asahi*' 2>/dev/null | head -3 | tr '\n' ' ' )"

# Warm round (capture warm first_event for the report)
printf '{"model":"%s","messages":[{"role":"user","content":"Warm: explain the sky."}],"max_tokens":32,"temperature":0,"stream":true}' "$SNAP" >/tmp/warm-body.json
$PY $ROOT/scripts-local/ttft_probe.py 127.0.0.1 8091 /tmp/warm-body.json >$R/warm-ttft.json 2>&1
log "warm: $(tr -d '\n ' <$R/warm-ttft.json | head -c 200)"

# Pinned 60/128 streamed request
printf '{"model":"%s","messages":[{"role":"user","content":"Explain why the sky is blue."}],"max_tokens":128,"temperature":0,"stream":true}' "$SNAP" >/tmp/pinned-body.json
log "starting pinned request"
$PY $ROOT/scripts-local/ttft_probe.py 127.0.0.1 8091 /tmp/pinned-body.json >$R/pinned-stream.json 2>&1
log "pinned: $(tr -d '\n ' <$R/pinned-stream.json | head -c 200)"

# POST-GENERATION memory sample (the moment the request finishes, the
# weights are resident + the KV cache is sized for the request).
log "POST_GEN smaps (Rss/Pss): $(grep -E '^(Rss|Pss):' /proc/$LISTEN/smaps_rollup 2>/dev/null | tr '\n' ' ')"
log "POST_GEN mappings: total=$(wc -l < /proc/$LISTEN/smaps) anonymous=$(awk '$5=="anonymous" {n++} END {print n+0}' /proc/$LISTEN/smaps) file=$(awk '$5~/\.bin$|\.dylib$|mlx_|qwen|model/ {n++} END {print n+0}' /proc/$LISTEN/smaps)"
log "POST_GEN gpu info: $(cat /proc/$LISTEN/status 2>/dev/null | grep -E 'VmRSS|VmSize|VmPeak|CapEff' | tr '\n' ' ')"

# Per-mapping RSS ranking (top 10) — the diagnostic for the unaccounted
# ~17GB MemAvailable delta. File-backed shared mappings (the safetensors)
# ARE counted in Rss but are shared with the file cache; private RSS for
# the server process should be the sum of private * anonymous + heap.
log "POST_GEN top mappings (RSS): $(awk '$5=="anonymous" || $5~/\.safetensors$|\.bin$/ {print $6}' /proc/$LISTEN/smaps | sort | uniq -c | sort -rn | head -10 | tr '\n' '|')"

# Probe events (if hook fires — for v5h the hook should fire because
# the shim is launched directly, not through the catalog CLI).
sleep 2
grep "^PROBE:" $R/serve.log > $R/probe-events.jsonl || true
log "probe events: $(wc -l < $R/probe-events.jsonl)"

kill -TERM $SRV 2>/dev/null
sleep 8
log "RUNID serving-v5h-memaccount-20260921T DONE $(date -Is)"
