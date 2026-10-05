#!/usr/bin/env bash
# FamQwen35Prefill M2 timing ledger — pf512/pf1024, flash-256 arm ON vs
# composed fallback. Runs AFTER 03:00Z in a gpu-turn ticket per Main's
# FIFO. h253 cell shape (M2Lane 2026-10-04T134800Z convention):
#   pf512  = --new-tokens 32 --prefill-tokens 512  --limit 10 --passes 3 --warmup 2
#   pf1024 = same with --prefill-tokens 1024
# Quiet gates before each cell: load1 < 0.5 and psi cpu avg10 = 0.
# One model load per arm block; digests recorded per cell; the ON and
# OFF arms must produce identical greedy digests (numerics gate 2 at
# the serving level) or the mismatch is recorded, not smoothed.
set -uo pipefail
LANE=FamQwen35Prefill
VENV=${FAMQWEN_VENV:-/var/tmp/famqwen35prefill-venv}
BENCH=$HOME/bench-scripts
WORK=$HOME/famqwen-timing
LOG=$WORK/timing.log
MODELS=(
  "mlx-community/Qwen3.5-2B-MLX-4bit"
  "mlx-community/Qwen3.5-9B-MLX-4bit"
)
mkdir -p "$WORK"

log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

log "boot $(cat /proc/sys/kernel/random/boot_id | cut -c1-8) load $(cut -d' ' -f1 /proc/loadavg)"

quiet_wait() {
  for try in $(seq 1 12); do
    L=$(cut -d' ' -f1 /proc/loadavg)
    P=$(cat /proc/pressure/cpu 2>/dev/null | awk '/avg10/ {print $2}' | tr -d ',')
    if awk -v l="$L" 'BEGIN{exit !(l < 0.5)}' && [ "${P:-0.00}" = "0.00" ]; then
      return 0
    fi
    sleep 5
  done
  log "NOTQUIET load=$L psi=$P — running anyway (recorded)"
}

cell() { # model env_setter cell_name pf
  local model="$1" envset="$2" cell="$3" pf="$4"
  quiet_wait
  log "CELL $cell start ($model, env: ${envset:-default})"
  env $envset "$VENV/bin/python" "$BENCH/qwen38-mlx-bench.py" \
    --model "$model" \
    --new-tokens 32 --prefill-tokens "$pf" \
    --limit 10 --passes 3 --warmup 2 \
    --json-out "$WORK/$cell.json" >>"$LOG" 2>&1
  local rc=$?
  log "CELL $cell rc=$rc $( [ -f "$WORK/$cell.json" ] && python3 -c "import json;d=json.load(open('$WORK/$cell.json'));print('prefill_tok_s',d.get('prefill_tok_s') or d.get('pure_prefill_tok_s'),'digest',d.get('ordered_records_sha256','')[:16])" 2>/dev/null )"
}

for model in "${MODELS[@]}"; do
  tag=$(basename "$model" | tr './' '__')
  cell "$model" ""                          "${tag}_pf512_on"  512
  cell "$model" "MLX_OMARCHY_SDPA_PREFILL_FLASH256=0" "${tag}_pf512_off" 512
  cell "$model" ""                          "${tag}_pf1024_on" 1024
  cell "$model" "MLX_OMARCHY_SDPA_PREFILL_FLASH256=0" "${tag}_pf1024_off" 1024
done

log "=== timing cells complete; JSONs in $WORK"
python3 - "$WORK" <<'PY'
import json, sys, glob, hashlib, os
work = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(work, "*.json"))):
    d = json.load(open(p))
    rows.append((os.path.basename(p), d.get("prefill_tok_s") or d.get("pure_prefill_tok_s"),
                 (d.get("ordered_records_sha256") or "")[:16]))
for r in rows:
    print(r)
PY
