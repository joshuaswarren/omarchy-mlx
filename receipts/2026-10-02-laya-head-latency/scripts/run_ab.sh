#!/bin/bash
# LayaHead2 lever A/B: same-boot alternating arms, fresh server process per
# arm, NDJSON per call. Arms differ ONLY by env (or server code flag given as
# EXTRA_ARGS); the wheel stamp is captured per arm and asserted equal in
# analysis.
# usage: run_ab.sh OUT_DIR SERVE SUITE MODEL PORT ROUNDS N "arm1:ENV=V,ENV=V" "arm2:..." [EXTRA_SRV_ARGS...]
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; PORT=$5; ROUNDS=$6; N=$7
shift 7
ARMS=(); while [ $# -gt 0 ] && [[ "$1" != -* ]]; do ARMS+=("$1"); shift; done
EXTRA_SRV=("$@")

PY=${PY:-"$HOME/.local/share/mlx-omarchy/venv/bin/python"}
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

state() {
  {
    date -u +%FT%TZ
    echo "boot $(cat /proc/sys/kernel/random/boot_id)"
    echo "uptime $(cat /proc/uptime)"
    echo "load $(cat /proc/loadavg)"
    cat /proc/pressure/cpu
  } > "$OUT/host-$1.txt"
}

# Stale laya-mlx reservation from any earlier killed ticket.
RESV="$HOME/.local/share/mlx-omarchy/reservations.json"
if [ -f "$RESV" ]; then
  RESV="$RESV" PYTHONPATH="$SERVE" "$PY" - <<'PYEOF' 2>/dev/null || true
import json, os
from pathlib import Path
p = Path(os.environ["RESV"])
if p.exists():
    d = json.loads(p.read_text())
    if "laya-mlx" in d:
        del d["laya-mlx"]
        p.write_text(json.dumps(d))
PYEOF
fi

for round in $(seq 1 "$ROUNDS"); do
  for spec in "${ARMS[@]}"; do
    name=${spec%%:*}
    envs=${spec#*:}
    declare -a ENV_KV=()
    [ "$envs" != "$name" ] && IFS=',' read -ra ENV_KV <<< "$envs"
    tag="${name}-r${round}"
    state "$tag-before"
    env "${ENV_KV[@]}" PYTHONPATH="$SERVE" "$PY" -m mlx_omarchy_laya.server \
      --model "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype float16 \
      --max-questions 8 "${EXTRA_SRV[@]}" > "$OUT/server-$tag.log" 2>&1 &
    WPID=$!
    up=0
    for i in $(seq 1 60); do
      if curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then up=1; break; fi
      sleep 2
    done
    if [ "$up" != 1 ]; then
      echo "[ab] arm $tag server failed" >&2
      kill -KILL "$WPID" 2>/dev/null || true
      exit 4
    fi
    env "${ENV_KV[@]}" "$PY" -c 'import mlx.core as mx; print(mx.__version__)' \
      > "$OUT/wheel-$name.txt"
    PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/head_probe.py" http \
      --serve "$SERVE" --url "http://127.0.0.1:${PORT}/v1/decisions" \
      --suite "$SUITE" --out "$OUT/ab.ndjson" --n "$N" --warmup 5 \
      --arm "$tag" --server-pid "$WPID"
    if [ "$round" = 1 ] && [ -n "${SWEEP:-}" ]; then
      # Full-answer capture on the 154-case dev set: the same-boot
      # correctness reference for this arm (resumable).
      PYTHONPATH="$SERVE:$SCRIPTS" "$PY" "$SCRIPTS/dev_sweep_run.py" \
        --decision-url "http://127.0.0.1:${PORT}/v1/decisions" \
        --suite "$SUITE" --deadline-seconds 2.0 --out "$OUT/devraw-$name.json"
    fi
    kill -TERM "$WPID" 2>/dev/null || true
    sleep 1
    kill -KILL "$WPID" 2>/dev/null || true
    wait "$WPID" 2>/dev/null || true
    state "$tag-after"
  done
done
ps -eo pid,ppid,etime,cmd > "$OUT/ps-end.txt"
sync
echo "[ab] DONE"
