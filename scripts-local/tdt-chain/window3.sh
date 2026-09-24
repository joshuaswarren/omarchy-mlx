#!/bin/bash
# TdtArch jwm1 window 3: candidate wheel A/B — device-chain (default) vs
# host control (MLX_OMARCHY_TDT_CHAIN=0), interleaved arms in one window,
# 10 warm in-process reps per arm plus interleaved fresh-process runs.
# Golden gate every run: status match, 104 emissions, transcript db501a8c.
set -uo pipefail
echo "HOST: $(hostname)"
exec 8>/tmp/m1-gpu.lock
flock -w 600 8 || { echo "FATAL: lock not acquired"; exit 3; }
echo "lock acquired $(date -u +%FT%TZ)"

VENV=/var/tmp/tdtchain-venv
PY="$VENV/bin/python"
DRIVER=/var/tmp/pk-sess-driver.py
WIN=/var/tmp/tdtchain-win

[ -x "$PY" ] || { echo "FATAL: candidate venv missing at $VENV"; exit 4; }

echo "== warm blocks: 10 in-process reps per arm =="
"$PY" "$DRIVER" --venv "$VENV" --out-root "$WIN/out/w3-chain" --runs 10 \
  --label chain 2>&1 | grep '"label"' | tee "$WIN/out/w3-chain-summary.txt"
MLX_OMARCHY_TDT_CHAIN=0 "$PY" "$DRIVER" --venv "$VENV" \
  --out-root "$WIN/out/w3-host" --runs 10 --label host \
  2>&1 | grep '"label"' | tee "$WIN/out/w3-host-summary.txt"

echo "== interleaved fresh-process runs (3 per arm) =="
for i in 1 2 3; do
  "$PY" "$DRIVER" --venv "$VENV" --out-root "$WIN/out/w3-chain-fresh-$i" \
    --runs 1 --label chain-fresh-$i 2>&1 | grep '"label"' \
    | tee -a "$WIN/out/w3-fresh.txt"
  MLX_OMARCHY_TDT_CHAIN=0 "$PY" "$DRIVER" --venv "$VENV" \
    --out-root "$WIN/out/w3-host-fresh-$i" --runs 1 --label host-fresh-$i \
    2>&1 | grep '"label"' | tee -a "$WIN/out/w3-fresh.txt"
done

echo "window phases done $(date -u +%FT%TZ)"
flock -u 8
echo "END: release jwm1"
