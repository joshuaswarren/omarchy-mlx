#!/usr/bin/env bash
# TdtArch jwm1 measurement window (window 1 of the TDT parity ticket).
# Queue discipline: flock /tmp/m1-gpu.lock (fd 8), <=25 min, then release.
# Phases: A dispatch/streaming floor, B chain-vs-host validation,
# C same-window host + megakernel golden stage tables (installed wheel).
set -uo pipefail
echo "HOST: $(hostname)"
exec 8>/tmp/m1-gpu.lock
flock -w 120 8 || { echo "FATAL: lock not acquired"; exit 3; }
echo "lock acquired $(date -u +%FT%TZ)"

VENV=/var/tmp/v072-venv-fused
PY="$VENV/bin/python"
DRIVER=/var/tmp/pk-sess-driver.py
WIN=/var/tmp/tdtchain-win
BENCH_DIR="$WIN/scripts"
mkdir -p "$BENCH_DIR" "$WIN/out"

CACHE=$(ls -d ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/*/*/ 2>/dev/null | head -1)
[ -n "$CACHE" ] || { echo "FATAL: parakeet reference cache not found"; exit 4; }
echo "cache: $CACHE"

ENC="$BENCH_DIR/encoder_hidden.npy"
[ -f "$ENC" ] || { echo "FATAL: encoder_hidden.npy missing at $ENC"; exit 5; }
OVERLAY="$BENCH_DIR/overlay/tools"
[ -d "$OVERLAY/coreml" ] || { echo "FATAL: overlay missing"; exit 6; }

echo "== A: dispatch floor + streaming bandwidth =="
"$PY" "$BENCH_DIR/bench_tdt_floor.py" 2>&1 | tee "$WIN/out/A-floor.txt"

echo "== B: chain vs host validation =="
"$PY" "$BENCH_DIR/validate_chain.py" \
  --cache-dir "$CACHE" --encoder "$ENC" --overlay "$OVERLAY" \
  --reps 5 --chunk 64 2>&1 | tee "$WIN/out/B-validate.txt"
B_RC=${PIPESTATUS[0]}
echo "validate rc=$B_RC"

echo "== C: same-window golden stage tables (installed wheel, host path) =="
"$PY" "$DRIVER" --venv "$VENV" --out-root "$WIN/out/host" --runs 2 \
  --label host 2>&1 | tail -5 | tee "$WIN/out/C-host.txt"

echo "== C2: megakernel opt-in (MLX_OMARCHY_TDT_HOST=0) =="
MLX_OMARCHY_TDT_HOST=0 "$PY" "$DRIVER" --venv "$VENV" \
  --out-root "$WIN/out/loop" --runs 2 --label loop 2>&1 | tail -5 \
  | tee "$WIN/out/C2-loop.txt"

echo "window phases done $(date -u +%FT%TZ)"
flock -u 8
echo "END: release jwm1"
