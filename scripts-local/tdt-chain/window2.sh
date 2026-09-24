#!/usr/bin/env bash
# TdtArch jwm1 window 2: device-chain validation + real-shape benches.
set -uo pipefail
echo "HOST: $(hostname)"
exec 8>/tmp/m1-gpu.lock
flock -w 120 8 || { echo "FATAL: lock not acquired"; exit 3; }
echo "lock acquired $(date -u +%FT%TZ)"

VENV=/var/tmp/v072-venv-fused
PY="$VENV/bin/python"
WIN=/var/tmp/tdtchain-win
BENCH_DIR="$WIN/scripts"

CACHE=$(ls -d ~/.cache/mlx-omarchy/parakeet-reference/mweinbach1/*/*/ 2>/dev/null | head -1)
[ -n "$CACHE" ] || { echo "FATAL: parakeet reference cache not found"; exit 4; }

echo "== B: chain vs host validation (golden encoder tensor) =="
"$PY" "$BENCH_DIR/validate_chain.py" \
  --cache-dir "$CACHE" --encoder "$BENCH_DIR/encoder_hidden.npy" \
  --overlay "$BENCH_DIR/overlay/tools" --reps 5 --chunk 200 \
  2>&1 | tee "$WIN/out/B2-validate.txt"
B_RC=${PIPESTATUS[0]}
echo "validate rc=$B_RC"

echo "== A2: real kernel-shape benches (bandwidth decision) =="
"$PY" "$BENCH_DIR/bench_tdt_floor.py" 2>&1 | grep -E "^(A/B|C |D |E |F |H |G )" \
  | tee "$WIN/out/A2-floor.txt"

echo "window phases done $(date -u +%FT%TZ)"
flock -u 8
echo "END: release jwm1"
