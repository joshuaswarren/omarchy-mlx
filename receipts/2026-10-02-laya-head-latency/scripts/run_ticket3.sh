#!/bin/bash
# LayaHead2 ticket 3: submission-cap A/B on the SubmitCap wheel in a PRIVATE
# venv (shared venv untouched). Arms differ ONLY by MLX_OMARCHY_BATCH_WORK on
# the same wheel stamp; the shared-venv gate wheel (cap absent) is the
# external baseline from ticket 1. Dev-set sweeps per arm (round 1) give the
# same-boot correctness references.
# usage: run_ticket3.sh OUT_DIR SERVE SUITE MODEL WHEEL PORT
set -euo pipefail

OUT=$1; SERVE=$2; SUITE=$3; MODEL=$4; WHEEL=$5; PORT=${6:-50518}
SHARED_SP=$HOME/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages
VENV=$HOME/scratch/laya2/venv-cap
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
state start

# --- private venv (idempotent) ------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
VPY="$VENV/bin/python"
if ! "$VPY" -c 'import mlx' 2>/dev/null; then
  "$VPY" -m pip -q install --no-index --no-deps "$WHEEL"
fi
# numpy + tokenizers copied from the shared venv into the PRIVATE venv
# (read-only source; each copied only when missing, so re-runs never
# overwrite).
for pkg in numpy tokenizers; do
  if ! "$VPY" -c "import $pkg" 2>/dev/null; then
    cp -r "$SHARED_SP"/${pkg}* "$VENV/lib/python3.14/site-packages/"
  fi
done
"$VPY" -c 'from tokenizers import Tokenizer' 2>/dev/null || {
  echo "tokenizers missing in private venv" >&2; exit 6; }
"$VPY" -c 'import mlx.core as mx, numpy; print(mx.__version__)' > "$OUT/wheel-cap.txt" 2>&1
grep -q '4b2929a' "$OUT/wheel-cap.txt" || { echo "wrong cap wheel stamp:" >&2; cat "$OUT/wheel-cap.txt" >&2; exit 5; }
state venv-ready

# --- in-process probe on the cap wheel (default env) ---------------------------
PYTHONPATH="$SERVE:$SCRIPTS" "$VPY" "$SCRIPTS/head_probe.py" phases \
  --serve "$SERVE" --model "$MODEL" --suite "$SUITE" --out "$OUT/phases-capwheel.ndjson" \
  --n 10 --warmup 3 --arm capwheel-eager

# --- A/B: same wheel, env-only arms, alternating, with dev sweeps --------------
export PY="$VPY"
export SWEEP=1
"$SCRIPTS/run_ab.sh" "$OUT" "$SERVE" "$SUITE" "$MODEL" "$PORT" 2 25 \
  "cap0:MLX_OMARCHY_BATCH_WORK=0" \
  "cap500:MLX_OMARCHY_BATCH_WORK=500" \
  "cap40000:MLX_OMARCHY_BATCH_WORK=40000"

state end
sync
echo "[ticket3] DONE"
