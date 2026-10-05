#!/usr/bin/env bash
# FamMinimaxM3 D2 correctness battery on the M2 (T6021 Honeykrisp).
#
# Runs INSIDE a gpu-turn ticket (<= 20 min): installs the lane wheel into a
# private copy of the shared venv, then runs the D2 parity battery
# (K1_SCALAR / K2 / TOPK_SELECT against CPU references; K1_SIMD,
# K1_SIMD_PACKED, DECODE_B1_SIMD share the same translator subset).
#
# Usage (from the M2):
#   bash harness/m2_minimax_m3_ticket.sh /path/to/mlx_omarchy-*.whl
#
# Contract:
# - private venv: cp -a of the read-only shared venv (never install into it)
# - install: --no-deps --force-reinstall (shared venv's deps are already right)
# - no timing claims: the battery asserts correctness only; wall-clock lines
#   printed to stderr are informational for the lane receipt, not ledger data
# - any failure exits nonzero so the ticket log shows the failing kernel by name
set -euo pipefail

WHEEL="${1:?usage: m2_minimax_m3_ticket.sh <mlx_omarchy wheel>}"
SHARED_VENV="${MLX_OMARCHY_SHARED_VENV:-/var/tmp/shared-omarchy-venv}"
PRIVATE_VENV="${MLX_OMARCHY_PRIVATE_VENV:-/var/tmp/fmm3-venv}"
BATTERY="${MLX_OMARCHY_D2_BATTERY:-tests/test_minimax_m3_parity.py}"

if [[ ! -f "$WHEEL" ]]; then
  echo "wheel not found: $WHEEL" >&2
  exit 2
fi
if [[ ! -d "$SHARED_VENV" ]]; then
  echo "shared venv missing: $SHARED_VENV" >&2
  exit 2
fi

# Fresh private copy each ticket: a stale copy silently serves the old wheel.
rm -rf "$PRIVATE_VENV"
cp -a "$SHARED_VENV" "$PRIVATE_VENV"
"$PRIVATE_VENV/bin/pip" install --no-deps --force-reinstall "$WHEEL" >/dev/null

"$PRIVATE_VENV/bin/python" - <<'PY'
import mlx.core as mx
print("mlx:", mx.__version__)
print("gpu backend alive:", mx.device(mx.gpu))
PY

# The battery is lane-local (tests/test_minimax_m3_parity.py); run it with
# the repo checkout mounted from the wheel build host via scp (the ticket
# payload carries tests/ alongside the wheel).
REPO_DIR="$(cd "$(dirname "$BATTERY")/.." && pwd)"
cd "$REPO_DIR"
MLX_OMARCHY_SPIRV_CACHE=/tmp/fmm3-spirv-cache \
  "$PRIVATE_VENV/bin/python" -m unittest discover -s tests \
  -p "test_minimax_m3_parity.py" -v 2>&1
