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
# - private venv: cp -a of the read-only shared venv (never install into it),
#   THEN immediately rewrite bin/ shebangs to the private python — a plain
#   cp -a leaves pip's shebang pointing at the shared venv, so
#   <private>/bin/pip writes into the SHARED venv (2026-10-04 incident).
#   All installs go through "<private>/bin/python -m pip".
# - post-install verification: mx.__file__ must resolve under the private
#   dir, and the installed libmlx.so sha256 must match the wheel's copy.
# - shared venv integrity: refuses to run unless the shared venv still
#   carries the announced build (+b8af62c); never writes to it.
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

# Guard: the shared venv must be untouched (announced build +b8af62c).
SHARED_VERSION="$("$SHARED_VENV/bin/python" -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null || true)"
if [[ "$SHARED_VERSION" != *+b8af62c* ]]; then
  echo "shared venv mlx version is '$SHARED_VERSION', expected +b8af62c; \
it may have been mutated — aborting instead of building on a drifted baseline" >&2
  exit 2
fi

# Fresh private copy each ticket: a stale copy silently serves the old wheel.
rm -rf "$PRIVATE_VENV"
cp -a "$SHARED_VENV" "$PRIVATE_VENV"
# Rewrite bin/ shebangs to the private python so nothing here can write to
# the shared venv (see incident note above). -m venv --upgrade is the
# sanctioned fix; fall back to sed only if it is unavailable.
if ! "$PRIVATE_VENV/bin/python" -m venv --upgrade "$PRIVATE_VENV" >/dev/null 2>&1; then
  for script in "$PRIVATE_VENV"/bin/*; do
    [[ -f "$script" ]] && sed -i "1s|^#!.*shared-omarchy-venv.*|#!$PRIVATE_VENV/bin/python|" "$script"
  done
  sed -i "s|^home *= *.*|home = $PRIVATE_VENV/bin|" "$PRIVATE_VENV/pyvenv.cfg" 2>/dev/null || true
fi

"$PRIVATE_VENV/bin/python" -m pip install --no-deps --force-reinstall "$WHEEL" >/dev/null

# Verification: the import must resolve INSIDE the private venv and the
# installed libmlx.so must be byte-identical to the wheel's copy.
RESOLVED="$("$PRIVATE_VENV/bin/python" -c 'import mlx.core as mx; print(mx.__file__)')"
case "$RESOLVED" in
  "$PRIVATE_VENV"/*) ;;
  *) echo "mlx resolves outside the private venv: $RESOLVED" >&2; exit 2 ;;
esac
echo "mlx under private venv: $RESOLVED"
"$PRIVATE_VENV/bin/python" - <<PY
import hashlib, mlx.core as mx, pathlib, zipfile
wheel = pathlib.Path("$WHEEL")
inst = pathlib.Path(mx.__file__).parent / "lib" / "libmlx.so"
want = None
with zipfile.ZipFile(wheel) as z:
    for name in z.namelist():
        if name.endswith("mlx/lib/libmlx.so"):
            want = hashlib.sha256(z.read(name)).hexdigest()
            break
got = hashlib.sha256(inst.read_bytes()).hexdigest()
print("wheel libmlx.so sha256:", want)
print("installed  sha256:", got)
assert want is not None and got == want, "installed libmlx.so does not match the wheel"
PY

"$PRIVATE_VENV/bin/python" - <<'PY'
import mlx.core as mx
print("mlx:", mx.__version__)
print("default device:", mx.default_device())
PY

# The battery is lane-local (tests/test_minimax_m3_parity.py); run it from
# the repo checkout carried alongside the wheel in the ticket payload.
REPO_DIR="$(cd "$(dirname "$BATTERY")/.." && pwd)"
cd "$REPO_DIR"
MLX_OMARCHY_SPIRV_CACHE=/tmp/fmm3-spirv-cache \
  "$PRIVATE_VENV/bin/python" -m unittest discover -s tests \
  -p "test_minimax_m3_parity.py" -v 2>&1
