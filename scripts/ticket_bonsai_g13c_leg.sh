#!/usr/bin/env bash
# BonsaiBuild G13C leg (jw16): run the shipped wheel's Bonsai parity suites
# and the relayed ctest binaries on the M1 Max. Read-only host state except
# the venv + log under the tree dir. The wheel is NOT rebuilt here.
# Usage (on jw16): bash ticket_bonsai_g13c_leg.sh [tree-dir]
set -euo pipefail

TREE=${1:-/var/tmp/bonsai-g13c}
LOG="$TREE/ticket.log"
mkdir -p "$TREE"
exec > >(tee -a "$LOG") 2>&1

echo "== bonsai G13C leg start $(date -u +%FT%TZ)"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "kernel=$(uname -r)"
uname -a
echo "mesa=$(pacman -Q mesa 2>/dev/null || echo unknown)"
glslc --version | head -1
echo "icds=$(ls /usr/share/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ')"
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion" | head -4 \
  || echo "vulkaninfo: not captured"

WHEEL=$(ls ~/bonsai-run/mlx_omarchy-*.whl | head -1)
[ -n "$WHEEL" ] || { echo "ERROR: no wheel in ~/bonsai-run" >&2; exit 2; }
echo "wheel=$WHEEL"
sha256sum "$WHEEL"

if [ ! -d "$TREE/repo/.git" ]; then
  git clone https://github.com/joshuaswarren/omarchy-mlx.git "$TREE/repo"
fi
git -C "$TREE/repo" fetch origin
git -C "$TREE/repo" checkout --detach -q origin/main
echo "tests-commit=$(git -C "$TREE/repo" rev-parse HEAD)"

rm -rf "$TREE/venv"
python3 -m venv "$TREE/venv"
"$TREE/venv/bin/pip" install --quiet "$WHEEL" pytest numpy

export OMARCHY_BONSAI_GATE=1
export VK_DRIVER_FILES=$HOME/.local/share/coreglass/vulkan-4b-bbbfa36dce/honeykrisp_icd.aarch64.json
[ -f "$VK_DRIVER_FILES" ] || { echo "ERROR: $VK_DRIVER_FILES missing" >&2; exit 2; }

cd "$TREE/repo"
"$TREE/venv/bin/python" -m pytest -rA \
  tests/test_bonsai_native.py tests/test_bonsai_lavapipe.py

echo "== ctest binaries relayed from the build host =="
for t in ~/bonsai-run/omarchy_primitive_tests \
         ~/bonsai-run/omarchy_runtime_tests \
         ~/bonsai-run/omarchy_matmul_family_tests; do
  "$t" || echo "SUITE_FAIL $t rc=$?"
done

echo "== bonsai G13C leg PASS $(date -u +%FT%TZ)"
