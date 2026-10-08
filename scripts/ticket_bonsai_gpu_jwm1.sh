#!/usr/bin/env bash
# BonsaiBuild GPU ticket for the M1 build host (G13G chip). Reuses the wheel
# and test binaries that ticket_bonsai_cpu_build.sh + build-wheel.sh produced
# there; no rebuild, no bundle staging (already inside the wheel).
# Usage: ~/bin/gpu-turn -m 25 -- bash scripts/ticket_bonsai_gpu_jwm1.sh [tree-dir]
set -euo pipefail

TREE=${1:-$HOME/bonsai-build}
REPO="$TREE/repo"
WHEEL=$(ls -t "$REPO"/dist/mlx_omarchy-*.whl | head -1)
[ -n "$WHEEL" ] || { echo "ERROR: no wheel in $REPO/dist" >&2; exit 2; }
LOG="$TREE/gpu-ticket.log"
exec > >(tee -a "$LOG") 2>&1

echo "== bonsai gpu ticket (G13G) start $(date -u +%FT%TZ)"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "kernel=$(uname -r)"
uname -a
echo "mesa=$(pacman -Q mesa 2>/dev/null || echo unknown)"
glslc --version | head -1
echo "icds=$(ls /usr/share/vulkan/icd.d/ 2>/dev/null | tr '\n' ' ')"
echo "wheel=$WHEEL"
sha256sum "$WHEEL"

# 10 s RSS sampler (15 GiB box).
(
  while :; do
    echo "$(date -u +%T) used_mb=$(free -m | awk '/^Mem:/{print $3}')" >> "$TREE/gpu-rss.log"
    sleep 10
  done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# Fresh consumer venv from the wheel alone: the ops must come from the wheel.
rm -rf "$TREE/gpu-venv"
python3 -m venv "$TREE/gpu-venv"
"$TREE/gpu-venv/bin/pip" install --quiet "$WHEEL" pytest numpy

# The M1 build host has no packaged Honeykrisp ICD; point the backend at
# the coreglass build the 0.7.31 gates used (mesa-git-sha bbbfa36dce7).
# Without an override the ICD search finds only the stock asahi ICD, which
# is not Honeykrisp, and mx reports no Vulkan device.
export VK_DRIVER_FILES="$HOME/.local/share/coreglass/vulkan-4b-bbbfa36dce/honeykrisp_icd.aarch64.json"
[ -f "$VK_DRIVER_FILES" ] || { echo "ERROR: $VK_DRIVER_FILES missing" >&2; exit 2; }
export OMARCHY_BONSAI_GATE=1
cd "$REPO"
"$TREE/gpu-venv/bin/python" -m pytest -rA \
  tests/test_bonsai_native.py tests/test_bonsai_lavapipe.py

echo "== ctest suites (binaries from the cpu build) =="
"$REPO/.work/build/tests/omarchy/omarchy_primitive_tests"
"$REPO/.work/build/tests/omarchy/omarchy_runtime_tests"
"$REPO/.work/build/tests/omarchy/omarchy_matmul_family_tests"

# Dispatch-trace check under MLX_OMARCHY_TRACE_DISPATCH=1, two isolated
# subprocesses: (a) CPU-device control must emit ZERO dispatch lines;
# (b) one bonsai qmv call must emit lines whose kernel ids map (by
# compute.h declaration order) to Bonsai* names.
"$TREE/gpu-venv/bin/python" - <<'PYEOF'
import os
import re
import subprocess
import sys

header = open("overlay/mlx/backend/omarchy/compute.h").read()
body = re.search(r"enum class ComputeKernel : uint16_t \{(.*?)\n\};", header, re.S).group(1)
names = re.findall(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*,", body, re.M)
bonsai = {i: n for i, n in enumerate(names) if n.startswith("Bonsai")}
print("bonsai ordinals:", bonsai)

env = dict(os.environ, MLX_OMARCHY_TRACE_DISPATCH="1")

control = (
    "import mlx.core as mx\n"
    "import numpy as np\n"
    "a = mx.array(np.ones((4, 4), dtype=np.float32))\n"
    "mx.eval(mx.add(a, a, stream=mx.cpu))\n"
)
out = subprocess.run([sys.executable, "-c", control], env=env,
                     capture_output=True, text=True)
print(out.stdout, out.stderr)
lines = re.findall(r"\[rtmod\] DISPATCH kernel=(\d+)", out.stdout + out.stderr)
assert not lines, f"CPU-device control produced dispatch lines: {lines}"

bonsai_prog = (
    "import mlx.core as mx\n"
    "import numpy as np\n"
    "n, k, gs = 8, 256, 64\n"
    "codes = (rng := np.random.default_rng(0)).normal(0, 0.05, (n, k)) >= 0\n"
    "bits = 1 << np.arange(8)\n"
    "packed = mx.array((codes.reshape(n, k // 8, 8) * bits).sum(axis=2, dtype='uint8'))\n"
    "scales = mx.ones((n, k // gs))\n"
    "biases = mx.zeros((n, k // gs))\n"
    "x = mx.array(rng.normal(0, 1, (4, k)))\n"
    "mx.eval(mx.fast.bonsai_q1_affine_qmv(x, packed, scales, biases, group=gs))\n"
)
out = subprocess.run([sys.executable, "-c", bonsai_prog], env=env,
                     capture_output=True, text=True)
print(out.stdout, out.stderr)
ids = {int(i) for i in re.findall(r"\[rtmod\] DISPATCH kernel=(\d+)", out.stdout + out.stderr)}
hit = {i: n for i, n in bonsai.items() if i in ids}
print("bonsai ids hit:", hit)
assert hit, "no Bonsai kernel id in the dispatch histogram — the op did not route to a Bonsai kernel"
print("dispatch check PASS")
PYEOF

echo "peak_rss_mb=$(sort -t= -k2 -n "$TREE/gpu-rss.log" 2>/dev/null | tail -1)"
echo "== bonsai gpu ticket (G13G) PASS $(date -u +%FT%TZ)"
