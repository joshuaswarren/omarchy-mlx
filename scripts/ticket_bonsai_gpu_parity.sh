#!/usr/bin/env bash
# BonsaiBuild ticket 2 (GPU, run inside a gpu-turn -m <=30 ticket): build the
# bundle-staged wheel, install it into a fresh venv, run both Bonsai parity
# suites with OMARCHY_BONSAI_GATE=1 (skips are failures), then check the
# dispatch trace: named Bonsai kernel ids hit, zero CPU-stream dispatches.
# Usage (on jw16): scripts/ticket_bonsai_gpu_parity.sh [tree-dir]
set -euo pipefail

TREE=${1:-/var/tmp/bonsai-gpu}
LOG="$TREE/ticket.log"
mkdir -p "$TREE"
exec > >(tee -a "$LOG") 2>&1

echo "== bonsai gpu parity ticket start $(date -u +%FT%TZ)"
echo "tree=$TREE"
echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
echo "kernel=$(uname -r)"
echo "uname=$(uname -a)"
nvidia-smi 2>/dev/null | head -1 || true
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion" | head -8 \
  || echo "vulkaninfo: not captured (recording ICD identity another way)"

if [ ! -d "$TREE/repo/.git" ]; then
  git clone https://github.com/joshuaswarren/omarchy-mlx.git "$TREE/repo"
fi
git -C "$TREE/repo" fetch origin
git -C "$TREE/repo" checkout --detach origin/main
COMMIT=$(git -C "$TREE/repo" rev-parse HEAD)
echo "commit=$COMMIT"
cd "$TREE/repo"

# Locate the staged parakeet-encoder-whole bundle. Golden status first, then
# the od-distributed copy. REFUSE without it: a wheel without the bundle
# silently falls back to the split-island path (build-wheel.sh enforces this
# too; locating it here keeps the error at ticket start, not minute 20).
BUNDLE=${MLX_OMARCHY_WHOLE_BUNDLE_DIR:-}
if [ -z "$BUNDLE" ] && [ -f /var/tmp/.golden-status/bundledir ]; then
  BUNDLE=$(cat /var/tmp/.golden-status/bundledir)
fi
if [ -z "$BUNDLE" ]; then
  BUNDLE=$(ls -d /var/tmp/od-distributed-wheel-*/.work/mlx/tools/mlx-omarchy-parakeet/share/mlx-omarchy/parakeet-1/bundles/parakeet-encoder-whole 2>/dev/null | head -1 || true)
fi
if [ -z "$BUNDLE" ] || [ ! -d "$BUNDLE" ]; then
  echo "ERROR: parakeet-encoder-whole bundle not found; set MLX_OMARCHY_WHOLE_BUNDLE_DIR" >&2
  exit 2
fi
echo "bundle=$BUNDLE"

export DEV_RELEASE=1
export MLX_OMARCHY_WHOLE_BUNDLE_DIR="$BUNDLE"
bash scripts/build-wheel.sh
WHEEL=$(ls "$TREE/repo/dist/"mlx_omarchy*.whl 2>/dev/null | head -1 || ls dist/mlx*.whl | head -1)
[ -n "$WHEEL" ] || { echo "ERROR: no wheel produced" >&2; exit 2; }
echo "wheel=$WHEEL"
sha256sum "$WHEEL"

# Fresh consumer venv: prove the wheel alone carries the bonsai ops.
python3 -m venv "$TREE/venv"
"$TREE/venv/bin/pip" install --quiet "$WHEEL" pytest numpy

export OMARCHY_BONSAI_GATE=1
"$TREE/venv/bin/python" -m pytest -rA \
  tests/test_bonsai_native.py tests/test_bonsai_lavapipe.py

# Dispatch-trace check under MLX_OMARCHY_TRACE_DISPATCH=1, two isolated
# subprocesses: (a) CPU-device control must emit ZERO dispatch lines;
# (b) one bonsai qmv call must emit lines whose kernel ids map (by
# compute.h declaration order) to Bonsai* names.
"$TREE/venv/bin/python" - <<'PYEOF'
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

echo "== bonsai gpu parity ticket PASS commit=$COMMIT $(date -u +%FT%TZ)"
