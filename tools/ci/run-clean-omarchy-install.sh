#!/usr/bin/env bash
# U5 clean-install gate: fresh venv, install the built wheel, import and
# compute smoke. tools/ci/run-clean-omarchy-install.sh
#
# Apple-silicon Honeykrisp is the supported target. On a development machine
# (no Apple GPU) the MLX_OMARCHY_ALLOW_NON_APPLE=1 override runs the same
# checks on a desktop or software Vulkan driver; receipts from such runs must
# record that the device is a development device, not Honeykrisp.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="$ROOT/dist"

if [[ -n "${MLX_OMARCHY_WHEEL:-}" ]]; then
  wheel="$MLX_OMARCHY_WHEEL"
  if [[ ! -f "$wheel" ]]; then
    echo "MLX_OMARCHY_WHEEL does not exist: $wheel" >&2
    exit 1
  fi
else
  shopt -s nullglob
  wheels=("$DIST_DIR"/mlx_omarchy-*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -eq 0 ]]; then
    echo "no mlx_omarchy wheel in $DIST_DIR; run scripts/build-wheel.sh first" >&2
    exit 1
  fi
  wheel="${wheels[0]}"
  for candidate in "${wheels[@]}"; do
    [[ "$candidate" -nt "$wheel" ]] && wheel="$candidate"
  done
fi
echo "[receipt] wheel: $(basename "$wheel")"

tmp="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp"
}
trap cleanup EXIT

python3 -m venv "$tmp/venv"
"$tmp/venv/bin/pip" --quiet install "$wheel"

"$tmp/venv/bin/python" - <<'EOF'
import json
import math
import pathlib
import subprocess

import mlx.core as mx

dev = mx.default_device()
assert dev.type == mx.DeviceType.gpu, f"default device is not gpu: {dev}"
label = "device name unavailable"
tool = pathlib.Path(mx.__file__).resolve().parent / "bin" / "mlx-omarchy-info"
if tool.exists():
    probe = subprocess.run([str(tool), "--json"], capture_output=True, text=True)
    if probe.returncode == 0:
        info = json.loads(probe.stdout)
        label = info.get("device_name", label)
        if not label.startswith("Apple"):
            label += " (development device, not Honeykrisp)"
print(f"[receipt] import mlx.core: ok, default device {dev}, {label}")

a = mx.array([1.0, 2.0])
b = mx.array([3.0, 4.0])
assert (a + b).tolist() == [4.0, 6.0]
print("[receipt] add: ok")

x = mx.array([[1.0, 2.0], [3.0, 4.0]])
y = mx.array([[5.0, 6.0], [7.0, 8.0]])
assert (x @ y).tolist() == [[19.0, 22.0], [43.0, 50.0]]
print("[receipt] matmul: ok")

px = [0.5, -0.25]
pw = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
s = [sum(px[i] * pw[i][j] for i in range(2)) for j in range(3)]
value_expect = sum(math.exp(v) for v in s)
grad_expect = [[px[i] * math.exp(v) for v in s] for i in range(2)]

xmx = mx.array(px)

def f(w):
    return mx.sum(mx.exp(xmx @ w))

value, grad = mx.value_and_grad(f)(mx.array(pw))
assert abs(value.item() - value_expect) < 1e-4
grad_list = grad.tolist()
for i in range(2):
    for j in range(3):
        assert abs(grad_list[i][j] - grad_expect[i][j]) < 1e-4
print("[receipt] value_and_grad: ok, gradient matches host expectation within 1e-4")

print("clean install verified")
EOF
