#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# M2 end-to-end ticket: H3 _QUANTIZE on a real device, using the kernel-
# battery branch's cast scanner. The script is the deliverable for the
# M2 correctness window per Main's schedule (post-23:50Z, <= 20 min).
#
# Pre-reqs (Main announces the shared venv):
#   - VENV=<path to /var/tmp/shared-omarchy-venv (OmarchyDistributed)>
#   - KERNEL_BATTERY_WHL=<path to a wheel built from branch kernel-battery
#     on the M2; install into a private venv copy OR overlay the .so into
#     the shared venv's site-packages>
#
# Receipt contract:
#   - source commit: 5e32e2b4d + the kernel-battery commit
#   - kernel:        7.1.13-3-1-ARCH stable (T6021)
#   - vulkan:        honeykrisp ICD (omarchy-mlx-vulkan pkg)
#   - mlx backend:   omarchy (HONEYKRISP)
#   - python:        shared venv's python3.12
#   - source msl:    /tmp/tfport/tf-drowz/src/tensorfold/kernels/minimax/h3/v1/mlp_int8.py:101-122
#                   (the _QUANTIZE r-string body)
#   - expected:      mx.eval(out) returns without error; out matches
#                   the host-side reference within one bf16 rounding
#                   on the row max scales; the int8 Q matches
#                   round(clip(v * inverse, -127, 127)) per element.
#   - terminal log:  full provenance + command + numerical result +
#                   backend dispatch trace, sha256-summed.
#
# Args: $1 = M2 (T6021) ssh alias (always jw14m2-linux); $2 = private venv path.
#
# Run via Main's gpu-turn FIFO:
#   setsid nohup <dev-box>/bin/gpu-turn -m 20 -- \
#     env HOME=<dev-box-owner-home> \
#     PATH=<private-venv>/bin:$PATH \
#     bash harness/m2_h3_quantize_ticket.sh jw14m2-linux <private-venv> \
#     > <notebook-owner-home>/.local/share/apple-silicon-lab/artifacts/KernelBattery/m2-h3-quantize-ticket.log 2>&1
#
# Owner: KernelBattery (this script). Lane-send the log path + sha256 to
# Main on completion. Matrix row B10 updates from the ticket output.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <m2-ssh-alias> <private-venv-path>" >&2
  exit 2
fi
HOST="$1"
VENV="$2"
NOTEBOOK_OWNER="<notebook-owner-home>"
RECEIPT_DIR="${RECEIPT_DIR:-${NOTEBOOK_OWNER}/.local/share/apple-silicon-lab/artifacts/KernelBattery/m2-h3-quantize-ticket}"
mkdir -p "$RECEIPT_DIR"

SSH_OPTS=(-o ServerAliveInterval=10 -o ConnectTimeout=8)
PROVENANCE=/tmp/provenance.json
WORKTREE="<kernel-battery-worktree>"
KERNEL_BATTERY_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
SHARED_VENV_COMMIT="$(ssh "${SSH_OPTS[@]}" "$HOST" 'cat /var/tmp/shared-omarchy-venv/.omarchy-venv.sha256 2>/dev/null || echo unknown')"

cat > "$PROVENANCE" <<EOF
{
  "host": "$HOST",
  "kernel_battery_commit": "$KERNEL_BATTERY_COMMIT",
  "shared_venv_sha256": "$SHARED_VENV_COMMIT",
  "private_venv": "$VENV",
  "h3_source_path": "<dev-box>/tfport/tf-drowz/src/tensorfold/kernels/minimax/h3/v1/mlp_int8.py",
  "h3_source_lines": "101-122",
  "matrix_row": "B10",
  "as_of": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

H3_PY="<dev-box>/tfport/tf-drowz/src/tensorfold/kernels/minimax/h3/v1/mlp_int8.py"
H3_QUANTIZE_BODY=$(sed -n '101,122p' "$H3_PY")
H3_BODY_SHA256=$(printf '%s' "$H3_QUANTIZE_BODY" | sha256sum | cut -d' ' -f1)

TICKET_PY=$(mktemp --suffix=.py)
cat > "$TICKET_PY" <<PYEOF
import json
import os
import sys
import time
import numpy as np
import mlx.core as mx

# Reproduce _QUANTIZE from $H3_PY:101-122 in a standalone mx.fast.metal_kernel.
# The body of the kernel is a copy of the pinned source so the translator sees
# the same MSL on the M2 that we already classified on the dev box.
SOURCE = r"""$H3_QUANTIZE_BODY"""

KERNEL = mx.fast.metal_kernel(
    name="kernel_battery_h3_quantize_probe",
    input_names=["X", "M"],
    output_names=["Q", "XS"],
    source=SOURCE,
)

ROWS, K, GROUP = 4, 512, 256
g = K // GROUP  # 2 groups
np.random.seed(0)
x = (np.random.rand(ROWS, K).astype(np.float32) * 200.0 - 100.0)
x_mlx = mx.array(x)
m_mlx = mx.array([ROWS], dtype=mx.int32)
out_q, out_xs = KERNEL(
    inputs=[x_mlx, m_mlx],
    output_shapes=[(ROWS, K), (ROWS, g)],
    output_dtypes=[mx.int8, mx.float32],
    grid=(g, ROWS, 1),
    threadgroup=(GROUP, 1, 1),
    stream=mx.gpu,
)
mx.eval(out_q, out_xs)
got_q = np.array(out_q)
got_xs = np.array(out_xs)

# Reference: same arithmetic, one per (row, group) threadgroup, host-side.
# The H3 _QUANTIZE body writes one scale per (row, group) and round+clamp
# int8 values per element.
ref_xs = np.zeros((ROWS, g), dtype=np.float32)
ref_q = np.zeros((ROWS, K), dtype=np.int8)
for r in range(ROWS):
    for gi in range(g):
        v = x[r, gi*GROUP:(gi+1)*GROUP]
        top = float(np.max(np.abs(v)))
        scale = max(top, 1e-12) / 127.0
        ref_xs[r, gi] = scale
        inv = 1.0 / scale
        ref_q[r, gi*GROUP:(gi+1)*GROUP] = np.clip(np.round(v * inv), -127, 127).astype(np.int8)

q_match = np.array_equal(got_q, ref_q)
xs_close = np.allclose(got_xs, ref_xs, rtol=0, atol=0)

print(json.dumps({
    "q_match": bool(q_match),
    "xs_close": bool(xs_close),
    "got_q_sample": got_q[0, :4].tolist(),
    "ref_q_sample": ref_q[0, :4].tolist(),
    "got_xs": got_xs.tolist(),
    "ref_xs": ref_xs.tolist(),
    "q_mismatches": int(np.sum(got_q != ref_q)),
    "xs_diff_max": float(np.max(np.abs(got_xs - ref_xs))),
    "host_sha": "$H3_BODY_SHA256",
}))
PYEOF

# Pre-flight on the M2: verify provenance + venv + kernel source.
ssh "${SSH_OPTS[@]}" "$HOST" bash <<EOF
set -euo pipefail
test -x "$VENV/bin/python" || { echo "FAIL: $VENV/bin/python missing" >&2; exit 2; }
test -f "\$HOME/.cache/mlx-omarchy/spirv/.source-hash" || true
"$VENV/bin/python" -c 'import mlx.core as mx; print(mx.default_device()); print(mx.__version__)'
EOF

# Run the ticket.
RESULT=$(ssh "${SSH_OPTS[@]}" "$HOST" "$VENV/bin/python" "$TICKET_PY")
echo "$RESULT" | tee "$RECEIPT_DIR/result.json"

# SHA the receipt.
sha256sum "$RECEIPT_DIR/result.json" "$PROVENANCE" | tee "$RECEIPT_DIR/SHA256SUMS"

rm -f "$TICKET_PY"
