#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# M2 end-to-end ticket: H3 _QUANTIZE on a real device, using the kernel-
# battery branch's cast scanner + MPP header strip. The script is the
# deliverable for the M2 correctness window per Main's schedule
# (gpu-turn, <= 20 min, queued after PlatformGate).
#
# Faithful reproduction of the pinned H3 `_kernel("quantize", ...)` build:
#   source = "constexpr N/K/GROUP" prefix + _QUANTIZE r-string body
#   header = MPP include + `using namespace mpp::tensor_ops;`
#   X      = bfloat16 activations, mdims = int32 [rows, padded]
#   grid   = (K//GROUP * 256, padded, 1), threadgroup = (256, 1, 1)
#   out    = int8 (padded, K) + float32 (padded, K//GROUP)
# (source: <dev-box>/tfport/tf-drowz/src/tensorfold/kernels/minimax/h3/
#  v1/mlp_int8.py, quantize_rows() + _kernel() + _HEADER + _QUANTIZE.)
#
# Receipt contract: source commit, kernel + Vulkan identity, exact command,
# numerical result (q_match, xs_max_abs_diff), SHA256SUMS.
#
# Args: $1 = M2 ssh alias; $2 = private venv path (cp -a of the shared venv
# with the kernel-battery wheel force-installed).

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <m2-ssh-alias> <private-venv-path>" >&2
  exit 2
fi
HOST="$1"
VENV="$2"
NOTEBOOK_ARTIFACTS="${NOTEBOOK_ARTIFACTS:?set NOTEBOOK_ARTIFACTS to the private notebook artifacts dir}"
RECEIPT_DIR="$NOTEBOOK_ARTIFACTS/m2-h3-quantize-ticket"
mkdir -p "$RECEIPT_DIR"
WORKTREE="${WORKTREE:?set WORKTREE to the kernel-battery worktree path}"

SSH_OPTS=(-o ServerAliveInterval=10 -o ConnectTimeout=8)
PROVENANCE="$RECEIPT_DIR/provenance.json"
KB_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
SHARED_VENV_STAMP="$(ssh "${SSH_OPTS[@]}" "$HOST" \
  'ls /var/tmp/shared-omarchy-venv 2>/dev/null >/dev/null && echo present || echo absent')"

cat > "$PROVENANCE" <<EOF
{
  "host_alias": "$HOST",
  "kernel_battery_commit": "$KB_COMMIT",
  "shared_venv": "$SHARED_VENV_STAMP",
  "private_venv": "<m2-private-venv>",
  "upstream_source": "drowzeys TensorFold fork mlp_int8.py (quantize_rows/_kernel/_HEADER/_QUANTIZE)",
  "matrix_row": "B10",
  "as_of": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

TICKET_PY=$(mktemp --suffix=.py)
trap 'rm -f "$TICKET_PY"' EXIT
cat > "$TICKET_PY" <<'PYEOF'
import hashlib
import json
import platform
import sys

import numpy as np
import mlx.core as mx

# _HEADER from the pinned source (MPP marker include + using-directive).
HEADER = (
    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
    "using namespace mpp::tensor_ops;\n"
)

# _QUANTIZE r-string body, verbatim from the pinned source.
QUANTIZE_BODY = """
  constexpr int PER = GROUP / 256;
  constexpr int KG = K / GROUP;
  const int M = mdims[0];
  const int row = threadgroup_position_in_grid.y;
  const int g = threadgroup_position_in_grid.x;
  const int first = g * GROUP + thread_position_in_threadgroup.x * PER;
  float v[PER];
  float top = 0.0f;
  for (int j = 0; j < PER; j++) {
    v[j] = row < M ? float(X[(int64_t)row * K + first + j]) : 0.0f;
    top = max(top, abs(v[j]));
  }
  threadgroup float tops[8];
  top = simd_max(top);
  if (thread_index_in_simdgroup == 0) tops[simdgroup_index_in_threadgroup] = top;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  top = 1e-12f;
  for (int j = 0; j < 8; j++) top = max(top, tops[j]);
  const float scale = top / 127.0f;
  if (thread_position_in_threadgroup.x == 0) XS[row * KG + g] = scale;
  const float inverse = 1.0f / scale;
  for (int j = 0; j < PER; j++)
    Q[(int64_t)row * K + first + j] = (int8_t)clamp(int(rint(v[j] * inverse)), -127, 127);
"""

ROWS, K, GROUP = 4, 512, 256
N = 0  # upstream passes n=0 for the quantize kernel
PADDED = ROWS + (-ROWS % 128)  # TILE = 128; 4 stays 4
KG = K // GROUP

SOURCE = (
    f"  constexpr int N = {N};\n"
    f"  constexpr int K = {K};\n"
    f"  constexpr int GROUP = {GROUP};\n"
    + QUANTIZE_BODY
)

kernel = mx.fast.metal_kernel(
    name="kernel_battery_h3_quantize_probe",
    input_names=["X", "mdims"],
    output_names=["Q", "XS"],
    source=SOURCE,
    header=HEADER,
)

rng = np.random.default_rng(0)
x_host = (rng.random((ROWS, K)).astype(np.float32) * 200.0 - 100.0)
# Upstream quantize_rows casts X to bfloat16 BEFORE the kernel sees it.
x_bf16 = mx.array(x_host).astype(mx.bfloat16)
mdims = mx.array([ROWS, PADDED], dtype=mx.int32)

out_q, out_xs = kernel(
    inputs=[x_bf16, mdims],
    grid=(KG * 256, PADDED, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(PADDED, K), (PADDED, KG)],
    output_dtypes=[mx.int8, mx.float32],
    stream=mx.gpu,
)
mx.eval(out_q, out_xs)
got_q = np.array(out_q)
got_xs = np.array(out_xs)

# Host reference on the SAME bf16-rounded values the kernel read.
x_ref = np.array(x_bf16.astype(mx.float32))
ref_xs = np.zeros((PADDED, KG), dtype=np.float32)
ref_q = np.zeros((PADDED, K), dtype=np.int8)
for r in range(PADDED):
    for gi in range(KG):
        if r >= ROWS:
            v = np.zeros(GROUP, dtype=np.float32)  # padded rows read 0.0f
        else:
            v = x_ref[r, gi * GROUP:(gi + 1) * GROUP]
        top = max(float(np.max(np.abs(v))), 1e-12)
        scale = top / 127.0
        ref_xs[r, gi] = scale
        ref_q[r, gi * GROUP:(gi + 1) * GROUP] = np.clip(
            np.round(v * (1.0 / scale)), -127, 127).astype(np.int8)

q_match = bool(np.array_equal(got_q, ref_q))
xs_exact = bool(np.array_equal(got_xs, ref_xs))
xs_max_diff = float(np.max(np.abs(got_xs - ref_xs)))

print(json.dumps({
    "q_match": q_match,
    "xs_exact": xs_exact,
    "xs_max_abs_diff": xs_max_diff,
    "q_mismatches": int(np.sum(got_q != ref_q)),
    "got_q_sample": got_q[0, :4].tolist(),
    "ref_q_sample": ref_q[0, :4].tolist(),
    "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
    "mlx_version": mx.__version__,
    "default_device": str(mx.default_device()),
    "python": sys.version.split()[0],
    "platform": platform.machine(),
}, indent=2))
PYEOF

# Pre-flight: identity + version + device on the M2 via the private venv.
ssh "${SSH_OPTS[@]}" "$HOST" \
  "'$VENV/bin/python' -c 'import mlx.core as mx; print(mx.__version__); print(mx.default_device())'"

# The ticket itself.
ssh "${SSH_OPTS[@]}" "$HOST" "'$VENV/bin/python' '$TICKET_PY'" \
  | tee "$RECEIPT_DIR/result.json"

sha256sum "$RECEIPT_DIR/result.json" "$PROVENANCE" \
  | tee "$RECEIPT_DIR/SHA256SUMS"
