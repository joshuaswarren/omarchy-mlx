#!/usr/bin/env python3
"""Pass bf16 recurrent state into the GDN raw decode route from mlx-lm.

The backend's end-to-end bf16 recurrent state (omarchy #377) defaults the
GDN decode state to bfloat16 when the caller hands it no initial state.
The mlx-lm wrapper never does that: it allocates an fp32 zeros state, and
after prefill it passes the fp32 carry, so the backend's bf16 default
never engages and every decode step keeps the fp32 state round trip the
B1-B5 probe chain measured as the largest excess (1.84 ms/token on the
13-inch M1, 4.4x the byte floor).

This patcher adds the missing handoff on the raw decode route only: when
the step is a bf16 decode, the state is cast to bfloat16 unless
MLX_OMARCHY_GDN_STATE_F32=1 opts out. The backend keeps all tile
arithmetic in float32 and returns the state in its own dtype, so the
carry stays bf16 for the rest of the step chain. Prefill and the
composed T>1 path are untouched.

Applies to both vendored series: the 0.31.3 raw site (patches/mlx-lm-gated-delta-raw.patch)
and the 0.32 raw site (patches/mlx-lm-0.32/mlx-lm-gated-delta-raw.patch).
Apply after the raw-route patches. Requires the #377 backend (bf16 state
twins); on an older wheel the cast would push the step into the composed
fallback instead. Idempotent; refuses unrecognized content.
Usage: python3 patch-mlx-lm-gdn-state-bf16.py /path/to/venv
"""
import glob
import sys

OLD = """            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )"""
NEW = """            if (
                q.dtype == mx.bfloat16
                and state is not None
                and state.dtype != mx.bfloat16
                and os.environ.get("MLX_OMARCHY_GDN_STATE_F32", "0") != "1"
            ):
                # mlx-omarchy bf16 recurrent state handoff (#377): the backend
                # defaults the decode state to bfloat16 only when the caller
                # passes none; cast the wrapper's fp32 carry so the default
                # engages. MLX_OMARCHY_GDN_STATE_F32=1 keeps the fp32 state.
                state = state.astype(mx.bfloat16)
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )"""

IMPORT_OLD = "import glob\n"
IMPORT_NEW = "import glob\nimport os\n"

venv = sys.argv[1] if len(sys.argv) > 1 else "."
hits = glob.glob(
    venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models/gated_delta.py"
)
if not hits:
    sys.exit("mlx_lm/models/gated_delta.py not found under " + venv)
path = hits[0]
text = open(path).read()
if "MLX_OMARCHY_GDN_STATE_F32" in text:
    print("already patched:", path)
elif OLD in text:
    if "import os\n" not in text:
        if IMPORT_OLD not in text:
            sys.exit("no import anchor for os in " + path + "; refusing to patch")
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    open(path, "w").write(text.replace(OLD, NEW, 1))
    print("patched:", path)
else:
    sys.exit("unrecognized gated_delta.py raw site in " + path + "; apply the raw-route patch first")
