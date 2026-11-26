#!/usr/bin/env python3
"""Route mlx-lm GDN decode (T == 1) to mx.fast.gated_delta_update_raw.

Idempotent follow-up to scripts/patch-mlx-lm-gdn.py for venvs whose
mlx-omarchy wheel ships the raw-gates decode fast path. The gate chain
(sigmoid/exp/softplus) rides into the fused kernel prologue.
Usage: python patch-mlx-lm-gdn-raw.py /path/to/venv"""
import glob
import sys

old = """    if hasattr(mx.fast, "gated_delta_update"):
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
        return out, st"""
new = """    if hasattr(mx.fast, "gated_delta_update"):
        if q.shape[1] == 1 and hasattr(mx.fast, "gated_delta_update_raw"):
            return mx.fast.gated_delta_update_raw(
                q, k, v, a, b, A_log, dt_bias, state, mask
            )
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
        return out, st"""

venv = sys.argv[1] if len(sys.argv) > 1 else "."
hits = glob.glob(
    venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models/gated_delta.py"
)
if not hits:
    sys.exit("mlx_lm/models/gated_delta.py not found under " + venv)
text = open(hits[0]).read()
if "gated_delta_update_raw" in text:
    print("already patched:", hits[0])
elif old in text:
    open(hits[0], "w").write(text.replace(old, new))
    print("patched:", hits[0])
else:
    sys.exit("unrecognized gated_delta.py content; run patch-mlx-lm-gdn.py first")
