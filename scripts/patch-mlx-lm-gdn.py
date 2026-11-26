#!/usr/bin/env python3
"""Route mlx-lm gated-delta updates to mx.fast.gated_delta_update.

Idempotent patch for mlx-lm 0.31.3 venvs (see patches/mlx-lm-gated-delta-fast-route.patch).
Usage: python scripts/patch-mlx-lm-gdn.py /path/to/venv
"""
import glob
import sys

new = """    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    if hasattr(mx.fast, "gated_delta_update"):
        out, st = mx.fast.gated_delta_update(q, k, v, g, beta, state, mask)
        return out, st
    return gated_delta_kernel(q, k, v, g, beta, state, mask)"""
old = """    if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():
        return gated_delta_ops(q, k, v, g, beta, state, mask)
    return gated_delta_kernel(q, k, v, g, beta, state, mask)"""

# Some venvs carry an explanatory comment between the if-line and the
# ops return; accept any text between them.
import re
old_re = re.compile(
    r"    if not use_kernel or mx\.default_device\(\) != mx\.gpu"
    r" or not mx\.metal\.is_available\(\):\n"
    r"(?:\s*#[^\n]*\n)*"
    r"\s+return gated_delta_ops\(q, k, v, g, beta, state, mask\)\n"
    r"    return gated_delta_kernel\(q, k, v, g, beta, state, mask\)")

venv = sys.argv[1] if len(sys.argv) > 1 else "."
hits = glob.glob(
    venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models/gated_delta.py"
)
if not hits:
    sys.exit("mlx_lm/models/gated_delta.py not found under " + venv)
text = open(hits[0]).read()
if "gated_delta_update" in text and new in text:
    print("already patched:", hits[0])
elif old in text:
    open(hits[0], "w").write(text.replace(old, new))
    print("patched:", hits[0])
elif old_re.search(text):
    open(hits[0], "w").write(old_re.sub(new, text, count=1))
    print("patched (commented variant):", hits[0])
else:
    sys.exit("unrecognized gated_delta.py content; refusing to patch")
