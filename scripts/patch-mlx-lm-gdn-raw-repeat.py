#!/usr/bin/env python3
"""DispatchFuse: route Hk<Hv GDN decode through the fused raw kernel.

mlx_lm gated_delta.py's dispatcher hands decode (T==1) to
mx.fast.gated_delta_update_raw, but the omarchy backend's
GatedDeltaUpdate::use_fallback requires Hk == Hv (Dk = Dv = 128). Models
with linear_num_key_heads < linear_num_value_heads (Qwen3.5-9B: 16 vs 32)
fall through to the composed per-token fallback: ~700 extra small
dispatches per decoded token (AsType/Multiply/Sum/Subtract F32 soup, per
the 2026-10-03 3-model census).

This patch expands q/k to the value-head count before the dispatch.
MLX_OMARCHY_GDN_RAW_REPEAT defaults to 1; set it to 0 to use the
composed path. The 9B route's free-run tokens can differ from the
composed path, but the updated bf16-ULP numerics bar accepts the measured
divergences: >=99% teacher-forced top-1 agreement and every free-run
first divergence within one bf16 ULP, alongside per-op fp64 and PPL gates.
Idempotent; refuses unrecognized content. Usage: patch <venv>
"""
import glob
import sys

IMPORT_OLD = "from functools import partial\n"
IMPORT_NEW = "from functools import partial\nimport os\n"

DISPATCH_OLD = """    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
"""

DISPATCH_NEW = """    # mlx-omarchy decode fast-route (GQA repeat): the fused raw decode
    # kernel requires Hk == Hv (omarchy GatedDeltaUpdate::use_fallback);
    # expand q/k to the value-head count so Hk<Hv models take the fused
    # route instead of the composed per-token fallback. Gate:
    # MLX_OMARCHY_GDN_RAW_REPEAT (default ON; set to 0 to opt out).
    if (
        os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") != "0"
        and use_kernel
        and q.shape[1] == 1
        and q.shape[-1] == 128
        and v.shape[-1] == 128
        and q.dtype == mx.bfloat16
        and v.shape[-2] % q.shape[-2] == 0
        and q.shape[-2] != v.shape[-2]
        and hasattr(mx.fast, "gated_delta_update_raw")
    ):
        q = mx.repeat(q, v.shape[-2] // q.shape[-2], -2)
        k = mx.repeat(k, v.shape[-2] // k.shape[-2], -2)
    if not use_kernel or mx.default_device() != mx.gpu:
        return gated_delta_ops(q, k, v, g, beta, state, mask)
"""


def patch_file(path, old, new, marker):
    text = open(path).read()
    if marker in text:
        print("already patched:", path)
        return
    if old not in text:
        sys.exit("unrecognized content in " + path + "; refusing to patch")
    open(path, "w").write(text.replace(old, new, 1))
    print("patched:", path)


venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
site = site[0]

g = site + "/gated_delta.py"
text = open(g).read()
if "MLX_OMARCHY_GDN_RAW_REPEAT" in text:
    print("already patched:", g)
else:
    if text.count(DISPATCH_OLD) != 1:
        sys.exit("dispatch anchor not unique in " + g + "; refusing to patch")
    if IMPORT_NEW not in text:
        if text.count(IMPORT_OLD) != 1:
            sys.exit("unrecognized import block in " + g + "; refusing to patch")
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    text = text.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
    open(g, "w").write(text)
    print("patched:", g)
