#!/usr/bin/env python3
"""DispatchFuse: route Hk<Hv GDN decode through the fused raw kernel.

mlx_lm gated_delta.py's dispatcher hands decode (T==1) to
mx.fast.gated_delta_update_raw, but the omarchy backend's
GatedDeltaUpdate::use_fallback requires Hk == Hv (Dk = Dv = 128). Models
with linear_num_key_heads < linear_num_value_heads (Qwen3.5-9B: 16 vs 32)
fall through to the composed per-token fallback: ~700 extra small
dispatches per decoded token (AsType/Multiply/Sum/Subtract F32 soup, per
the 2026-10-03 3-model census).

This patch expands q/k to the value-head count before the dispatch. The
repeat is a bit-exact copy, and the fused kernel's gate chain and walks
keep the composed fallback's rounding sites (per-call replay of live 9B
operands: 288/288 calls bit-identical on the tiled, perrow_pf and perrow
kernels).

Gate MLX_OMARCHY_GDN_RAW_REPEAT: "1" expands on every part, "0" never
expands (the exact composed dispatch). Unset: on, except G13 legacy parts
(device_name contains G13 but not G13C), which keep the composed chain:
the expansion route is unmeasured for bit-exactness on their older Mesa.
This is the only per-chip GDN policy. GatedDeltaUpdate::use_fallback
carries none, because it gates every GDN launch, not just this route.

Idempotent; upgrades the previous release's block; refuses unrecognized
content. Usage: patch <venv>
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
    # MLX_OMARCHY_GDN_RAW_REPEAT (see _gdn_raw_repeat_on).
    if (
        _gdn_raw_repeat_on()
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

# The previous release's gate lines (env default "1", no chip policy).
COND_PREV = '        os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "1") == "1"\n'
COND_NEW = "        _gdn_raw_repeat_on()\n"
NOTE_PREV = "    # MLX_OMARCHY_GDN_RAW_REPEAT (default 0 = exact existing dispatch).\n"
NOTE_NEW = "    # MLX_OMARCHY_GDN_RAW_REPEAT (see _gdn_raw_repeat_on).\n"

HELPER_ANCHOR = "\ndef gated_delta_update("
HELPER = '''
_GDN_G13_LEGACY = None


def _gdn_raw_repeat_on():
    # MLX_OMARCHY_GDN_RAW_REPEAT: "1" expands q/k on every part, "0" never.
    # Unset: on, except G13 legacy parts, which keep the composed chain.
    value = os.environ.get("MLX_OMARCHY_GDN_RAW_REPEAT", "")
    if value:
        return value == "1"
    global _GDN_G13_LEGACY
    if _GDN_G13_LEGACY is None:
        name = str(mx.device_info().get("device_name", ""))
        _GDN_G13_LEGACY = "G13" in name and "G13C" not in name
    return not _GDN_G13_LEGACY

'''

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
site = site[0]

g = site + "/gated_delta.py"
text = open(g).read()
if "def _gdn_raw_repeat_on(" in text:
    print("already patched:", g)
    sys.exit(0)
if text.count(COND_PREV) == 1 and text.count(NOTE_PREV) == 1:
    text = text.replace(COND_PREV, COND_NEW, 1).replace(NOTE_PREV, NOTE_NEW, 1)
elif "MLX_OMARCHY_GDN_RAW_REPEAT" in text:
    sys.exit("unrecognized raw-repeat block in " + g + "; refusing to patch")
else:
    if text.count(DISPATCH_OLD) != 1:
        sys.exit("dispatch anchor not unique in " + g + "; refusing to patch")
    text = text.replace(DISPATCH_OLD, DISPATCH_NEW, 1)
if IMPORT_NEW not in text:
    if text.count(IMPORT_OLD) != 1:
        sys.exit("unrecognized import block in " + g + "; refusing to patch")
    text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
if text.count(HELPER_ANCHOR) != 1:
    sys.exit("gated_delta_update anchor not unique in " + g + "; refusing to patch")
text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
open(g, "w").write(text)
print("patched:", g)
