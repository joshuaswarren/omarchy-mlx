#!/usr/bin/env python3
"""Qwen3-Next: compile the gated-norm + weight-mul + swiglu tail as ONE region.

Measured 2026-11-23 on the 16-inch M1 Max (GAP-F, M1/M2 card 10320790811): a
2B decode token at ctx 512 issues 198 cast dispatches (CastBF16F32 162,
CastF32BF16 36) - 108 of them on the 2048-wide row and 72 on the 16-element
gated-delta state - and every one of those launches costs 9-20 us end to end
because the cast is launch-bound, not data-bound. The casts come from the
composed path in mlx_lm/models/qwen3_next.py: the gated norm writes bf16, the
weight multiply runs in f32 (so the tail is widened), _precise_swiglu widens
gate and x again, and the product is narrowed back to bf16.

The fix is to let the compiler see the whole tail as one graph: mx.fast.rms_norm
plus its weight multiply plus the swiglu, so the backend's compiled path can
fold the elementwise chain into the swiglu dispatch instead of issuing the
casts as separate launches. The operands and the bf16 rounding sequence are
unchanged (widen bf16->f32 is exact, the sigmoid and the product run in f32,
only the final result is rounded to bf16), so the token ids must not move -
the ABBA gates on ids_sha256 equality.

Behind MLX_OMARCHY_FUSED_SWIGLU=1; the default path is untouched.
Idempotent; refuses unrecognized content. Usage: patch <venv>
"""
import glob
import sys

FLAG = "_FUSED_SWIGLU"

OLD_CALL = """        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)
"""

NEW_CALL = """        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            if _FUSED_SWIGLU:
                # mlx-omarchy: one compiled region for norm + weight-mul +
                # swiglu so the casts fold into the chain dispatch.
                return _fused_norm_swiglu(hidden_states, gate, self.weight, self.eps)
            return _precise_swiglu(hidden_states, gate, x)
        else:
            return x.astype(hidden_states.dtype)
"""

HELPER = """@partial(mx.compile, shapeless=True)
def _fused_norm_swiglu(hidden_states, gate, weight, eps):
    x = mx.fast.rms_norm(hidden_states, weight, eps)
    return _precise_swiglu(hidden_states, gate, x)


"""

ANCHOR = "class Qwen3NextRMSNormGated(nn.Module):\n"
FLAG_NEW = (
    '_FUSED_SWIGLU = os.environ.get("MLX_OMARCHY_FUSED_SWIGLU", "0") != "0"\n\n\n' + ANCHOR
)

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
path = site[0] + "/qwen3_next.py"
text = open(path).read()
if FLAG in text:
    print("already patched:", path)
    sys.exit(0)
if text.count(OLD_CALL) != 1:
    sys.exit("gated-norm call anchor not unique in " + path + "; refusing to patch")
if text.count(ANCHOR) != 1:
    sys.exit("class anchor not unique in " + path + "; refusing to patch")
if "\nimport os\n" not in text:
    sys.exit("qwen3_next.py does not import os; refusing to patch")
if "def _precise_swiglu" not in text:
    sys.exit("_precise_swiglu not found in " + path + "; refusing to patch")
text = text.replace(ANCHOR, FLAG_NEW, 1)
text = text.replace("def _precise_swiglu", HELPER + "def _precise_swiglu", 1)
text = text.replace(OLD_CALL, NEW_CALL, 1)
open(path, "w").write(text)
print("patched:", path)
