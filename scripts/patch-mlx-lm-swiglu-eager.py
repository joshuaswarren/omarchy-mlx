#!/usr/bin/env python3
"""Uncompile mlx-lm's swiglu so the Omarchy GEMV planner can fold it.

mlx_lm.models.activations.swiglu is mx.compile'd for Metal's kernel
fusion, and so is the nn.silu it calls. On Omarchy a Compiled node is
opaque to the eager planner: the compiled tape still runs as one
SwigluBF16 dispatch, but the gate/up GEMV group cannot see the
silu(gate) * up chain behind it, so the SwiGLU store epilogue
(fused_chain.cpp swiglu_plans) never fires and every MLP layer pays a
standalone swiglu dispatch. Spelled out as eager ops,
(gate * sigmoid(gate)) * x, the planner sees Multiply(Multiply(gate,
Sigmoid(gate)), up): decode rows fold into the gate/up dispatch (-1
dispatch per MLP layer), prefill rows form the same three-instruction
chain and take the same swiglu.comp dispatch the compiled tape did.

Idempotent patch for mlx-lm 0.31.3 venvs. Usage:
python3 patch-mlx-lm-swiglu-eager.py /path/to/venv
"""
import glob
import sys

OLD = """@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x
"""
NEW = """def swiglu(gate, x):
    # mlx-omarchy swiglu-eager patch: eager, uncompiled silu so the GEMV
    # planner sees Multiply(Multiply(gate, Sigmoid(gate)), x) and folds it.
    return (gate * mx.sigmoid(gate)) * x
"""
MARKER = "swiglu-eager patch"

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
path = site[0] + "/activations.py"
text = open(path).read()
if MARKER in text:
    print("already patched:", path)
elif OLD not in text:
    sys.exit("unrecognized content in " + path + "; refusing to patch")
else:
    open(path, "w").write(text.replace(OLD, NEW))
    print("patched:", path)
