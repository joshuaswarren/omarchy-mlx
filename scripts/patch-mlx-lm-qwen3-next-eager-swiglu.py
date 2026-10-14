#!/usr/bin/env python3
"""Let the GEMV store-epilogue fold fire on the Qwen3-Next family's decode MLP.

mlx_lm's Qwen3NextMLP (Qwen3.5 dense, Qwen3.8, the Qwen3.6 MoE shared expert) calls the
compiled swiglu(gate, up) of mlx_lm/models/activations.py, defined there as swiglu(gate, x) = nn.silu(gate) * x. The compiled function reaches the omarchy backend as ONE Compiled
node, so the backend's SwiGLU store-epilogue fold (fused_chain.cpp: the gate and up GEMVs
compute silu(gate) * up in one dispatch) never matches it: a standalone Swiglu dispatch runs
per layer per token (32 per Qwen3.5-9B decode token). The fold's planner matches eager
Sigmoid and Multiply nodes, so for single-token rows this patch writes the same product as
raw ops: gate * sigmoid(gate) * up. silu is gate * sigmoid(gate) and the product order is
(gate * sigmoid) * up, so the operands and the bf16 rounding sequence equal the compiled
swiglu. Rows > 1 (prefill) keep the compiled function unchanged.
MLX_OMARCHY_EAGER_SWIGLU=0 keeps the compiled function for decode too.
Idempotent; refuses unrecognized content. Usage: patch <venv>
"""
import glob
import sys

OLD = """    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
"""

NEW = """    def __call__(self, x) -> mx.array:
        # mlx-omarchy: single-token rows write silu(gate) * up as raw ops so the
        # backend can fold it into the gate/up GEMV dispatch (the compiled
        # swiglu is one opaque node the fold cannot match). Same operands and
        # rounding as swiglu(); prefill keeps the compiled function.
        # Kill switch: MLX_OMARCHY_EAGER_SWIGLU=0.
        if _EAGER_SWIGLU and x.shape[-2] == 1:
            gate = self.gate_proj(x)
            return self.down_proj(gate * mx.sigmoid(gate) * self.up_proj(x))
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
"""

FLAG_ANCHOR = "class Qwen3NextMLP(nn.Module):\n"
FLAG_NEW = '_EAGER_SWIGLU = os.environ.get("MLX_OMARCHY_EAGER_SWIGLU", "1") != "0"\n\n\n' + FLAG_ANCHOR
MARKER = "_EAGER_SWIGLU"

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
path = site[0] + "/qwen3_next.py"
text = open(path).read()
if MARKER in text:
    print("already patched:", path)
    sys.exit(0)
if text.count(OLD) != 1:
    sys.exit("MLP anchor not unique in " + path + "; refusing to patch")
if text.count(FLAG_ANCHOR) != 1:
    sys.exit("class anchor not unique in " + path + "; refusing to patch")
if "\nimport os\n" not in text:
    sys.exit("qwen3_next.py does not import os; refusing to patch")
text = text.replace(FLAG_ANCHOR, FLAG_NEW, 1).replace(OLD, NEW, 1)
open(path, "w").write(text)
print("patched:", path)
