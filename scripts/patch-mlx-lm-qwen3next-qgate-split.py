#!/usr/bin/env python3
"""Split Qwen3-Next's fused q_proj into query and gate projections.

Qwen3NextAttention.q_proj emits [heads, 2 * head_dim] per token and the
model splits queries from the output gate along the last axis. On
Omarchy both halves are then strided views, so q_norm and the output
gate each pay a CopyGeneral dispatch to become contiguous (2 dispatches
per attention layer per token, 12/token on Qwen3.8-2B). Rows of a
transposed quantized weight (and its per-row scales and biases) are
independent, so permuting them into [all query rows | all gate rows]
at load time and running two projections changes no arithmetic: every
output row is the same dot product of the same normed row. The two
projections share x, so the GEMV planner groups q, gate, k, v into one
dispatch (kQmmVecMultiWeights = 4), and both outputs are contiguous.

Sites (mlx-lm 0.31.3):
- Qwen3NextAttention.__init__ / __call__ (qwen3_next.py): q_gate_proj
  module beside q_proj; the split becomes two projections.
- Model.sanitize (qwen3_next.py) and TextModel.sanitize (qwen3_5.py):
  split the checkpoint's fused q_proj rows per head.
- Model.shard (qwen3_5.py): shard q_gate_proj like q_proj.

Idempotent. Usage: python3 patch-mlx-lm-qwen3next-qgate-split.py /venv
"""
import glob
import sys

MARKER = "qgate-split patch"

INIT_OLD = """        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=args.attention_bias,
        )
"""
INIT_NEW = """        # mlx-omarchy qgate-split patch: query and output-gate rows of the
        # checkpoint's fused q_proj become two projections (see sanitize).
        self.q_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.q_gate_proj = nn.Linear(
            args.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=args.attention_bias,
        )
"""
CALL_OLD = """        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
"""
CALL_NEW = """        queries = self.q_proj(x).reshape(B, L, self.num_attention_heads, -1)
        gate = self.q_gate_proj(x)
"""
HELPER_OLD = """class Qwen3NextAttention(nn.Module):
"""
HELPER_NEW = """def split_q_gate_weights(weights, num_heads, head_dim):
    \"\"\"Split fused [heads * 2 * head_dim, ...] q_proj rows into q_proj
    (query rows) and q_gate_proj (gate rows). Row order inside each head
    is kept; weight, scales and biases are all row-indexed.\"\"\"
    for key in list(weights):
        name = key.rsplit(".", 1)[-1]
        if not key.endswith((".q_proj.weight", ".q_proj.scales", ".q_proj.biases")):
            continue
        value = weights[key]
        if value.shape[0] != num_heads * 2 * head_dim:
            continue
        both = value.reshape(num_heads, 2, head_dim, *value.shape[1:])
        weights[key] = both[:, 0].reshape(num_heads * head_dim, *value.shape[1:])
        weights[key[: -len(".q_proj." + name)] + ".q_gate_proj." + name] = both[
            :, 1
        ].reshape(num_heads * head_dim, *value.shape[1:])
    return weights


class Qwen3NextAttention(nn.Module):
"""
NEXT_SANITIZE_OLD = """    def sanitize(self, weights):
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
"""
NEXT_SANITIZE_NEW = """    def sanitize(self, weights):
        weights = split_q_gate_weights(
            weights, self.args.num_attention_heads, self.args.head_dim
        )
        if "model.layers.0.mlp.experts.0.up_proj.weight" not in weights:
            return weights
"""
Q35_SANITIZE_OLD = """    def sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
"""
Q35_SANITIZE_NEW = """    def sanitize(self, weights):
        weights = split_q_gate_weights(
            weights, self.args.num_attention_heads, self.args.head_dim
        )
        has_mtp_weights = any("mtp." in k for k in weights)
"""
Q35_IMPORT_OLD = """from .qwen3_next import Qwen3NextAttention as Attention
"""
Q35_IMPORT_NEW = """from .qwen3_next import Qwen3NextAttention as Attention
from .qwen3_next import split_q_gate_weights
"""
Q35_SHARD_OLD = """                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
"""
Q35_SHARD_NEW = """                layer.self_attn.q_proj = shard_linear(
                    layer.self_attn.q_proj, "all-to-sharded", group=group
                )
                layer.self_attn.q_gate_proj = shard_linear(
                    layer.self_attn.q_gate_proj, "all-to-sharded", group=group
                )
"""


def patch(path, edits):
    text = open(path).read()
    if MARKER in text:
        print("already patched:", path)
        return
    for old, new in edits:
        if old not in text:
            sys.exit("unrecognized content in " + path + "; refusing to patch:\n" + old)
        text = text.replace(old, new, 1)
    open(path, "w").write(text)
    print("patched:", path)


venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
site = site[0]
patch(
    site + "/qwen3_next.py",
    [
        (HELPER_OLD, HELPER_NEW),
        (INIT_OLD, INIT_NEW),
        (CALL_OLD, CALL_NEW),
        (NEXT_SANITIZE_OLD, NEXT_SANITIZE_NEW),
    ],
)
patch(
    site + "/qwen3_5.py",
    [
        (Q35_IMPORT_OLD, Q35_IMPORT_NEW + "# " + MARKER + "\n"),
        (Q35_SANITIZE_OLD, Q35_SANITIZE_NEW),
        (Q35_SHARD_OLD, Q35_SHARD_NEW),
    ],
)
