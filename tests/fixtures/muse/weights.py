"""Deterministic tiny Muse Spark weights for the adapter test and its fixture generator.

The names and shapes are the transformers MuseSparkTextModel state dict (checked by generate_fixture.py against the real class) plus
lm_head.weight. Values come from numpy RandomState, whose stream does not change between numpy versions, so the test rebuilds
the exact weights the reference run used and only the reference logits are stored.
"""

import hashlib

import numpy as np

# Dimensions differ on purpose (moe_hidden_size 32, moe_intermediate_size 48) so a swapped axis cannot pass.
CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 96,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "vocab_size": 128,
    "sliding_window": 8,
    "moe_hidden_size": 32,
    "moe_intermediate_size": 48,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "max_position_embeddings": 256,
}
SEED = 0
HEAD_SCALE = 1.5
IDS_COUNT = 24


def names_and_shapes(cfg=CONFIG):
    h, m, i = cfg["hidden_size"], cfg["moe_hidden_size"], cfg["moe_intermediate_size"]
    e = cfg["num_local_experts"]
    qd = cfg["num_attention_heads"] * cfg["head_dim"]
    kd = cfg["num_key_value_heads"] * cfg["head_dim"]
    out = [("embed_tokens.weight", (cfg["vocab_size"], h))]
    for n in range(cfg["num_hidden_layers"]):
        p = f"layers.{n}."
        out += [
            (p + "self_attn.q_proj.weight", (qd, h)),
            (p + "self_attn.k_proj.weight", (kd, h)),
            (p + "self_attn.v_proj.weight", (kd, h)),
            (p + "self_attn.o_proj.weight", (h, qd)),
            (p + "self_attn.gate_proj.weight", (qd, h)),
            (p + "mlp.gate.weight", (e, h)),
            (p + "mlp.gate.e_score_correction_bias", (e,)),
            (p + "mlp.pre_expert_proj.weight", (m, h)),
            (p + "mlp.pre_expert_norm.weight", (m,)),
            (p + "mlp.experts.gate_up_proj", (e, 2 * i, m)),
            (p + "mlp.experts.down_proj", (e, m, i)),
            (p + "mlp.experts.post_expert_norm.weight", (m,)),
            (p + "mlp.post_expert_proj.weight", (h, m)),
            (p + "input_layernorm.weight", (h,)),
            (p + "pre_feedforward_layernorm.weight", (h,)),
            (p + "post_feedforward_layernorm.weight", (h,)),
            (p + "post_attention_residual_gate.gate", (h,)),
            (p + "post_feedforward_residual_gate.gate", (h,)),
        ]
    out.append(("norm.weight", (h,)))
    return out


def make_weights(cfg=CONFIG, seed=SEED):
    """Return (state, lm_head, ids): state maps transformers text-model names to float32 arrays."""
    rng = np.random.RandomState(seed)
    state = {}
    for name, shape in names_and_shapes(cfg):
        if len(shape) >= 2:
            scale = 0.5 if name.endswith("mlp.gate.weight") else 0.08
        elif "residual_gate" in name:
            scale = 1.0
        else:
            scale = 0.1
        state[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    lm_head = (rng.standard_normal((cfg["vocab_size"], cfg["hidden_size"])) * HEAD_SCALE).astype(np.float32)
    ids = rng.randint(0, cfg["vocab_size"], size=IDS_COUNT)
    return state, lm_head, ids


def weights_digest(state, lm_head):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].tobytes())
    digest.update(lm_head.tobytes())
    return digest.hexdigest()
