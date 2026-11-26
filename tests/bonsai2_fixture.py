"""Shared tiny-pack fixture for Bonsai-2 loader/server tests.

Builds a real, tiny prism_hadamard_qwen35 checkpoint (schema 2): a 2-layer
qwen3_5 TextModel with every wide projection replaced by a Hadamard-folded
affine 2-bit group-128 `Packed` module, saved in the pack's tensor
namespace (language_model.* prefix) with a fake vision tensor to exercise
exclusion accounting. No network, no large weights, CPU only.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "serve") not in sys.path:
    sys.path.insert(0, str(REPO / "serve"))

BLOCK = 512
TINY_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "vocab_size": 16,
    "max_position_embeddings": 512,
    "full_attention_interval": 2,
    "linear_num_value_heads": 16,
    "linear_num_key_heads": 4,
    "linear_value_head_dim": 32,
    "linear_key_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "tie_word_embeddings": False,
    "rope_parameters": {"type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
}


def packed_targets(config):
    """Packed module paths, mirroring the real pack (all wide projections)."""
    interval = config["full_attention_interval"]
    targets = ["model.embed_tokens", "lm_head"]
    for i in range(config["num_hidden_layers"]):
        if (i + 1) % interval == 0:
            targets += [f"model.layers.{i}.self_attn.{p}_proj" for p in "qkvo"]
        else:
            targets += [
                f"model.layers.{i}.linear_attn.{n}"
                for n in ("in_proj_qkv", "in_proj_z", "out_proj")
            ]
        targets += [f"model.layers.{i}.mlp.{n}_proj" for n in ("gate", "up", "down")]
    return targets


def build_reference_model():
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from mlx_omarchy_bonsai2.packed import Packed

    rng = np.random.default_rng(20260920)
    args = TextModelArgs.from_dict(TINY_TEXT_CONFIG)
    model = TextModel(args)
    for name, arr in tree_flatten(model.parameters()):
        if isinstance(arr, mx.array) and arr.dtype in (mx.float16, mx.float32):
            replacement = mx.array(rng.normal(0, 0.05, size=arr.shape).astype(np.float32))
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            setattr(parent, parts[-1], replacement.astype(arr.dtype))

    for path in packed_targets(TINY_TEXT_CONFIG):
        parent = model
        parts = path.split(".")
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        original = getattr(parent, parts[-1])
        embedding = isinstance(original, nn.Embedding)
        w = original.weight.astype(mx.float16)
        packed, scales, biases = mx.quantize(w, group_size=128, bits=2)
        signs = mx.array(rng.choice([-1.0, 1.0], size=(w.shape[1],)).astype(np.float32))
        setattr(parent, parts[-1], Packed((packed, scales, biases), BLOCK, signs, embedding))
    model.eval()
    return model


def write_tokenizer(pack_dir: Path):
    """Minimal offline fast tokenizer + chat template for the server tests."""
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2}
    for i in range(3, 16):
        vocab["<t%d>" % i] = i
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(pack_dir / "tokenizer.json"))
    (pack_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<pad>",
                "chat_template": "{% for message in messages %}{{ message['content'] }} {% endfor %}",
            }
        )
    )


def build_tiny_pack(directory=None):
    """Write the fixture pack; returns (pack_dir, reference_model)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    root = Path(tempfile.mkdtemp()) if directory is None else Path(directory)
    pack_dir = root / "pack"
    pack_dir.mkdir(parents=True, exist_ok=True)
    model = build_reference_model()
    tree = {("language_model." + name): value for name, value in tree_flatten(model.parameters())}
    tree["visual.tower.weight"] = mx.ones((2, 2), mx.float16)  # fake vision tensor
    mx.save_safetensors(str(pack_dir / "model.safetensors"), tree)
    config = {
        "schema_version": 2,
        "model_type": "prism_hadamard_qwen35",
        "requires_runtime": "runtime/artifact.py",
        "base_model_type": "qwen3_5",
        "tensor_namespace": "mlx-vlm-qwen3_5",
        "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
        "text_config": TINY_TEXT_CONFIG,
        "modules": [
            {"path": path, "block": BLOCK, "embedding": path == "model.embed_tokens", "dtype": "float16"}
            for path in packed_targets(TINY_TEXT_CONFIG)
        ],
    }
    (pack_dir / "config.json").write_text(json.dumps(config, indent=2))
    (pack_dir / "LICENSE").write_text("Apache-2.0 fixture\n")
    (pack_dir / "NOTICE.txt").write_text("fixture notice\n")
    write_tokenizer(pack_dir)
    return pack_dir, model
