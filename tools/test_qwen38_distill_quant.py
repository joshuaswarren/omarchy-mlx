#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Tests for tools/qwen38_distill_quant.py.

The converter's transform must mirror mlx-lm v0.31.3
(qwen3_5_moe/qwen3_5 sanitize + quant_predicate) bit for bit; these
tests pin that contract on synthetic tensors, so a loader-contract
regression fails here before any multi-GiB artifact is produced.
"""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import qwen38_distill_quant as q  # noqa: E402

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402  (exercises the real loader stack too)


# ---------------------------------------------------------------------------
# key planning
# ---------------------------------------------------------------------------

def test_rename_drops_vision_and_mtp():
    assert q.rename_key("model.visual.blocks.0.mlp0.fc1.weight") is None
    assert q.rename_key("mtp.fc.weight") is None
    assert q.rename_key("mtp.layers.0.mlp.gate.weight") is None
    assert q.rename_key("vision_tower.vision_model.patch_embed.weight") is None


def test_rename_prefixes():
    assert q.rename_key(
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ) == "language_model.model.layers.3.self_attn.q_proj.weight"
    assert q.rename_key("lm_head.weight") == "language_model.lm_head.weight"
    assert q.rename_key(
        "model.language_model.embed_tokens.weight"
    ) == "language_model.model.embed_tokens.weight"


def test_expert_split_keys():
    k, part = q.expert_split(
        "model.language_model.layers.0.mlp.experts.gate_up_proj"
    )
    assert part == "gate_up"
    assert k == "model.language_model.layers.0.mlp."
    k, part = q.expert_split(
        "model.language_model.layers.0.mlp.experts.down_proj"
    )
    assert part is None
    assert k == "model.language_model.layers.0.mlp.switch_mlp.down_proj"
    # shared expert is a plain module, untouched
    k, part = q.expert_split(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight"
    )
    assert part is None and k.endswith("shared_expert.down_proj.weight")


def test_plan_router_gets_8bit():
    for key in (
        "model.language_model.layers.7.mlp.gate.weight",
        "model.language_model.layers.7.mlp.shared_expert_gate.weight",
    ):
        actions, note = q.plan_tensor(key, [256, 2048], bits=4, group=64)
        assert actions == [(q.rename_key(key), "quantize", (64, 8))], key


def test_plan_small_tensors_stay_fp():
    for key, shape in (
        ("model.language_model.layers.0.linear_attn.A_log", [32]),
        ("model.language_model.layers.0.linear_attn.dt_bias", [32]),
        ("model.language_model.layers.0.linear_attn.norm.weight", [128]),
        ("model.language_model.layers.0.linear_attn.conv1d.weight", [8192, 1, 4]),
        ("model.language_model.layers.0.input_layernorm.weight", [2048]),
    ):
        actions, note = q.plan_tensor(key, shape, bits=4, group=64)
        assert len(actions) == 1 and actions[0][1] == "copy", key


def test_plan_gate_up_split_rows():
    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    actions, _ = q.plan_tensor(key, [256, 1024, 2048], bits=4, group=64)
    assert [a[0] for a in actions] == [
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
        "language_model.model.layers.0.mlp.switch_mlp.up_proj",
    ]
    assert [a[2] for a in actions] == [(0, 512), (512, None)]


# ---------------------------------------------------------------------------
# transform correctness
# ---------------------------------------------------------------------------

def test_transform_norm_shift_and_conv_moveaxis():
    has_mtp = True
    weights = {
        "model.language_model.layers.0.input_layernorm.weight":
            mx.full((4,), 3.0, mx.bfloat16),
        "model.language_model.layers.0.linear_attn.conv1d.weight":
            mx.arange(8, dtype=mx.float32).reshape(2, 1, 4),
        "model.language_model.layers.0.linear_attn.A_log":
            mx.ones((2,), mx.float32),
    }
    out = q.transform_shard(weights, bits=4, group=64, has_mtp=has_mtp)
    ln = out["language_model.model.layers.0.input_layernorm.weight"]
    assert mx.max(mx.abs(ln - mx.full((4,), 4.0, mx.bfloat16))).item() == 0
    conv = out["language_model.model.layers.0.linear_attn.conv1d.weight"]
    assert conv.shape == (2, 4, 1)
    assert out["language_model.model.layers.0.linear_attn.A_log"].shape == (2,)

    # Without an mtp block the loader applies no shift; neither may we.
    out2 = q.transform_shard(
        {"model.language_model.layers.0.input_layernorm.weight":
            mx.full((4,), 3.0, mx.bfloat16)},
        bits=4, group=64, has_mtp=False,
    )
    assert mx.min(out2[
        "language_model.model.layers.0.input_layernorm.weight"
    ]).item() == 3.0


def test_split_then_quantize_is_bit_identical_to_quantize_then_split():
    w = mx.random.normal((4, 8, 256), key=mx.random.key(0))
    bits, group = 4, 64
    whole_w, whole_s, whole_b = mx.quantize(w, group, bits)
    g = q.transform_shard(
        {"model.language_model.layers.0.mlp.experts.gate_up_proj": w},
        bits, group, has_mtp=False,
    )
    # group quantization is row-wise along the last dim, so slicing the
    # bf16 tensor before quantizing must reproduce the packed rows of
    # quantize-then-slice exactly.
    assert mx.array_equal(
        g["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"],
        whole_w[..., :4, :],
    )
    assert mx.array_equal(
        g["language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales"],
        whole_s[..., :4, :],
    )
    assert mx.array_equal(
        g["language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases"],
        whole_b[..., :4, :],
    )


def test_quantized_bytes_matches_mx_layout():
    shape = (4, 512)
    assert q.quantized_bytes(shape, 64, 4) == 1024 + 128


def test_kv_and_state_math():
    config = {"text_config": {
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] * 1,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }}
    m = q.kv_and_state_per_token(config)
    assert m["full_attention_layers"] == 1
    assert m["linear_attention_layers"] == 3
    # 2 kv heads x 256 x (K+V) x bf16 = 2048 B/token/layer
    assert m["kv_bytes_per_token"] == 2048
    # 32 heads x 128 x 128 x fp32 = 2 MiB/layer
    assert m["gdn_ssm_state_bytes"] == 3 * 32 * 128 * 128 * 4


# ---------------------------------------------------------------------------
# output config contract (what mlx_lm.load consumes)
# ---------------------------------------------------------------------------

def test_output_quantization_config_contract():
    cfg = q.output_quantization_config(num_layers=2, bits=4, group=64)
    assert cfg["bits"] == 4 and cfg["group_size"] == 64
    assert cfg["mode"] == "affine"
    for l in range(2):
        for path in (
            f"language_model.model.layers.{l}.mlp.gate",
            f"language_model.model.layers.{l}.mlp.shared_expert_gate",
        ):
            assert cfg[path] == {"group_size": 64, "bits": 8}
    assert len(cfg) == 3 + 4


# ---------------------------------------------------------------------------
# end-to-end transform + safetensors round trip on a synthetic mini model
# ---------------------------------------------------------------------------

def _mini_source_shard():
    return {
        "model.language_model.embed_tokens.weight":
            mx.random.normal((64, 128), key=mx.random.key(1)),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight":
            mx.random.normal((32, 128), key=mx.random.key(2)),
        "model.language_model.layers.0.linear_attn.conv1d.weight":
            mx.random.normal((32, 1, 4), key=mx.random.key(3)),
        "model.language_model.layers.0.linear_attn.A_log":
            mx.random.normal((2,), key=mx.random.key(4)),
        "model.language_model.layers.0.linear_attn.dt_bias":
            mx.random.normal((2,), key=mx.random.key(5)),
        "model.language_model.layers.0.mlp.experts.gate_up_proj":
            mx.random.normal((4, 8, 128), key=mx.random.key(6)),
        "model.language_model.layers.0.mlp.experts.down_proj":
            mx.random.normal((4, 128, 64), key=mx.random.key(7)),
        "model.language_model.layers.0.mlp.gate.weight":
            mx.random.normal((4, 128), key=mx.random.key(8)),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight":
            mx.random.normal((4, 128), key=mx.random.key(9)),
        "model.language_model.layers.0.input_layernorm.weight":
            mx.full((128,), 2.0, mx.bfloat16),
        "model.language_model.norm.weight":
            mx.full((128,), 2.0, mx.bfloat16),
        "lm_head.weight":
            mx.random.normal((64, 128), key=mx.random.key(10)),
        "model.visual.patch_embed.proj.weight":
            mx.random.normal((4, 4), key=mx.random.key(11)),
        "mtp.norm.weight":
            mx.full((128,), 2.0, mx.bfloat16),
    }


EXPECTED_MINI_KEYS = {
    "language_model.model.embed_tokens.weight",
    "language_model.model.embed_tokens.scales",
    "language_model.model.embed_tokens.biases",
    "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
    "language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
    "language_model.model.layers.0.linear_attn.in_proj_qkv.biases",
    "language_model.model.layers.0.linear_attn.conv1d.weight",
    "language_model.model.layers.0.linear_attn.A_log",
    "language_model.model.layers.0.linear_attn.dt_bias",
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
    "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
    "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight",
    "language_model.model.layers.0.mlp.switch_mlp.up_proj.scales",
    "language_model.model.layers.0.mlp.switch_mlp.up_proj.biases",
    "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
    "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales",
    "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases",
    "language_model.model.layers.0.mlp.gate.weight",
    "language_model.model.layers.0.mlp.gate.scales",
    "language_model.model.layers.0.mlp.gate.biases",
    "language_model.model.layers.0.mlp.shared_expert_gate.weight",
    "language_model.model.layers.0.mlp.shared_expert_gate.scales",
    "language_model.model.layers.0.mlp.shared_expert_gate.biases",
    "language_model.model.layers.0.input_layernorm.weight",
    "language_model.model.norm.weight",
    "language_model.lm_head.weight",
    "language_model.lm_head.scales",
    "language_model.lm_head.biases",
}


def test_transform_key_set_exact():
    out = q.transform_shard(_mini_source_shard(), bits=4, group=64, has_mtp=True)
    assert set(out) == EXPECTED_MINI_KEYS
    # norms shifted by +1.0 under mtp
    assert mx.min(out["language_model.model.norm.weight"]).item() == 3.0
    # routers quantized 8/64, everything else global 4/64
    assert out["language_model.model.layers.0.mlp.gate.scales"].shape[-1] == 2
    assert out["language_model.lm_head.scales"].shape[-1] == 2


def test_safetensors_roundtrip(tmp_path):
    out = q.transform_shard(_mini_source_shard(), bits=4, group=64, has_mtp=True)
    p = tmp_path / "mini.safetensors"
    mx.save_safetensors(str(p), out)
    loaded = mx.load(str(p))
    assert set(loaded) == EXPECTED_MINI_KEYS
    assert mx.array_equal(
        loaded["language_model.lm_head.weight"],
        out["language_model.lm_head.weight"],
    )
    d = mx.dequantize(
        loaded["language_model.lm_head.weight"],
        loaded["language_model.lm_head.scales"],
        loaded["language_model.lm_head.biases"], 64, 4,
    )
    src_w, src_s, src_b = mx.quantize(
        _mini_source_shard()["lm_head.weight"], 64, 4
    )
    src = mx.dequantize(src_w, src_s, src_b, 64, 4)
    assert mx.array_equal(d, src)


def test_dequant_parity_vs_fp_baseline():
    x = mx.random.normal((128, 256), key=mx.random.key(12))
    for bits, mean_tol in ((4, 0.12), (6, 0.03), (8, 0.008)):
        w, s, b = mx.quantize(x, 64, bits)
        err = x - mx.dequantize(w, s, b, 64, bits)
        mean_err = mx.abs(err).mean().item()
        assert mean_err < mean_tol, (bits, mean_err)


# ---------------------------------------------------------------------------
# end-to-end: the real convert + verify commands on a local mini source
# ---------------------------------------------------------------------------

def test_convert_and_verify_end_to_end(tmp_path, monkeypatch):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    shard = "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(src_dir / shard), _mini_source_shard())
    src_index = {
        "weight_map": {k: shard for k in _mini_source_shard()}
    }
    (src_dir / "model.safetensors.index.json").write_text(
        json.dumps(src_index)
    )

    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 1,
            "mtp_num_hidden_layers": 1,
            "num_experts": 4,
        },
    }

    def fake_fetch(fname):
        if fname == "model.safetensors.index.json":
            return json.dumps(src_index).encode()
        if fname == "config.json":
            return json.dumps(config).encode()
        return b"placeholder:" + fname.encode()

    monkeypatch.setattr(q, "fetch_small", fake_fetch)
    monkeypatch.setattr(
        q,
        "shard_headers_from_http",
        lambda index: {
            shard: {
                k: {"shape": list(w.shape)}
                for k, w in _mini_source_shard().items()
            }
        },
    )

    out_dir = tmp_path / "out"
    args = argparse.Namespace(
        bits=4, group=64, out_dir=str(out_dir), source_dir=str(src_dir)
    )
    q.cmd_convert(args)
    assert out_dir.exists() and not out_dir.with_name("out.staging").exists()
    out_index = json.loads(
        (out_dir / "model.safetensors.index.json").read_text()
    )
    assert set(out_index["weight_map"]) == EXPECTED_MINI_KEYS
    out_shards = set(out_index["weight_map"].values())
    assert len(out_shards) == 1
    assert next(iter(out_shards)) == "model-1-of-1.safetensors"
    out_config = json.loads((out_dir / "config.json").read_text())
    assert out_config["quantization"]["bits"] == 4
    assert "quantization_config" in out_config
    quality = json.loads((out_dir / "quality_sample.json").read_text())
    assert quality, "quality samples must be recorded"
    assert all(v["rel_matmul_err"] < 0.35 for v in quality.values())

    v_args = argparse.Namespace(
        bits=4, group=64, out_dir=str(out_dir), source_dir=None
    )
    q.cmd_verify(v_args)
    receipt = json.loads((out_dir / "receipt.json").read_text())
    assert list(receipt["shard_sha256"]) == ["model-1-of-1.safetensors"]
    assert len(receipt["shard_sha256"]["model-1-of-1.safetensors"]) == 64
    assert receipt["n_quantized"] > 0
    assert receipt["output_license"] == "apache-2.0"

    # convert refuses to clobber a published output
    with pytest.raises(SystemExit):
        q.cmd_convert(args)


# ---------------------------------------------------------------------------
# the definitive offline contract check: load the artifact through the
# REAL pinned mlx_lm qwen3_5_moe module tree and run a forward pass
# ---------------------------------------------------------------------------

def _dim_consistent_source():
    """HF-layout bf16 weights for a 2-layer qwen3_5_moe mini model
    (layer 0 = gated delta linear attention, layer 1 = full attention).
    """
    import mlx.core as mx

    k = [mx.random.key(i) for i in range(40)]
    H, E, I = 128, 4, 64
    kdim, nv, hd = 64, 4, 32  # key_dim = 2 key heads x 32; v: 4 heads x 32
    vdim = nv * hd
    conv_dim = 2 * kdim + vdim

    def r(shape, i):
        return mx.random.normal(shape, key=k[i])

    w = {
        "model.language_model.embed_tokens.weight": r((256, H), 0),
        "model.language_model.norm.weight": mx.full((H,), 2.0, mx.bfloat16),
        "lm_head.weight": r((256, H), 17),
    }
    for l in range(2):
        p = f"model.language_model.layers.{l}"
        w[f"{p}.input_layernorm.weight"] = mx.full((H,), 2.0, mx.bfloat16)
        w[f"{p}.post_attention_layernorm.weight"] = mx.full(
            (H,), 2.0, mx.bfloat16
        )
        w[f"{p}.mlp.experts.gate_up_proj"] = r((E, 2 * I, H), 10 + l)
        w[f"{p}.mlp.experts.down_proj"] = r((E, H, I), 11 + l)
        w[f"{p}.mlp.gate.weight"] = r((E, H), 12 + l)
        w[f"{p}.mlp.shared_expert.gate_proj.weight"] = r((I, H), 13 + l)
        w[f"{p}.mlp.shared_expert.up_proj.weight"] = r((I, H), 14 + l)
        w[f"{p}.mlp.shared_expert.down_proj.weight"] = r((H, I), 15 + l)
        w[f"{p}.mlp.shared_expert_gate.weight"] = r((1, H), 16 + l)
        if l == 0:  # linear attention (gated delta net)
            w[f"{p}.linear_attn.in_proj_qkv.weight"] = r(
                (2 * kdim + vdim, H), 1
            )
            w[f"{p}.linear_attn.in_proj_z.weight"] = r((vdim, H), 2)
            w[f"{p}.linear_attn.in_proj_b.weight"] = r((nv, H), 3)
            w[f"{p}.linear_attn.in_proj_a.weight"] = r((nv, H), 4)
            w[f"{p}.linear_attn.conv1d.weight"] = r((conv_dim, 1, 4), 5)
            w[f"{p}.linear_attn.A_log"] = r((nv,), 6).astype(mx.float32)
            w[f"{p}.linear_attn.dt_bias"] = r((nv,), 7).astype(mx.float32)
            w[f"{p}.linear_attn.norm.weight"] = r((hd,), 8)
            w[f"{p}.linear_attn.out_proj.weight"] = r((H, vdim), 9)
        else:  # full attention (GQA, head_dim 64)
            w[f"{p}.self_attn.q_proj.weight"] = r((2 * 2 * 64, H), 20)  # attn_output_gate doubles q
            w[f"{p}.self_attn.k_proj.weight"] = r((1 * 64, H), 21)
            w[f"{p}.self_attn.v_proj.weight"] = r((1 * 64, H), 22)
            w[f"{p}.self_attn.o_proj.weight"] = r((H, 2 * 64), 23)
            w[f"{p}.self_attn.q_norm.weight"] = r((64,), 24)
            w[f"{p}.self_attn.k_norm.weight"] = r((64,), 25)
    return w


MINI_TEXT_CONFIG = {
    "hidden_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "head_dim": 64,
    "num_key_value_heads": 1,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 32,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "vocab_size": 256,
    "rms_norm_eps": 1e-5,
    "layer_types": ["linear_attention", "full_attention"],
    "full_attention_interval": 2,
    "tie_word_embeddings": False,
    "mtp_num_hidden_layers": 1,
}


def test_real_mlx_lm_loader_roundtrip():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
    import mlx.nn as nn

    src = _dim_consistent_source()
    out = q.transform_shard(src, bits=4, group=64, has_mtp=True)

    model = Model(ModelArgs.from_dict({
        "model_type": "qwen3_5_moe",
        "text_config": dict(MINI_TEXT_CONFIG),
    }))
    qcfg = q.output_quantization_config(num_layers=2, bits=4, group=64)

    def class_predicate(p, m):
        if p in qcfg:
            return qcfg[p]
        if not hasattr(m, "to_quantized"):
            return False
        return f"{p}.scales" in out

    nn.quantize(
        model, group_size=64, bits=4, mode="affine",
        class_predicate=class_predicate,
    )
    # strict load: any key/shape mismatch in the artifact fails here
    model.load_weights(list(out.items()))

    logits = model(mx.array([[1, 2, 3]]))
    mx.eval(logits)
    assert logits.shape == (1, 3, 256)
    assert bool(mx.isfinite(logits).all())
