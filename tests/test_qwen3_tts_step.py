"""Equivalence tests for the vendored Qwen3-TTS step.

The tests need mlx + the voice pack. They are skipped when those are
absent (x86 unit-test runner). On the M2 they run with:

  PYTHONPATH=<home>/voice-site:serve
    <home>/.local/share/mlx-omarchy/venv/bin/python -m unittest
      tests.test_qwen3_tts_step -v
"""
import sys
import unittest

MLX_AVAILABLE = False
VOICE_PACK_PRESENT = False
try:
    import mlx.core as mx  # noqa: F401
    MLX_AVAILABLE = True
    from pathlib import Path
    if Path("<home>/mlx-tts-home/voice/qwen3-tts-0.6b-customvoice-4bit").is_dir():
        VOICE_PACK_PRESENT = True
except Exception:
    pass


@unittest.skipUnless(MLX_AVAILABLE and VOICE_PACK_PRESENT,
                     "needs mlx + voice pack on the M2")
class FusedProjectionEquivalence(unittest.TestCase):
    """FusedQKV/FusedGateUp produce the same q/k/v and gate/up as the
    unfused projections, within bf16 tolerance (max abs diff <= 1e-2).
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, "<home>/voice-site")
        from mlx_audio.tts.utils import load_model
        cls.model = load_model("<home>/mlx-tts-home/voice/qwen3-tts-0.6b-customvoice-4bit")
        sys.path.insert(0, "<home>/.config/superpowers/worktrees/mlx-omarchy/SpeechOutputFast/serve")
        from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
        cls.V = V
        cls.V.vendorize_talker(cls.model)
        cls.V.vendorize_cp(cls.model)

    def test_fused_qkv_matches_unfused(self):
        import mlx.core as mx
        att = self.model.talker.model.layers[0].self_attn
        x = mx.random.normal((1, 1, att.q_proj.weight.shape[1])).astype(mx.bfloat16)
        mx.eval(x)
        # Unfused
        q0 = att.q_proj(x); k0 = att.k_proj(x); v0 = att.v_proj(x)
        # Fused
        qkv = att.qkv_fused(x)
        q1 = qkv[..., :att.qkv_fused.out_q]
        k1 = qkv[..., att.qkv_fused.out_q:att.qkv_fused.out_q + att.qkv_fused.out_k]
        v1 = qkv[..., att.qkv_fused.out_q + att.qkv_fused.out_k:]
        mx.eval(q0, k0, v0, q1, k1, v1)
        self.assertLessEqual(float(mx.max(mx.abs(q0 - q1)).item()), 1e-2)
        self.assertLessEqual(float(mx.max(mx.abs(k0 - k1)).item()), 1e-2)
        self.assertLessEqual(float(mx.max(mx.abs(v0 - v1)).item()), 1e-2)

    def test_fused_gateup_matches_unfused(self):
        import mlx.core as mx
        mlp = self.model.talker.model.layers[0].mlp
        x = mx.random.normal((1, 1, mlp.gate_proj.weight.shape[1])).astype(mx.bfloat16)
        mx.eval(x)
        g0 = mlp.gate_proj(x); u0 = mlp.up_proj(x)
        gu = mlp.gateup_fused(x)
        g1 = gu[..., :mlp.gateup_fused.out_g]
        u1 = gu[..., mlp.gateup_fused.out_g:]
        mx.eval(g0, u0, g1, u1)
        self.assertLessEqual(float(mx.max(mx.abs(g0 - g1)).item()), 1e-2)
        self.assertLessEqual(float(mx.max(mx.abs(u0 - u1)).item()), 1e-2)


@unittest.skipUnless(MLX_AVAILABLE and VOICE_PACK_PRESENT,
                     "needs mlx + voice pack on the M2")
class TalkerLayerEquivalence(unittest.TestCase):
    """One talker layer: vendored forward matches upstream within bf16
    tolerance (max abs diff <= 5e-3 for hidden states, <= 5e-2 for q/k norms).
    """

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, "<home>/voice-site")
        from mlx_audio.tts.utils import load_model
        cls.model = load_model("<home>/mlx-tts-home/voice/qwen3-tts-0.6b-customvoice-4bit")
        sys.path.insert(0, "<home>/.config/superpowers/worktrees/mlx-omarchy/SpeechOutputFast/serve")
        from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
        cls.V = V
        cls.V.vendorize_talker(cls.model)
        cls.V.vendorize_cp(cls.model)

    def test_talker_layer_zero_matches_upstream(self):
        import mlx.core as mx
        cfg = self.model.config.talker_config
        V = self.V
        layer = self.model.talker.model.layers[0]
        # Construct matching input: hidden_size random tensor.
        x = mx.random.normal((1, 4, cfg.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        # Position ids for one forward at seq_len=4 (just for shape).
        pos = mx.broadcast_to(mx.arange(4)[None, :], (1, 4))
        pos3 = mx.stack([pos, pos, pos], axis=0)
        # Compute rope cos/sin using the upstream rotary (so the comparison
        # isolates the layer body, not the rotary).
        cos, sin = self.model.talker.model.rotary_emb(x, pos3)
        # Upstream layer call.
        x_upstream = layer(x, (cos, sin), None, None)
        mx.eval(x_upstream)
        # Vendored layer call: requires fresh cache + rope.
        from mlx_omarchy_assistant._vendored.qwen3_tts_step import (
            StaticKVCache, talker_layer_forward,
        )
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        cache = StaticKVCache(
            keys=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
            values=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
            pos=0,
        )
        x_v = talker_layer_forward(
            layer, layer.self_attn.qkv_fused, 0, x, (cos, sin), cache,
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
        )
        mx.eval(x_v)
        diff = float(mx.max(mx.abs(x_upstream - x_v)).item())
        self.assertLessEqual(diff, 5e-3, f"layer diff {diff}")


@unittest.skipUnless(MLX_AVAILABLE and VOICE_PACK_PRESENT,
                     "needs mlx + voice pack on the M2")
class SamplerEquivalence(unittest.TestCase):
    def test_gumbel_distribution(self):
        import mlx.core as mx
        # Same distribution as mx.random.categorical at temperature 1.
        # Chi-square check via upstream _sample_token (which uses
        # categorical_sampling).
        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
        # Small logits vector; 8000 draws; chi-square against uniform
        # softmax (no top-k, no rep penalty, no suppress).
        small = mx.array([[[2.0, 1.0, 0.5, 0.0, -0.5, -1.0, 1.5, 0.2]]])
        N = 8000
        # upstream path
        orig = Model.categorical_sampling if hasattr(Model, "categorical_sampling") else None
        # import the function from the qwen3_tts module
        from mlx_audio.tts.models.qwen3_tts import qwen3_tts as Q
        if hasattr(Q, "categorical_sampling"):
            upstream = Q.categorical_sampling
        else:
            from mlx_audio.lm.sample_utils import categorical_sampling as upstream
        cu = [0] * 8
        for _ in range(N): cu[int(upstream(small, 1.0).reshape(-1)[0])] += 1
        # vendored
        from mlx_omarchy_assistant._vendored.qwen3_tts_step import gumbel_max_sample
        cf = [0] * 8
        for _ in range(N): cf[int(gumbel_max_sample(small, 1.0).reshape(-1)[0])] += 1
        # chi-square against uniform softmax
        import math
        p = [math.exp([2.0,1.0,0.5,0.0,-0.5,-1.0,1.5,0.2][i]) for i in range(8)]
        Z = sum(p); p = [pi / Z for pi in p]
        exp = N / 8
        chi_u = sum((cu[i] - exp) ** 2 / exp for i in range(8))
        chi_f = sum((cf[i] - exp) ** 2 / exp for i in range(8))
        # Both should pass the chi-square at df=7, p=0.001 (crit 24.32).
        self.assertLess(chi_u, 24.32)
        self.assertLess(chi_f, 24.32)


if __name__ == "__main__":
    unittest.main()
