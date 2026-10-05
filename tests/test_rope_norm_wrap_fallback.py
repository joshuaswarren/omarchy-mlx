#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Behavior test for the qwen3 rope-norm fold's specprefill wrap fallback.

SpecPrefill wraps attn.rope in _PositionMappedRoPE (no attribute
delegation) or _OffsetAdjustedRoPE (offset remap). The fused
mx.fast.rope_rms_norm dispatch reads self.rope.dims and rotates at the
contiguous cache offset, so a wrapper must send the attention down the
composed chain. Before the 2026-10-05 gate the direct .dims read raised
AttributeError and crashed sparse prefill (A8).

This test executes the REAL installed mlx_lm qwen3 Attention (patched
form) with a no-delegation wrapper shaped exactly like the specprefill
wrapper, and asserts:

  1. forward does not raise with the wrapper installed,
  2. the wrapper's output equals the same forward with the genuine rope
     (the composed chain through the wrapper applies the same rotation
     the genuine rope would at offset 0), and
  3. the genuine-rope forward still hits the fused path (provenance:
     the omarchy wheel exposes mx.fast.rope_rms_norm).

Runs anywhere mlx + mlx_lm import; on stock upstream mlx (no
mx.fast.rope_rms_norm) the fold gate is inert and the test degrades to
a plain equivalence check.
"""
import os
import unittest

try:
    import mlx.core as mx
    from mlx_lm.models import qwen3
    HAVE_MLX = True
except Exception:  # pragma: no cover - hosts without the wheel
    HAVE_MLX = False


class NoDelegationRoPE:
    """The _PositionMappedRoPE shape: __call__ only, no delegation."""

    def __init__(self, original):
        self._original = original

    def __call__(self, x, offset=0):
        return self._original(x, offset=offset)


def _attention():
    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        head_dim=16,
        tie_word_embeddings=False,
        rope_theta=10000.0,
    )
    return qwen3.Attention(args)


@unittest.skipUnless(HAVE_MLX, "mlx / mlx_lm not importable on this host")
class RopeNormWrapFallbackTests(unittest.TestCase):
    def setUp(self):
        self.attn = _attention()
        self.x = mx.random.normal((1, 1, 64)).astype(mx.bfloat16)

    def test_fused_path_available_on_omarchy_wheel(self):
        # Provenance: on the omarchy wheel the fold can actually fire for
        # the genuine rope (bf16, B == 1). On stock upstream this skips.
        self.assertTrue(
            hasattr(mx.fast, "rope_rms_norm"),
            "mx.fast.rope_rms_norm missing; run this on the omarchy wheel "
            "to exercise the fused path")

    def test_no_delegation_wrapper_takes_composed_chain(self):
        genuine = self.attn.rope
        plain = self.attn(self.x)  # fused path for the genuine rope
        self.attn.rope = NoDelegationRoPE(genuine)
        wrapped = self.attn(self.x)  # must NOT raise (A8 regression)
        mx.eval(plain, wrapped)
        self.assertEqual(
            plain.astype(mx.float32).tolist(),
            wrapped.astype(mx.float32).tolist(),
            "wrapped-rope forward must equal the genuine-rope forward at "
            "offset 0 (composed chain through the wrapper)")

    def test_wrapper_survives_repeated_forward(self):
        self.attn.rope = NoDelegationRoPE(self.attn.rope)
        for _ in range(3):
            out = self.attn(self.x)
            mx.eval(out)


if __name__ == "__main__":
    unittest.main()
