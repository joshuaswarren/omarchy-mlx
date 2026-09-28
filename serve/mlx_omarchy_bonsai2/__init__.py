"""Bonsai-2 (prism_hadamard_qwen35) text serving for mlx-omarchy.

Serves prism-ml/Ternary-Bonsai-2-27B-mlx-2bit style packs: a Qwen3.5 hybrid
(GDN + attention) language model whose projections are stored as Hadamard-
folded ternary affine 2-bit group-128 quantized matrices. The pack's own
remote runtime is never imported — Hadamard + 2-bit affine semantics are
ported in-repo (packed.py).
"""
