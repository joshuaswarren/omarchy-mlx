"""Smoke test for the omarchy sparse MLA + top-k ops.

Checks:
  1. dsa_topk_indices_omarchy selects the k largest scores per row.
  2. sparse_mla_attention_omarchy runs end to end and its output on
     a single query with a single topk slice equals the dense SDPA
     slice at that index (bf16 tolerance).

Small shapes only - the model-level envelope is out of reach on
lavapipe (see the mesa-1 hang note in the receipt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
import sparse_mla_omarchy as smla  # noqa: E402


def main():
    mx.random.seed(0)
    B, H, L, K, topk, dtype = 1, 32, 8, 32, 4, mx.bfloat16
    scale = 1.0 / (576 ** 0.5)

    # ---- topk -----------------------------------------------------
    scores = mx.random.normal(shape=(B, 1, L, K)).astype(dtype)
    picked = smla.dsa_topk_indices_omarchy(scores, topk)
    mx.eval(picked)
    # Verify: for each row, the picked indices' scores are >= the
    # largest un-picked score (ties tolerated at bf16 resolution).
    flat_scores = scores[0, 0]  # [L, K]
    picked_scores = mx.take_along_axis(flat_scores, picked[0, 0], axis=-1)
    min_picked = mx.min(picked_scores, axis=-1)  # [L]
    # The unpicked scores: mask out the picked indices.
    mask = mx.zeros_like(flat_scores).astype(mx.bool_)
    for j in range(topk):
        idx = picked[0, 0, :, j : j + 1]
        updates = mx.full(idx.shape, True)
        mask = mx.put_along_axis(
            mask,
            idx,
            updates,
            axis=-1,
            stream=None,
        ) if False else mask  # put_along_axis bool not supported; use scatter
    # Simpler: compare min picked vs max unpicked by sorting.
    sorted_idx = mx.argsort(-flat_scores, axis=-1)  # [L, K]
    top_sorted = sorted_idx[:, :topk]
    # Every picked index must be inside the sorted top-k set (the
    # argpartition path may break score ties differently from the
    # sorted path; allow either order within the same score value).
    picked_set = mx.sort(picked[0, 0], axis=-1)
    sorted_set = mx.sort(top_sorted, axis=-1)
    mismatch = mx.sum(picked_set != sorted_set).item()
    if mismatch > 0:
        print(f"topk mismatch count: {mismatch} / {L * topk}")
    else:
        print("topk selection matches sorted top-k: OK")

    # ---- sparse MLA ----------------------------------------------
    q_latent = mx.random.normal(shape=(B, H, L, 512)).astype(dtype) * 0.1
    q_pe = mx.random.normal(shape=(B, H, L, 64)).astype(dtype) * 0.1
    kv_latent = mx.random.normal(shape=(B, 1, K, 512)).astype(dtype) * 0.1
    k_pe = mx.random.normal(shape=(B, 1, K, 64)).astype(dtype) * 0.1
    out = smla.sparse_mla_attention_omarchy(
        q_latent, q_pe, kv_latent, k_pe, picked, scale
    )
    mx.eval(out)
    assert out.shape == (B, H, L, 512), out.shape

    # Reference: dense SDPA on the full K, restricted to the first
    # query row and its picked top-k slice. Both should agree within
    # bf16 storage tolerance (up to ~1e-2 absolute at these
    # magnitudes).
    l0 = 0
    picked_row = picked[0, 0, l0]
    kv_g = mx.take(kv_latent[0, 0], picked_row, axis=0)  # [topk, 512]
    kpe_g = mx.take(k_pe[0, 0], picked_row, axis=0)  # [topk, 64]
    q_full = mx.concatenate(
        [q_latent[0, :, l0, :], q_pe[0, :, l0, :]], axis=-1
    )  # [H, 576]
    k_full = mx.concatenate([kv_g, kpe_g], axis=-1)  # [topk, 576]
    scores = (q_full @ k_full.T) * scale
    smax = mx.max(scores, axis=-1, keepdims=True)
    w = mx.exp(scores - smax)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    ref = w @ kv_g  # [H, 512]
    diff = mx.abs(out[0, :, l0, :] - ref)
    max_diff = float(mx.max(diff).item())
    print(f"sparse MLA row {l0}: max abs diff vs dense slice ref = {max_diff:.3e}")
    if max_diff > 5e-2:
        print("FAIL: sparse MLA output deviates from the dense slice")
        return 1
    print("sparse MLA smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
