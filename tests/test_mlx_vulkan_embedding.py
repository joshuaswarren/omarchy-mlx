# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""PT-T1 regression: the embedding lowering takes both call orders.

F.embedding(input, weight) and aten.embedding(weight, indices) name the same
two tensors in opposite orders; the single-order lowering served to the aten
name produced a transposed-garbage tensor and crashed the first training run
(G708 PT-T1, 128-vs-384 broadcast failure).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")


def test_embedding_lowerings_agree_for_both_orders(mlx_vulkan):
    weight = torch.randn(5, 4)
    idx = torch.tensor([[3, 1], [0, 2]])
    reference = np.asarray(weight.numpy())[idx.numpy()]
    for lowering in (mlx_vulkan._embedding_f, mlx_vulkan._embedding,
                     mlx_vulkan.ATEN_TABLE["embedding"]):
        out = lowering(idx, weight) if lowering is mlx_vulkan._embedding_f \
            else lowering(weight, idx)
        assert out.shape == (2, 2, 4)
        assert np.array_equal(np.asarray(out), reference)


def test_aten_table_uses_the_aten_order(mlx_vulkan):
    weight = torch.randn(7, 3)
    idx = torch.tensor([[6, 0]])
    out = mlx_vulkan.ATEN_TABLE["embedding"](weight, idx)
    assert out.shape == (1, 2, 3)
    assert np.array_equal(
        np.asarray(out), np.asarray(weight.numpy())[idx.numpy()])


def test_embedding_dense_backward_freq_pad_nan(mlx_vulkan):
    # indices with repeats + the padding row; gout carries NaN at the padded
    # index - it must not leak into the weight grads.
    nw, d = 8, 4
    idx = np.array([3, 3, 1, 6, 2, 3])
    gout = np.arange(6 * d, dtype=np.float32).reshape(6, d)
    gout[3] = np.nan  # the padded occurrence (idx 6)

    def ref(scale_by_freq, pad):
        out = np.zeros((nw, d), dtype=np.float32)
        keep = idx != pad
        np.add.at(out, idx[keep], gout[keep])
        if scale_by_freq:
            cnt = np.bincount(idx[keep], minlength=nw).astype(np.float32)
            out = out / np.maximum(cnt, 1.0)[:, None]
        return out

    got = np.asarray(mlx_vulkan.ATEN_TABLE["embedding_dense_backward"](
        mx.array(gout), mx.array(idx), nw, 6, False))
    assert np.allclose(got, ref(False, 6)), "plain accumulation differs"

    got = np.asarray(mlx_vulkan.ATEN_TABLE["embedding_dense_backward"](
        mx.array(gout), mx.array(idx), nw, 6, True))
    assert np.allclose(got, ref(True, 6)), "scale_grad_by_freq differs"

    # NaN at a NON-padded index is a real gradient and must survive
    got = np.asarray(mlx_vulkan.ATEN_TABLE["embedding_dense_backward"](
        mx.array(gout), mx.array(idx), nw, -1, False))
    assert np.isnan(got[6]).all(), "NaN at a used index must propagate"
    assert np.isfinite(got[:6]).all() and np.isfinite(got[7:]).all()
