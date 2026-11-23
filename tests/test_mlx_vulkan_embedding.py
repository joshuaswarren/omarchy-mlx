# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""PT-T1 regression: the embedding lowering takes both call orders.

F.embedding(input, weight) and aten.embedding(weight, indices) name the same
two tensors in opposite orders; the single-order lowering served to the aten
name produced a transposed-garbage tensor and crashed the first training run
(G708 PT-T1, 128-vs-384 broadcast failure).
"""
import numpy as np
import torch


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
