# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""G708/PT-T1 regression: the embedding lowering must take both orders.

F.embedding(input, weight) and aten.embedding(weight, indices) name the same
two tensors in opposite orders; the single-order lowering served to the aten
name produced a transposed-garbage tensor and crashed the first training run
(G708 PT-T1, 128-vs-384 broadcast failure).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")

spec = importlib.util.spec_from_file_location(
    "mlx_vulkan_under_test",
    Path(__file__).resolve().parents[1] / "tools" / "torch-mlx-vulkan"
    / "mlx_vulkan.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)


def test_embedding_lowerings_agree_for_both_orders():
    weight = torch.randn(5, 4)
    idx = torch.tensor([[3, 1], [0, 2]])
    reference = np.asarray(weight.numpy())[idx.numpy()]
    for lowering in (mv._embedding_f, mv._embedding,
                     mv.ATEN_TABLE["embedding"]):
        out = lowering(idx, weight) if lowering is mv._embedding_f \
            else lowering(weight, idx)
        assert out.shape == (2, 2, 4)
        assert np.array_equal(np.asarray(out), reference)


def test_aten_table_uses_the_aten_order():
    weight = torch.randn(7, 3)
    idx = torch.tensor([[6, 0]])
    out = mv.ATEN_TABLE["embedding"](weight, idx)
    assert out.shape == (1, 2, 3)
    assert np.array_equal(
        np.asarray(out), np.asarray(weight.numpy())[idx.numpy()])
