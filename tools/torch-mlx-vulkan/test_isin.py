#!/usr/bin/env python3
# PT-AL-i: aten.isin / in1d lowering — broadcast-compare membership parity
# vs torch CPU across element/test shapes, dtypes, invert, empty and
# scalar test sets, and the in1d alias.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _check(x, t, invert=False, assume_unique=False, key="isin"):
    xt = torch.as_tensor(np.asarray(x))
    tt = torch.as_tensor(np.asarray(t))
    got = mlx_vulkan._FUNC_SPECS[key](mlx_vulkan._mx(xt), mlx_vulkan._mx(tt),
                                    assume_unique=assume_unique, invert=invert)
    want = torch.isin(xt, tt, assume_unique=assume_unique, invert=invert)
    g = np.array(got)
    w = want.numpy()
    assert g.shape == w.shape, (x, t, g.shape, w.shape)
    assert g.dtype == np.bool_, g.dtype
    assert np.array_equal(g, w), (x, t, invert, g.tolist(), w.tolist())


def test_isin():
    rng = np.random.RandomState(7)

    # 1-D int elements, hits and misses, duplicates in the test set
    _check([1, 2, 3, 4, 5, 3, 2], [2, 4, 4, 2])
    # 2-D float elements vs a 1-D test set
    _check(rng.randn(3, 5).astype(np.float32), rng.randn(4).astype(np.float32))
    # 2-D test set: membership is over the whole set, not per-row
    _check(rng.randint(0, 10, (2, 4)), rng.randint(0, 10, (3, 3)))
    # scalar (0-d) test element
    _check([5, 7, 9], np.int64(7))
    # empty test set -> all False; empty elements -> empty bool
    _check([1, 2, 3], np.array([], dtype=np.int64))
    _check(np.array([], dtype=np.float32).reshape(0, 3), [1.0, 2.0])
    # invert
    _check([1, 2, 3, 4], [2, 4], invert=True)
    # assume_unique is a hint only — same answer with duplicates present
    _check([1, 2, 2, 3], [2, 3, 3], assume_unique=True)
    # no hits / all hits
    _check([1, 2, 3], [9, 8, 7])
    _check([2, 4, 2], [2, 4])
    # NaN matches nothing, per IEEE and torch
    _check([1.0, np.nan, 3.0], [np.nan, 1.0])
    # the legacy in1d alias lowers through the same spec
    _check([3, 1, 4], [1, 4], key="in1d")

    print("test_isin PASS: shapes, dtypes, invert, empty/scalar test sets, in1d alias")


if __name__ == "__main__":
    test_isin()
