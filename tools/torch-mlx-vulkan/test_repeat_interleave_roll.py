#!/usr/bin/env python3
# PT-AL-i: aten.repeat_interleave (per-element repeats + dim arg, plus the
# scalar and repeats-only overloads) and aten.roll — fp32/int parity vs
# torch CPU across dims, shifts and the dims=[] schema-default form.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan

SPEC = mlx_vulkan._FUNC_SPECS


def test_repeat_interleave_self_int():
    rng = np.random.RandomState(7)
    x = rng.randn(3, 4, 2).astype(np.float32)
    for dim in (None, 0, 1, 2, -1, -2):
        for rep in (1, 2, 3):
            got = SPEC["repeat_interleave"](mlx_vulkan._mx(x), rep,
                                            **({"dim": dim} if dim is not None else {}))
            want = torch.repeat_interleave(torch.tensor(x), rep,
                                           **({"dim": dim} if dim is not None else {}))
            g, w = np.array(got, dtype=np.float32), want.numpy()
            assert g.shape == w.shape and np.array_equal(g, w), (dim, rep, g.shape, w.shape)
    print("test_repeat_interleave_self_int PASS: 3x4x2 x dims x reps")


def test_repeat_interleave_self_tensor():
    rng = np.random.RandomState(13)
    x = rng.randn(2, 5).astype(np.float32)
    cases = [
        (np.array([0, 1, 2, 3, 1]), 1),
        (np.array([2, 2]), 0),
        (np.array([0, 0, 0, 0, 0]), 1),
        (np.array([4, 0, 1, 0, 2]), -1),
    ]
    for reps, dim in cases:
        got = SPEC["repeat_interleave"](mlx_vulkan._mx(x), mlx_vulkan._mx(reps.astype(np.int64)), dim)
        want = torch.repeat_interleave(torch.tensor(x), torch.tensor(reps), dim=dim)
        g, w = np.array(got, dtype=np.float32), want.numpy()
        assert g.shape == w.shape and np.array_equal(g, w), (reps.tolist(), dim, g.shape, w.shape)
    print("test_repeat_interleave_self_tensor PASS: per-element reps incl. zeros")


def test_repeat_interleave_self_tensor_output_size():
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    reps = np.array([1, 0, 2])
    got = SPEC["repeat_interleave"](mlx_vulkan._mx(x), mlx_vulkan._mx(reps), 1, output_size=3)
    want = torch.repeat_interleave(torch.tensor(x), torch.tensor(reps), dim=1, output_size=3)
    assert np.array_equal(np.array(got), want.numpy())
    print("test_repeat_interleave_self_tensor_output_size PASS")


def test_repeat_interleave_tensor_overload():
    reps = np.array([2, 0, 3, 1])
    got = SPEC["repeat_interleave"](mlx_vulkan._mx(reps.astype(np.int64)))
    want = torch.repeat_interleave(torch.tensor(reps))
    assert np.array_equal(np.array(got), want.numpy()), (np.array(got), want.numpy())
    print("test_repeat_interleave_tensor_overload PASS: arange indices")


def test_repeat_interleave_int_input():
    x = np.array([[5, -3], [7, 9]], dtype=np.int32)
    reps = np.array([2, 1])
    got = SPEC["repeat_interleave"](mlx_vulkan._mx(x), mlx_vulkan._mx(reps), 0)
    want = torch.repeat_interleave(torch.tensor(x), torch.tensor(reps), dim=0)
    g, w = np.array(got), want.numpy()
    assert g.dtype == w.dtype and np.array_equal(g, w), (g, w)
    print("test_repeat_interleave_int_input PASS: int32 preserved")


def test_roll_flatten_forms():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    for shifts, dims in ((1, None), (-2, None), (5, [])):
        kw = {"dims": dims} if dims is not None else {}
        got = SPEC["roll"](mlx_vulkan._mx(x), shifts, **kw)
        want = torch.roll(torch.tensor(x), shifts)
        g, w = np.array(got), want.numpy()
        assert g.shape == w.shape and np.array_equal(g, w), (shifts, dims, g, w)
    print("test_roll_flatten_forms PASS: dims=None and dims=[] flatten")


def test_roll_dimmed_forms():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    cases = [
        (1, 0), (-1, 1), (2, -1),
        ([1, 2], [0, 2]), ([-1, -1], [1, 2]),
    ]
    for shifts, dims in cases:
        got = SPEC["roll"](mlx_vulkan._mx(x), shifts, dims)
        want = torch.roll(torch.tensor(x), shifts, dims)
        g, w = np.array(got), want.numpy()
        assert g.shape == w.shape and np.array_equal(g, w), (shifts, dims, g, w)
    print("test_roll_dimmed_forms PASS: scalar + tuple shift/dim pairs")


if __name__ == "__main__":
    test_repeat_interleave_self_int()
    test_repeat_interleave_self_tensor()
    test_repeat_interleave_self_tensor_output_size()
    test_repeat_interleave_tensor_overload()
    test_repeat_interleave_int_input()
    test_roll_flatten_forms()
    test_roll_dimmed_forms()
