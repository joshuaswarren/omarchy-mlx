"""Q-65 audit: every tuple-returning lowering in mlx_vulkan.py returns
(values, indices) whose shapes equal torch's for keepdim on and off, across
every dim, on 1-D and multi-D inputs — plus the held-value contract that the
returned index names a position holding the returned value along dim.

Runs the ATEN_TABLE entries positionally, the way the executor calls them,
so the table wiring is what gets audited. Median and sort are pending
landings (their specs do not exist in the table yet); the matrix notes them
rather than fabricating coverage.
"""
import numpy as np
import torch

import mlx.core as mx
import mlx_vulkan


def _np(x):
    return np.array(mx.array(x))


def _held(x, vals, idx, dim):
    """The returned index names a position along dim holding the returned value."""
    s = np.asarray(x)
    i = _np(idx).astype(np.int64)
    while i.ndim < s.ndim:
        i = np.expand_dims(i, axis=dim % s.ndim)
    got = np.take_along_axis(s, i, axis=dim % s.ndim)
    return np.array_equal(got.reshape(_np(vals).shape), _np(vals).astype(s.dtype))


fails = []


def check(name, tp, mp, x=None, dim=None):
    """tp/mp are torch/mlx (values, indices) pairs."""
    tshape = (tuple(tp[0].shape), tuple(tp[1].shape))
    mshape = (tuple(_np(mp[0]).shape), tuple(_np(mp[1]).shape))
    ok = mshape == tshape
    if ok and x is not None:
        ok = _held(x, mp[0], mp[1], dim)
    if ok:
        print(f"ok   {name}")
    else:
        fails.append(name)
        print(f"FAIL {name}: torch {tshape} vs mlx {mshape}")


T = mlx_vulkan.ATEN_TABLE
rng = np.random.RandomState(7)

# dim-returning forms: every dim x keepdim on/off x {1-D, 2-D, 3-D}
for shape in [(7,), (4, 9), (2, 3, 5)]:
    x = rng.randint(0, 5, size=shape).astype(np.float32)
    xt = torch.tensor(x)
    for d in range(len(shape)):
        for keepdim in (True, False):
            check(f"mode {shape} dim={d} keepdim={keepdim}",
                  torch.mode(xt, dim=d, keepdim=keepdim),
                  T["mode"](mx.array(x), d, keepdim), x, d)
            check(f"kthvalue {shape} dim={d} keepdim={keepdim}",
                  torch.kthvalue(xt, 2, dim=d, keepdim=keepdim),
                  T["kthvalue"](mx.array(x), 2, d, keepdim), x, d)
            check(f"max {shape} dim={d} keepdim={keepdim}",
                  torch.max(xt, dim=d, keepdim=keepdim),
                  T["max"](mx.array(x), d, keepdim), x, d)
            check(f"min {shape} dim={d} keepdim={keepdim}",
                  torch.min(xt, dim=d, keepdim=keepdim),
                  T["min"](mx.array(x), d, keepdim), x, d)
    for d in list(range(len(shape))) + [-1]:
        k = min(3, shape[d % len(shape)])
        check(f"topk {shape} dim={d}",
              torch.topk(xt, k, dim=d), T["topk"](mx.array(x), k, d), x, d)

# flat (no-dim) overloads: torch returns scalar values + scalar flat index
for shape in [(7,), (4, 9)]:
    x = rng.randint(0, 5, size=shape).astype(np.float32)
    xt = torch.tensor(x)
    for name, torch_fn, margs, targs in [
        ("mode", torch.mode, (), ()),
        ("kthvalue", torch.kthvalue, (2,), (2,)),
        ("max", torch.max, (), ()),
        ("min", torch.min, (), ()),
    ]:
        tp = torch_fn(xt, *targs)
        mp = T[name](mx.array(x), *margs)
        tv, ti = tp if isinstance(tp, tuple) else (tp, None)
        mv, mi = mp if isinstance(mp, tuple) else (mp, None)
        tshape = (tuple(tv.shape), None if ti is None else tuple(ti.shape))
        mshape = (tuple(_np(mv).shape), None if mi is None else tuple(_np(mi).shape))
        if mshape == tshape:
            print(f"ok   {name} flat {shape}")
        else:
            fails.append(f"{name} flat {shape}")
            print(f"FAIL {name} flat {shape}: torch {tshape} vs mlx {mshape}")

# median and sort are pending landings: audit them when their spec exists
for pending in ("median", "sort", "msort"):
    state = "landed - extend the matrix above" if pending in T else "not in ATEN_TABLE yet"
    print(f"note {pending}: {state}")

print(f"{len(fails)} failures" if fails else "ALL PASS")
raise SystemExit(1 if fails else 0)
