#!/usr/bin/env python3
"""The order-statistics family tests: the shim vs torch, GPU only, no CPU fallback.

Every case runs the aten op through the shim's FUNC_TABLE dispatch (the same
path the executor takes) and compares against torch's reference on the same
inputs. The median/without-dim and the kthvalue/without-dim return scalars; the
named-dim variants return (values, indices) tuples like aten does."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx  # noqa: E402
import torch  # noqa: E402

import mlx_vulkan  # noqa: E402

CASES = []


def case(name, torch_fn, shim_fn, check):
    CASES.append((name, torch_fn, shim_fn, check))


def approx(t, m):
    import numpy as np
    return np.allclose(np.array(t, dtype="float32"), np.array(mx.array(m), dtype="float32"))


def eq(t, m):
    import numpy as np
    return (np.array(t, dtype="int64") == np.array(mx.array(m), dtype="int64")).all()


def values_of(x):
    # aten's tuple-returning overloads hand back namedtuples; a plain
    # tensor's .values is the sparse-tensor builtin, not a result part.
    return x.values if isinstance(x, tuple) else x


def indices_of(x):
    return x.indices if isinstance(x, tuple) else None


a = torch.arange(20, dtype=torch.float32) % 7 - 2  # repeats, negatives, unordered
b = torch.randn(4, 6)

case("median flat", lambda: torch.median(a), lambda: mlx_vulkan._FUNC_SPECS["median"](mx.array(a)),
     lambda t, m: approx(values_of(t), m))
case("median dim0 keepdim", lambda: torch.median(b, dim=0, keepdim=True),
     lambda: mlx_vulkan._FUNC_SPECS["median"](mx.array(b), dim=0, keepdim=True),
     lambda t, m: approx(t.values, m[0]) and eq(t.indices, m[1]))
case("median dim-1", lambda: torch.median(b, dim=-1),
     lambda: mlx_vulkan._FUNC_SPECS["median"](mx.array(b), dim=-1),
     lambda t, m: approx(t.values, m[0]) and eq(t.indices, m[1]))
case("mode flat", lambda: torch.mode(a), lambda: mlx_vulkan._FUNC_SPECS["mode"](mx.array(a)),
     lambda t, m: approx(t.values, m[0]))
case("mode dim-1", lambda: torch.mode(b, dim=-1),
     lambda: mlx_vulkan._FUNC_SPECS["mode"](mx.array(b), dim=-1),
     lambda t, m: approx(t.values, m[0]))
case("kthvalue k=3 flat", lambda: torch.kthvalue(a, 3), lambda: mlx_vulkan._FUNC_SPECS["kthvalue"](mx.array(a), 3),
     lambda t, m: approx(values_of(t), m[0]))
case("kthvalue k=2 dim0", lambda: torch.kthvalue(b, 2, dim=0),
     lambda: mlx_vulkan._FUNC_SPECS["kthvalue"](mx.array(b), 2, dim=0),
     lambda t, m: approx(t.values, m[0]) and eq(t.indices, m[1]))
case("kthvalue k=5 dim-1 keepdim", lambda: torch.kthvalue(b, 5, dim=-1, keepdim=True),
     lambda: mlx_vulkan._FUNC_SPECS["kthvalue"](mx.array(b), 5, dim=-1, keepdim=True),
     lambda t, m: approx(t.values, m[0]))

fails = 0
for name, torch_fn, shim_fn, check in CASES:
    t = torch_fn()
    m = shim_fn()
    if not check(t, m):
        print(f"FAIL {name}")
        fails += 1
    else:
        print(f"ok   {name}")

print(f"{len(CASES) - fails}/{len(CASES)} passed")
sys.exit(1 if fails else 0)
