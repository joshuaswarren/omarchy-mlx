#!/usr/bin/env python3
"""PT-AF executor-level test: the 0-dim scalar family.

aten._local_scalar_dense (the .item() graph op feeding .item()-style
branches - the whisper getitem-adjacent family) must hand the exact Python
scalar to the graph, and aten.Float (the Scalar/Tensor overloads wrap a
number back into a 0-dim fp32 tensor) must lower through MLX with the
executor stats staying on mlx - no torch CPU fallback records.

The graph is hand-built: dynamo rewrites scalar extraction into
get_attr/sym plumbing and never reaches these executor lines.
    python3 test_scalar_executor.py
"""
import sys
import os
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.fx

import mlx_vulkan  # noqa: F401 - registers the backend


def build_graph():
    g = torch.fx.Graph()
    x = g.placeholder("x")            # 0-dim tensor: the scalar source
    y = g.placeholder("y")            # a real tensor for the broadcast arm
    scalar = g.call_function(
        torch.ops.aten._local_scalar_dense.default, (x,))
    zt = g.call_function(torch.ops.aten.Float.Scalar, (scalar,))
    out = g.call_function(torch.ops.aten.mul.default, (zt, y))
    g.output((out, scalar, zt))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    return gm


def main():
    run, stats = mlx_vulkan.make_executor(build_graph())
    x = torch.tensor(2.5)
    y = torch.tensor([1.0, -2.0, 3.0])
    out, scalar, zt = run(x, y)

    expect = [2.5, -5.0, 7.5]
    got_list = out.tolist() if hasattr(out, "tolist") else list(out)
    # PT-AF review round 2: native scalar types (torch .item() keeps int/bool)
    # and the non-0-dim Float.Tensor cast (int64 -> fp32, torch .float()
    # semantics) that previously slipped through uncast.
    for v, kind in ((torch.tensor(7), 7), (torch.tensor(True), True)):
        got = mlx_vulkan.ATEN_TABLE["_local_scalar_dense"](mx.array(v.numpy()))
        assert got == kind and isinstance(got, type(kind)), \
            f"scalar extraction gave {got!r} ({type(got).__name__}), want {kind!r}"
        print("PASS scalar-extract", type(kind).__name__, got)
    ft = mlx_vulkan.ATEN_TABLE["Float"](mx.array(np.array([1, 2, 3], dtype=np.int64)))
    assert ft.dtype == mx.float32 and ft.tolist() == [1.0, 2.0, 3.0], \
        f"Float.Tensor int64 came back {ft.dtype} {ft.tolist()}"
    print("PASS Float.Tensor int64 -> fp32", ft.tolist())

    mlx_ops = stats.get("mlx", [])
    cpu_ops = stats.get("cpu", [])
    lowered = "_local_scalar_dense" in mlx_ops and "Float" in mlx_ops \
        and "mul" in mlx_ops
    clean = not any("local_scalar" in c or c == "Float" for c in cpu_ops)
    print(("PASS" if lowered and clean else "FAIL"),
          "stats mlx:", mlx_ops, "cpu:", cpu_ops)
    return 0 if ok and lowered and clean else 1


if __name__ == "__main__":
    sys.exit(main())
