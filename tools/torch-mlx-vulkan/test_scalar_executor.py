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
    wrapped = zt.tolist() if hasattr(zt, "tolist") else zt
    ok = (all(abs(a - b) < 1e-9 for a, b in zip(got_list, expect))
          and abs(float(scalar) - 2.5) < 1e-9
          and abs(float(wrapped) - 2.5) < 1e-9)
    print(("PASS" if ok else "FAIL"), "scalar", scalar,
          "wrapped", wrapped, "out", got_list, "expect", expect)

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
