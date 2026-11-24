#!/usr/bin/env python3
"""Executor-level failing-first test for the PT-K getitem blocker.

wAX's iter1 crash: torch.export traces whisper-small's tuple indexing so the
executor sees operator.getitem with a 0-dim mx.array as the index, and the raw
`env[node] = args[0][args[1]]` subscript raises on a Python tuple. The
dispatch-table lowerings were fixed in #767; the executor path was not.

The graph is built by hand: dynamo/symbolic_trace rewrite a tensor index into
a select and never reach that executor line. Exits 0 when the executor handles
the 0-dim tensor index; before the fix it raises TypeError.
    python3 test_getitem_executor.py
"""
import operator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.fx

import mlx_vulkan  # noqa: F401 - registers the backend


def build_graph():
    g = torch.fx.Graph()
    idx = g.placeholder("idx")
    options = g.get_attr("options")
    out = g.call_function(operator.getitem, (options, idx))
    g.output(out)
    inner = torch.nn.Module()
    inner.options = (torch.tensor([2.0, 4.0]), torch.tensor([3.0, 6.0]))
    gm = torch.fx.GraphModule(inner, g)
    return gm


def main():
    run, stats = mlx_vulkan.make_executor(build_graph())
    out = run(torch.tensor(1))
    got = out.tolist() if hasattr(out, "tolist") else out
    expect = [3.0, 6.0]
    ok = list(got) == expect
    print(("PASS" if ok else "FAIL"), "got", got, "expect", expect)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
