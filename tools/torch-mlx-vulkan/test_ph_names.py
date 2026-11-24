#!/usr/bin/env python3
"""Placeholder-name coverage for the torch>=2.15 dynamo naming scheme.

Dynamo on torch 2.15 lifts nn params/buffers as L_self_* placeholders and
StaticCache buffers as L_kwargs_past_key_values_layers_N_{keys,values,
cumulative_length}. The compiled executor only knew the older G_* /
_cache_layers_ conventions, so every weight was re-converted and
re-uploaded each call and the KV + cumulative_length buffers never
entered the state path (the frozen mask bound then diverged greedy
decode at step 2).

The graph is built by hand so the placeholder target strings are exact:
keys/cumulative_length are mutated in place (index_copy_/add_), so the
state machinery must engage; the sum() output after the mutation makes
state write-back observable across two calls. Before the fix stats
report state=0/static=0 and call 2 returns the call-1 output.

    python3 test_ph_names.py
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
    w = g.placeholder("l_self_modules_fc_parameters_weight_")
    keys = g.placeholder("l_kwargs_past_key_values_layers_0_keys")
    cum = g.placeholder("l_kwargs_past_key_values_layers_0_cumulative_length")
    x = g.placeholder("l_kwargs_x")
    g.call_method("index_copy_", (keys, 0, cum, x))
    g.call_method("add_", (cum, 1))
    s = g.call_function(torch.sum, (keys,))
    y = g.call_function(torch.mm, (x, w))
    out = g.call_function(torch.add, (y, s))
    g.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def main():
    run = mlx_vulkan.make_compiled_executor(build_graph())
    stats = run._mlx_stats

    w = torch.eye(2)
    keys = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    cum = torch.tensor([1])
    x = torch.tensor([[2.0, 2.0]])

    fails = []
    if stats["state"] != 2:
        fails.append(f"state={stats['state']} expect 2")
    if stats["static"] != 1:
        fails.append(f"static={stats['static']} expect 1")

    out1 = run(w, keys, cum, x)[0].tolist()
    # call 2 replays mutated state: cum=1 wrote row 1 and ticked to 2,
    # so this call writes row 2 -> keys sum 8, not the call-1 sum 6.
    out2 = run(w, keys, cum, x)[0].tolist()
    if out1 != [[8.0, 8.0]]:
        fails.append(f"call1={out1} expect [[8.0, 8.0]]")
    if out2 != [[12.0, 12.0]]:
        fails.append(f"call2={out2} expect [[12.0, 12.0]]")

    for f in fails:
        print("FAIL", f)
    print(f"{0 if fails else 1} checks pass; stats={stats}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
