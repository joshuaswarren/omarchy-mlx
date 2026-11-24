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

Also covered: the _fuse_rope matcher binds the rotated tensor inside
mul(x, cos), requires cos/sin of one shared angle, and only accepts
last-axis (Ellipsis) slices; the runtime gate rejects non-geometric
inv_freq, per-row and non-contiguous positions, and non-[B,H,L,D]
layouts; _ph_of state detection follows only same-storage aliases
(detach), not derived tensors (neg); index_copy_ honours negative dims;
and a mutated L_self_ placeholder counts as state, not static.

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


def build_rope_graph(half=3, bad_cos=False, bad_slice=False, bad_angle=False):
    """HF NeoX decomposition:
    add(mul(x, cos(ang)), mul(cat(neg(x[...,h:]), x[...,:h]), sin(ang)))
    with ang = cat(pos*inv_freq, pos*inv_freq) shared by cos and sin.
    """
    g = torch.fx.Graph()
    x = g.placeholder("l_kwargs_x")
    pos = g.placeholder("l_kwargs_position_ids")
    invf = g.placeholder("l_self_modules_rotary_buffers_inv_freq_")
    u1 = g.call_method("unsqueeze", (pos, -1))
    ang = g.call_function(torch.mul, (u1, invf))
    angf = g.call_function(torch.cat, ((ang, ang),), {"dim": -1})
    c = g.call_function(torch.cos, (angf,))
    if bad_angle:
        s = g.call_function(torch.sin, (g.call_function(
            torch.cat, ((ang, ang),), {"dim": -1}),))
    else:
        s = g.call_function(torch.sin, (angf,))
    if bad_slice:
        # bounded slice on axis 1 (no Ellipsis) — must not match
        g1 = g.call_function(operator.getitem,
                             (x, (slice(None), slice(None, half))))
        g2 = g.call_function(operator.getitem,
                             (x, (slice(None), slice(half, None))))
    else:
        g1 = g.call_function(operator.getitem, (x, (Ellipsis, slice(None, half))))
        g2 = g.call_function(operator.getitem, (x, (Ellipsis, slice(half, None))))
    nneg = g.call_function(torch.neg, (g2,))
    catn = g.call_function(torch.cat, ((nneg, g1),), {"dim": -1})
    m1 = g.call_function(torch.mul, (g.placeholder("l_kwargs_y") if bad_cos else x, c))
    m2 = g.call_function(torch.mul, (catn, s))
    out = g.call_function(torch.add, (m1, m2))
    g.output(out)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def build_alias_graph(alias):
    """keys mutated through `alias` (detach -> state, neg -> not state).

    keys is named l_self_modules_past_key_values_keys: it matches both the
    static prefix and the state suffix, so a correct implementation counts
    it as state (and subtracts it from static) only when the mutation
    actually reaches it through a same-storage alias.
    """
    g = torch.fx.Graph()
    keys = g.placeholder("l_self_modules_past_key_values_keys")
    idx = g.placeholder("l_kwargs_idx")
    x = g.placeholder("l_kwargs_x")
    if alias == "detach":
        t = g.call_method("detach", (keys,))
    else:
        t = g.call_function(torch.neg, (keys,))
    g.call_method("index_copy_", (t, 0, idx, x))
    s = g.call_function(torch.sum, (keys,))
    g.output(s)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def build_negdim_graph():
    g = torch.fx.Graph()
    keys = g.placeholder("l_kwargs_past_key_values_layers_0_keys")
    cum = g.placeholder("l_kwargs_past_key_values_layers_0_cumulative_length")
    x = g.placeholder("l_kwargs_x")
    g.call_method("index_copy_", (keys, -1, cum, x))
    s = g.call_function(torch.sum, (keys,))
    g.output(s)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def expect_raise(run, inputs, frag, fails):
    try:
        run(*inputs)
    except RuntimeError as e:
        if frag in str(e):
            return
        fails.append(f"raise '{e}' missing '{frag}'")
        return
    fails.append(f"no RuntimeError for '{frag}'")


def main():
    fails = []

    # --- state/static detection + write-back on torch 2.15 names ----------
    run = mlx_vulkan.make_compiled_executor(build_graph())
    stats = run._mlx_stats
    w = torch.eye(2)
    keys = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    cum = torch.tensor([1])
    x = torch.tensor([[2.0, 2.0]])

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

    # --- _fuse_rope matcher: positive + structural rejections -------------
    gm = build_rope_graph()
    _, rfused = mlx_vulkan._fuse_rope(list(gm.graph.nodes))
    if len(rfused) != 1:
        fails.append(f"rope fused={len(rfused)} expect 1")
    for tag, kw in (("m_cos-not-x", {"bad_cos": True}),
                    ("non-last-axis-slice", {"bad_slice": True}),
                    ("distinct-angle", {"bad_angle": True})):
        _, bad = mlx_vulkan._fuse_rope(list(build_rope_graph(**kw).graph.nodes))
        if bad:
            fails.append(f"rope fused on rejected pattern: {tag}")

    # --- fused rope: numeric match vs manual NeoX rotation ----------------
    half, L, D = 3, 2, 6
    x = torch.arange(D * L, dtype=torch.float32).reshape(1, 1, L, D) / 10.0
    pos = torch.arange(L).reshape(1, L)
    invf = torch.tensor([1.0, 1e-2, 1e-4])            # theta = 1e6, geometric
    rrun = mlx_vulkan.make_compiled_executor(build_rope_graph())
    if rrun._mlx_stats["fused_rope"] != 1:
        fails.append(f"fused_rope={rrun._mlx_stats['fused_rope']} expect 1")
    out = rrun(x, pos, invf)[0]
    ang = pos[..., None].float() * invf                    # [1,L,h]
    angf = torch.cat([ang, ang], -1).unsqueeze(0)          # [1,1,L,2h]
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    exp = x * angf.cos() + rot * angf.sin()
    if not torch.allclose(out, exp, atol=2e-2):
        fails.append(f"rope out maxdiff {(out - exp).abs().max().item():.4f}")

    # --- fused rope: runtime gates fail loudly ----------------------------
    bad_gm = build_rope_graph()
    expect_raise(mlx_vulkan.make_compiled_executor(bad_gm),
                 (x, pos, torch.tensor([1.0, 0.1, 0.5])),
                 "non-geometric", fails)
    expect_raise(mlx_vulkan.make_compiled_executor(build_rope_graph()),
                 (x, pos, torch.tensor([1.0])), "inv_freq numel", fails)
    expect_raise(mlx_vulkan.make_compiled_executor(build_rope_graph()),
                 (x.expand(2, 1, L, D).contiguous(),
                  torch.tensor([[0, 1], [2, 3]]), invf),
                 "per-row", fails)
    expect_raise(mlx_vulkan.make_compiled_executor(build_rope_graph()),
                 (x, torch.tensor([[0, 2]]), invf), "non-contiguous", fails)
    expect_raise(mlx_vulkan.make_compiled_executor(build_rope_graph()),
                 (torch.randn(1, L, 1, D), pos, invf), "layout", fails)

    # --- alias walk: detach marks state + writes back; neg does not -------
    keys = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    idx = torch.tensor([1])
    xs = torch.tensor([[2.0, 2.0]])
    arun = mlx_vulkan.make_compiled_executor(build_alias_graph("detach"))
    # mutation through detach reaches the ph: state=1, and the L_self_
    # state ph is subtracted from static. Without the env[ph] write-back
    # sum(keys) would still read the initial buffer (2.0, not 6.0).
    if arun._mlx_stats["state"] != 1:
        fails.append(f"alias state={arun._mlx_stats['state']} expect 1")
    if arun._mlx_stats["static"] != 0:
        fails.append(f"alias static={arun._mlx_stats['static']} expect 0")
    a1 = arun(keys, idx, xs)[0].item()
    a2 = arun(keys, idx, xs)[0].item()
    if (a1, a2) != (6.0, 6.0):
        fails.append(f"alias write-back ({a1},{a2}) expect (6.0, 6.0)")

    nrun = mlx_vulkan.make_compiled_executor(build_alias_graph("neg"))
    # neg() output is a new tensor: mutation must NOT mark keys as state
    # and must not write back to env[keys] (sum stays 2.0 either way).
    if nrun._mlx_stats["state"] != 0:
        fails.append(f"non-alias state={nrun._mlx_stats['state']} expect 0")
    n1 = nrun(keys, idx, xs)[0].item()
    n2 = nrun(keys, idx, xs)[0].item()
    if (n1, n2) != (2.0, 2.0):
        fails.append(f"non-alias replay ({n1},{n2}) expect (2.0, 2.0)")

    # --- index_copy_ with a negative dim ----------------------------------
    drun = mlx_vulkan.make_compiled_executor(build_negdim_graph())
    kk = torch.zeros(2, 4)
    d1 = drun(kk, cum, torch.tensor([[5.0], [7.0]]))[0].item()
    if d1 != 12.0:  # col 1 -> [[0,5,0,0],[0,7,0,0]] sums to 12
        fails.append(f"neg-dim index_copy_ {d1} expect 12.0")

    for f in fails:
        print("FAIL", f)
    print(f"{0 if fails else 1} checks pass; stats={stats}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
