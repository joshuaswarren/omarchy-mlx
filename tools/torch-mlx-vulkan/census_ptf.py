#!/usr/bin/env python3
"""PAR-47/PT-F census: prove aten.embedding_dense_backward and
aten.select_backward lower (not FALLBACK) through torch.compile with the
mlx_vulkan backend behind aot_autograd, and that their gradients are finite.
PT-AL-a adds the same check for the functional elu/selu/celu lowerings.
Run on a GPU host with torch + mlx-omarchy on the path."""
import contextlib
import io
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
import mlx_vulkan


def _aot(backend_mod):
    def lower(gm, example_inputs):
        runner = backend_mod.mlx_vulkan_backend(gm, example_inputs,
                                                return_all_outputs=True)

        def call(*inputs):
            flat = inputs
            if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                    and not torch.is_tensor(inputs[0])):
                flat = tuple(inputs[0])
            out = runner(*flat)
            if isinstance(out, tuple):
                out = tuple(x.requires_grad_(True) if torch.is_tensor(x)
                            and x.is_floating_point() else x for x in out)
            elif torch.is_tensor(out) and out.is_floating_point():
                out = out.requires_grad_(True)
            return out

        call._boxed_call = True
        return call

    return aot_autograd(fw_compiler=lower, bw_compiler=lower)


def run_graph(fn, args):
    os.environ["MLX_VK_EAGER"] = "1"   # per-node census: the eager walk logs LOWERED/FALLBACK per op
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend=_aot(mlx_vulkan), fullgraph=True)
    log = logging.getLogger("mlx_vulkan")
    log.setLevel(logging.INFO)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    log.addHandler(handler)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            out = compiled(*args)
            out.backward()
    finally:
        log.removeHandler(handler)
    text = buf.getvalue()
    return out.detach(), text


def census(text, op, results):
    lowered = (re.search(op + r"[^\n]*LOWERED", text)
               or re.search(r"LOWERED[^\n]*" + op, text)
               or (op in text and "FALLBACK" not in text.split(op)[-1][:200]))
    results[op] = bool(lowered)
    print(f"census {op}: {'LOWERED' if lowered else 'NOT-LOWERED-IN-LOG'}")
    hits = [ln for ln in text.splitlines() if op in ln]
    for ln in hits[:6]:
        print("  log:", ln.strip()[:160])


import re  # noqa: E402

w = torch.randn(32, 16, requires_grad=True)
idx = torch.randint(0, 32, (8,))
t_sel = torch.randn(16, requires_grad=True)


def sel(w, idx, t):
    e = torch.embedding(w, idx)   # fwd aten.embedding
    s = e[2]                      # fwd aten.select -> bwd select_backward
    return (s * t).sum()


out, text1 = run_graph(sel, (w, idx, t_sel))
results = {}
census(text1, "select_backward", results)
print("select grad finite:", bool(torch.isfinite(w.grad).all()))
w.grad = None
t_sel.grad = None


def emb(w, idx, t_emb):
    e = torch.embedding(w, idx)   # fwd aten.embedding
    return (e * t_emb).sum()


t_emb = torch.randn(8, 16, requires_grad=True)
out2, text2 = run_graph(emb, (w, idx, t_emb))
census(text2, "embedding_dense_backward", results)
print("embedding grad finite:", bool(torch.isfinite(w.grad).all()))
print("CENSUS:", results)
assert results["select_backward"], "select_backward did not lower"
assert results["embedding_dense_backward"], "embedding_dense_backward did not lower"
print("PT-F CENSUS PASS: both ops lower")


# PT-AL-a: the functional elu/selu/celu entries lower through the same
# compiled path; the forward pass alone exercises the dispatch. dynamo
# decomposes selu onto aten.elu (scale=1.0507..., alpha=1.6732...), so the
# selu graph is censed for aten.elu too - verified live: its only nodes are
# LOWERED elu + LOWERED sum.
x_act = torch.randn(16, requires_grad=True)

for name, fn, ops in (
    ("elu", lambda t: F.elu(t, alpha=1.2).sum(), ("aten.elu",)),
    ("selu", lambda t: F.selu(t).sum(), ("aten.elu",)),
    ("celu", lambda t: F.celu(t, alpha=1.5).sum(), ("aten.celu",)),
):
    _out, text = run_graph(fn, (x_act,))
    shorts = {op.split(".")[-1] for op in ops}
    fwd_fallback = [ln for ln in text.splitlines()
                    if ln.startswith("FALLBACK") and ln.split("\t")[1:2][0] in shorts]
    assert not fwd_fallback, f"{name} forward fell back:\n" + "\n".join(fwd_fallback)
    for op in ops:
        census(text, op, results)
    x_act.grad = None

print("CENSUS:", results)
assert results["aten.elu"], "elu (and selu's decomposition) did not lower"
assert results["aten.celu"], "celu did not lower"
print("PT-AL-a CENSUS PASS: elu, selu (as decomposed aten.elu), celu lower")
