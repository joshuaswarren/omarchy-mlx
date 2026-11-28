"""torch.compile backend 'mlx_vulkan': lowers dynamo FX graphs to mlx (omarchy-mlx, Vulkan/Apple GPU).

dynamo hands us torch-level graphs: call_function targets are torch.* /
torch.nn.functional.* builtins (and, for AOT-style graphs, aten.* OpOverloads),
call_method targets are Tensor method names. Three dispatch outcomes:

  LOWERED   -> executed through mlx.core on the GPU
  GLUE      -> python/shape bookkeeping, no kernel on either side
  FALLBACK  -> executed through torch on CPU (mx<->torch boundary conversion)

Every decision is logged to the 'mlx_vulkan' logger; that log is the
op-coverage evidence for the milestone.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
import math
import operator
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as _mxnn
import torch._dynamo

log = logging.getLogger("mlx_vulkan")
_DBG_ENV = None

# Cross-process cache: when the user opts in (MLX_VK_DISK_CACHE=1 -> dynamo
# caching_precompile + save_cache_artifacts at exit), dynamo stores each frame's
# CompilePackage under <inductor-cache>/dynamo keyed on the wrapped fn. A package
# is only loadable if every __compiled_fn_<id> it references has a registered
# BackendCacheArtifact - we serialize the traced GraphModule (all our executors
# need) and rebuild the backend callable in after_deserialization, so a warm
# process skips the whole dynamo trace AND our lowering walk before first call.
try:
    from torch._dynamo.precompile_context import (
        BackendCacheArtifact as _BCA,
        PrecompileContext as _PCtx,
    )
except Exception:  # older torch without the precompile API
    _BCA = None
    _PCtx = None

if _BCA is not None:

    @dataclasses.dataclass
    class _MlxVkArtifact(_BCA):
        def __getstate__(self):
            # torch re-binds this artifact's backend as
            # module.__dict__[backend_id] for the frame's recorded
            # python_module - which comes from fn.__module__ and is spoofed by
            # @functools.wraps wrappers (HF deprecate_kwarg): the compiled code
            # then runs under the *wrapper* module's globals and LOAD_GLOBAL
            # misses. At save time the name is bound in the real dicts, so
            # snapshot those module names and restore them at load.
            mods = [
                m.__name__
                for m in list(sys.modules.values())
                if isinstance(getattr(m, "__dict__", None), dict)
                and self.key in m.__dict__
            ]
            # The same spoofed __module__ makes torch install() plant the
            # entry's other dynamo globals in the wrong dict; snapshot the
            # module-valued ones (__import_* aliases, ___module_* handles)
            # from the dicts that really ran the code.
            imports = {}
            for m in list(sys.modules.values()):
                d = getattr(m, "__dict__", None)
                if isinstance(d, dict) and self.key in d:
                    for k, v in d.items():
                        if isinstance(v, types.ModuleType) and (
                                k.startswith("__import_")
                                or k.startswith("___module_")):
                            imports[k] = v.__name__
            return {"key": self.key, "content": self.content, "mods": mods,
                    "imports": imports}

        def __setstate__(self, state):
            self.key = state["key"]
            self.content = state["content"]
            self._mods = state.get("mods", [])
            self._imports = state.get("imports", {})

        def after_deserialization(self):
            backend = _restore_backend(self.content)
            for name in getattr(self, "_mods", ()):
                mod = sys.modules.get(name)
                if mod is not None:
                    mod.__dict__[self.key] = backend
                    for alias, modname in self._imports.items():
                        if alias not in mod.__dict__:
                            mod.__dict__[alias] = importlib.import_module(modname)
            return backend


def _restore_backend(gm):
    return mlx_vulkan_backend(gm, [])


def _record_artifact(gm):
    if _PCtx is None:
        return
    bid = getattr(gm, "meta", {}).get("backend_id")
    if not bid:
        return
    try:
        _PCtx.record_artifact(_MlxVkArtifact(key=str(bid), content=gm))
    except Exception as e:
        log.debug("precompile artifact record failed: %s", e)


if os.environ.get("MLX_VK_DISK_CACHE") and _PCtx is not None:
    torch._dynamo.config.caching_precompile = True

    import atexit

    def _save_artifacts():
        try:
            torch.compiler.save_cache_artifacts()
        except Exception as e:
            log.debug("cache artifact save failed: %s", e)

    atexit.register(_save_artifacts)

# ---------------- conversions ----------------

def _mx(v):
    if isinstance(v, mx.array):
        return v
    if isinstance(v, torch.Tensor):
        if v.dtype == torch.bfloat16:
            return mx.view(mx.array(v.detach().cpu().view(torch.uint16).numpy()), mx.bfloat16)
        return mx.array(v.detach().cpu().numpy())
    return v

def _torch(v):
    if isinstance(v, mx.array):
        if v.dtype == mx.bfloat16:
            # numpy has no bf16 buffer descriptor and mx.view needs the last
            # axis contiguous (transposed arrays are not): widen to f32 in mlx.
            return torch.from_numpy(np.array(mx.astype(v, mx.float32)))
        return torch.from_numpy(np.array(v, copy=True))
    if isinstance(v, (tuple, list)):
        return type(v)(_torch(x) for x in v)
    return v

def _opname(node):
    """aten.mm.default -> mm ; aten::mm -> mm ; else the raw target string."""
    t = str(node.target)
    if t.startswith("aten."):
        return t[5:].split(".")[0]
    if t.startswith("aten::"):
        return t[6:]
    return t

# ---------------- helpers ----------------

def _perm_swap(a, d0, d1):
    n = a.ndim
    d0, d1 = d0 % n, d1 % n
    p = list(range(n))
    p[d0], p[d1] = p[d1], p[d0]
    return tuple(p)

def _wt(w):
    return mx.transpose(w, axes=_perm_swap(w, -2, -1))

def _linear(x, w, bias=None):
    y = mx.matmul(x, _wt(w))
    return y + _mx(bias) if bias is not None else y

def _squeeze(a, dim=None):
    if isinstance(dim, (list, tuple)):
        dim = tuple(dim)
    return mx.squeeze(a, axis=dim) if dim is not None else mx.squeeze(a)

def _reshape(a, *s):
    if len(s) == 1 and isinstance(s[0], (list, tuple)):
        s = tuple(s[0])
    if len(s) == 1 and isinstance(s[0], torch.Size):
        s = tuple(s[0])
    return mx.reshape(a, tuple(s))

# view/reshape: mx arrays carry no strides, so layout is tracked
# structurally. aten.view must alias its input storage; mx.reshape silently
# copies a strided input instead, so a view whose input may be
# non-contiguous is refused to the CPU path where torch computes the real
# strides (or the real error). reshape never refuses: torch's own
# copy-if-needed semantics already match mx.reshape.
_VIEW_OPS = {"view", "_unsafe_view", "unsafe_view"}

# ops whose output may be non-contiguous regardless of input layout
_STRIDED_OPS = {
    "t", "mT", "transpose", "permute", "movedim", "moveaxis", "swapaxes",
    "swapdims", "expand", "expand_as", "broadcast_to", "unbind", "split",
    "split_with_sizes", "unsafe_split", "chunk", "tensor_split", "hsplit",
    "vsplit", "dsplit", "diagonal", "unfold", "as_strided", "flip",
    "rot90",
}

# layout-transparent ops: the result shares storage with the input, so the
# stridedness flag passes through
_LAYOUT_PASS = {
    "detach", "detach_", "clone", "alias", "lift_fresh", "lift_fresh_copy",
    "to", "_to_copy", "type", "cpu", "cuda", "requires_grad",
    "requires_grad_", "float", "half", "bfloat16", "double", "int", "long",
    "bool", "short", "byte", "char", "type_as",
}


def _narg(n, i, key, default=None):
    if key in n.kwargs:
        return n.kwargs[key]
    return n.args[i] if len(n.args) > i else default


def _aint(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _dim0_index(idx):
    """True when a getitem index provably keeps contiguity: a prefix of
    integer indices (each removing its dim), then at most one unit-step
    partial slice on the next dim, then only non-restricting decorations
    (None, full slices, one Ellipsis). An int or partial slice after a dim
    survives, a step!=1 slice, or any restriction after an Ellipsis (it may
    touch a trailing dim while earlier dims survive) can all produce a
    non-contiguous result."""
    if isinstance(idx, torch.fx.Node):
        return False
    if not isinstance(idx, (tuple, list)):
        idx = (idx,)
    pre, post, ell = [], [], False
    for i in idx:
        if i is Ellipsis:
            ell = True
            continue
        (post if ell else pre).append(i)
    for i in post:
        if i is None:
            continue
        if (isinstance(i, slice) and i.start is None and i.stop is None
                and _aint(i.step) in (None, 1)):
            continue
        return False
    restricted = False
    for i in pre:
        if i is None:
            continue
        if isinstance(i, torch.fx.Node):
            return False
        if isinstance(i, (int, torch.SymInt)):
            if restricted:
                return False
            continue
        if isinstance(i, slice):
            if _aint(i.step) not in (None, 1):
                return False
            if i.start is not None or i.stop is not None:
                if restricted:
                    return False
            restricted = True
            continue
        return False
    return True


def _strided_nodes(nodes):
    """Graph pass: nodes whose output may be non-contiguous under torch
    stride semantics. A view on a member is refused rather than silently
    copied by mx.reshape."""
    strided = set()
    for n in nodes:
        name = _opname(n)
        a0 = n.args[0] if (n.args and isinstance(n.args[0], torch.fx.Node)) else None
        if name in _STRIDED_OPS:
            strided.add(n)
        elif name in ("select", "narrow"):
            if a0 in strided or _aint(_narg(n, 1, "dim"), -1) != 0:
                strided.add(n)
        elif name == "slice":
            if (a0 in strided or _aint(_narg(n, 1, "dim"), -1) != 0
                    or _aint(_narg(n, 4, "step"), 1) != 1):
                strided.add(n)
        elif n.target is operator.getitem or name == "getitem":
            if a0 in strided or not _dim0_index(n.args[1] if len(n.args) > 1 else None):
                strided.add(n)
        elif name in _LAYOUT_PASS:
            if a0 in strided:
                strided.add(n)
    return strided


def _view_refused(name, n, strided):
    return (name in _VIEW_OPS and n.args
            and isinstance(n.args[0], torch.fx.Node)
            and n.args[0] in strided)


def _node_args(a):
    if isinstance(a, torch.fx.Node):
        yield a
    elif isinstance(a, (tuple, list)):
        for x in a:
            yield from _node_args(x)
    elif isinstance(a, dict):
        for x in a.values():
            yield from _node_args(x)


def _refused_nodes(nodes, strided):
    """Nodes refused to the CPU path for view/stride semantics: every view
    on a flagged input, plus the flagged producers behind it. Refusing the
    view alone is not enough -- an mx-lowered producer hands the torch
    fallback a materialized contiguous tensor, so the view then returns
    strides (or succeeds) where eager would raise. Refusing the strided
    cone keeps every producer under torch, where real strides are kept."""
    refused = set()
    stack = []
    for n in nodes:
        if _view_refused(_opname(n), n, strided):
            refused.add(n)
            stack.append(n)
    while stack:
        n = stack.pop()
        for a in _node_args(n.args):
            if a in strided and a not in refused:
                refused.add(a)
                stack.append(a)
        for a in _node_args(n.kwargs):
            if a in strided and a not in refused:
                refused.add(a)
                stack.append(a)
    return refused

def _slice(a, dim=0, start=None, end=None, step=1):
    dim = dim % a.ndim
    n = a.shape[dim]
    # aten clamps out-of-range bounds; mlx requires int32. None = open bound.
    if start is not None:
        start = max(min(start, n), -n)
    if end is not None:
        end = max(min(end, n), -n)
    idx = [slice(None)] * a.ndim
    idx[dim] = slice(start, end, step)
    return a[tuple(idx)]

def _split_sizes(a, sizes, dim):
    idx = list(np.cumsum(sizes))[:-1]
    return tuple(mx.split(a, idx, axis=dim))

def _split_even(a, split_size, dim):
    nsec = -(-a.shape[dim] // split_size)
    return tuple(mx.split(a, nsec, axis=dim))

def _chunk(a, chunks, dim):
    sz = a.shape[dim]
    step = -(-sz // chunks)
    return _split_sizes(a, [min(step, sz - i * step) for i in range(chunks) if i * step < sz], dim)

def _unbind(a, dim):
    dim = dim % a.ndim
    idx = [slice(None)] * a.ndim
    out = []
    for j in range(a.shape[dim]):
        idx[dim] = j
        out.append(a[tuple(idx)])
    return tuple(out)

def _meshgrid(*tensors, indexing="ij"):
    # torch.meshgrid(*tensors, indexing=) splats the tensor args; the
    # aten signature is Tensor[] so a single list arg can arrive too
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    ts = [_mx(t) for t in tensors]
    # aten requires all inputs same dtype and 0D/1D rank; mx accepts
    # mixed dtypes and silently flattens >1-D input — defer so the CPU
    # fallback raises torch's own error
    if any(t.dtype != ts[0].dtype for t in ts) or any(t.ndim > 1 for t in ts):
        return _raise("meshgrid mixed dtypes or >1-D input")
    return tuple(mx.meshgrid(*ts, indexing=indexing))

def _red(reduction, size_average, reduce):
    # torch legacy kwargs: reduce=False -> "none"; else size_average picks mean/sum.
    if isinstance(reduction, int):
        reduction = {0: "none", 1: "mean", 2: "sum"}[reduction]
    if reduce is not None and not reduce:
        return "none"
    if size_average is not None:
        return "mean" if size_average else "sum"
    return reduction


def _class_last(logits):
    # torch puts the class dim at axis 1 for N>2 shapes; gather needs it last.
    return mx.moveaxis(logits, 1, -1) if _mx(logits).ndim > 2 else logits


def _gather_logp(logp, t):
    return mx.take_along_axis(_mx(logp), t[..., None].astype(mx.int32), axis=-1)[..., 0]


def _reduce_loss(l_i, eff, keep, reduction):
    l_i = l_i * keep
    if reduction == "none":
        return l_i
    if reduction == "sum":
        return mx.sum(l_i)
    return mx.sum(l_i) / mx.maximum(mx.sum(eff * keep), 1e-12)


def _ce(logits, target, weight=None, ignore_index=-100, label_smoothing=0.0, reduction="mean"):
    logits, target = _mx(logits), _mx(target).astype(mx.int32)
    logits = _class_last(logits)
    C = logits.shape[-1]
    lse = mx.logsumexp(logits, axis=-1)
    keep = (target != ignore_index).astype(logits.dtype)
    t = mx.where(target != ignore_index, target, mx.zeros_like(target))
    nll = lse - _gather_logp(logits, t)
    eps = float(label_smoothing or 0.0)
    if weight is not None:
        w = _mx(weight)
        wt, wsum = w[t], mx.sum(w)
    else:
        w, wt, wsum = 1.0, mx.ones_like(nll), C
    l_i = wt * nll
    if eps:
        smooth = (lse * wsum - mx.sum(logits * w, axis=-1)) / C
        l_i = (1 - eps) * l_i + eps * smooth
    # torch mean denom = sum(w[t]) over non-ignored; smoothing adds nothing
    return _reduce_loss(l_i, wt, keep, reduction)


def _nll(logp, target, weight=None, ignore_index=-100, reduction="mean"):
    logp, target = _mx(logp), _mx(target).astype(mx.int32)
    logp = _class_last(logp)
    keep = (target != ignore_index).astype(logp.dtype)
    t = mx.where(target != ignore_index, target, mx.zeros_like(target))
    l = -_gather_logp(logp, t)
    wt = _mx(weight)[t] if weight is not None else mx.ones_like(l)
    return _reduce_loss(wt * l, wt, keep, reduction)


def _ln_backward(gout, x, normalized_shape, mean, rstd, weight, bias, output_mask):
    gout, x = _mx(gout), _mx(x)
    mean, rstd = _mx(mean), _mx(rstd)
    w = _mx(weight) if weight is not None else None
    ndim = len(tuple(normalized_shape))
    axes = tuple(range(x.ndim - ndim, x.ndim))
    # aten returns mean/rstd with keepdims, but callers through aot_autograd
    # and the OpInfo graph can hand them squeezed (shape x.shape[:-ndim]);
    # restore the trailing singletons so they broadcast over normalized dims.
    if mean.ndim < x.ndim:
        shp = mean.shape + (1,) * (x.ndim - mean.ndim)
        mean, rstd = mx.reshape(mean, shp), mx.reshape(rstd, shp)
    xhat = (x - mean) * rstd
    g = gout * w if w is not None else gout
    dx = rstd * (g - mx.mean(g, axis=axes, keepdims=True)
                 - xhat * mx.mean(g * xhat, axis=axes, keepdims=True))
    sum_axes = tuple(range(x.ndim - ndim))
    dw = (mx.sum(gout * xhat, axis=sum_axes) if sum_axes else gout * xhat)
    db = (mx.sum(gout, axis=sum_axes) if sum_axes else gout)
    mask = tuple(output_mask) if output_mask is not None else (True, True, True)
    z = mx.zeros((1,), mx.float32)
    return (dx if mask[0] else z,
            dw if (weight is not None and mask[1]) else z,
            db if (bias is not None and mask[2]) else z)


def _gqa_expand(a, hq):
    hk = a.shape[-3]
    if hk == hq:
        return a
    rep = hq // hk
    a = mx.broadcast_to(mx.expand_dims(a, axis=-3),
                        a.shape[:-3] + (hk, rep) + a.shape[-2:])
    return mx.reshape(a, a.shape[:-4] + (hq,) + a.shape[-2:])


def _gqa_reduce(d, hk):
    hq = d.shape[-3]
    if hq == hk:
        return d
    rep = hq // hk
    d = mx.reshape(d, d.shape[:-3] + (hk, rep) + d.shape[-2:])
    return mx.sum(d, axis=-3)


def _slice_backward(g, input_shape, dim, start, end, step=1):
    g = _mx(g)
    shape = tuple(input_shape)
    dim = dim % len(shape)
    N = shape[dim]
    lo = min(max(start + N if start < 0 else start, 0), N)
    hi = min(max(end + N if end < 0 else end, 0), N)
    if step != 1 or lo >= hi:
        _raise(f"slice_backward step={step} range={lo}:{hi}")
    parts = []
    if lo:
        parts.append(mx.zeros(shape[:dim] + (lo,) + shape[dim + 1:], g.dtype))
    parts.append(g)
    if shape[dim] - hi:
        parts.append(mx.zeros(shape[:dim] + (shape[dim] - hi,) + shape[dim + 1:], g.dtype))
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=dim)


def _silu_backward(g, x):
    g, x = _mx(g), _mx(x)
    sig = mx.sigmoid(x)
    return g * sig * (1.0 + x * (1.0 - sig))


def _embedding_dense_backward(go, idx, num_weights, padding_idx,
                              scale_grad_by_freq):
    # aten: grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    g = _mx(go)
    g = g.reshape(-1, g.shape[-1])
    i = _int_index(idx, "aten.embedding_dense_backward").reshape(-1)
    out = mx.zeros((num_weights, g.shape[-1]), dtype=g.dtype)
    out = out.at[i].add(g)
    if scale_grad_by_freq:
        # counts in int32 keep exactness past 256 (bf16) / 65504 (fp16);
        # the division itself runs in float32 like torch's.
        c = mx.zeros((num_weights,), dtype=mx.int32).at[i].add(
            mx.ones(i.shape, dtype=mx.int32))
        out = (out.astype(mx.float32)
               / mx.maximum(c.astype(mx.float32), 1.0)[:, None]).astype(g.dtype)
    if padding_idx >= 0:
        out = mx.where((mx.arange(num_weights) != padding_idx)[:, None],
                       out, 0.0)
    return out


def _sdpa_bias(q, k, is_causal=False, attn_mask=None, causal_off=0):
    # additive scores bias shared by the flash-cpu/efficient lowerings: an
    # explicit attn_mask wins over is_causal; bool masks mean keep/drop.
    if attn_mask is not None:
        m = _mx(attn_mask)
        return mx.where(m, 0.0, -mx.inf) if m.dtype == mx.bool_ else m
    if not is_causal:
        return None
    sq, sk = q.shape[-2], k.shape[-2]
    keep = mx.less_equal(mx.arange(sk).reshape(1, sk),
                         mx.arange(sq).reshape(sq, 1) + causal_off)
    return mx.where(keep, 0.0, -mx.inf)


def _sdpa_flash_cpu(q, k, v, dropout_p=0.0, is_causal=False, attn_mask=None, scale=None):
    if dropout_p:
        _raise("flash_cpu dropout_p")
    q, k, v = _mx(q), _mx(k), _mx(v)
    k = _gqa_expand(k, q.shape[-3])
    v = _gqa_expand(v, q.shape[-3])
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    bias = _sdpa_bias(q, k, is_causal, attn_mask)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=s,
                                                  mask=None if bias is None else bias.astype(q.dtype))
    scores = s * mx.matmul(q.astype(mx.float32),
                           mx.swapaxes(k.astype(mx.float32), -2, -1))
    if bias is not None:
        scores = scores + _mx(bias).astype(mx.float32)
    lse = mx.logsumexp(scores, axis=-1)
    return (out, lse)


def _sdpa_flash_cpu_bwd(gout, q, k, v, out, lse, dropout_p, is_causal,
                        attn_mask=None, scale=None):
    if dropout_p:
        _raise("flash_cpu_bwd dropout_p")
    gout, q, k, v, out = _mx(gout), _mx(q), _mx(k), _mx(v), _mx(out)
    lse = _mx(lse)
    dq_t, dk_t, dv_t = q.dtype, k.dtype, v.dtype
    hk = k.shape[-3]
    k = _gqa_expand(k, q.shape[-3])
    v = _gqa_expand(v, q.shape[-3])
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    bias = _sdpa_bias(q, k, is_causal, attn_mask)
    scores = s * mx.matmul(q.astype(mx.float32),
                           mx.swapaxes(k.astype(mx.float32), -2, -1))
    if bias is not None:
        scores = scores + _mx(bias).astype(mx.float32)
    if lse.ndim == scores.ndim:
        lse = lse[..., 0]
    # backward in fp32: inputs can be bf16/fp16, so gout/out/v/k/q are
    # promoted before the products and dq/dk/dv cast back at the end.
    gout, out = gout.astype(mx.float32), out.astype(mx.float32)
    p = mx.exp(scores - lse[..., None])
    dv = mx.matmul(mx.swapaxes(p, -2, -1), gout)
    dP = mx.matmul(gout, mx.swapaxes(v.astype(mx.float32), -2, -1))
    dsum = mx.sum(gout * out, axis=-1, keepdims=True)
    dS = p * (dP - dsum)
    dq = s * mx.matmul(dS, k.astype(mx.float32))
    dk = s * mx.matmul(mx.swapaxes(dS, -2, -1), q.astype(mx.float32))
    return (dq.astype(dq_t), _gqa_reduce(dk, hk).astype(dk_t),
            _gqa_reduce(dv, hk).astype(dv_t))


def _ln_native(a, normalized_shape, weight, bias, eps):
    """native_layer_norm with real mean/rstd.

    aot_autograd saves the returned mean and rstd and feeds them to
    aten.native_layer_norm_backward; dummies (the old lowering) make that
    backward divide by rstd = 0 and every gradient upstream of the norm
    turns NaN at the first training step (PT-T1 bisect). The stats reduce
    over the trailing len(normalized_shape) axes and keep them singleton,
    matching aten's saved shapes for any normalized_shape.
    """
    w = _mx(weight) if weight is not None else None
    b = _mx(bias) if bias is not None else None
    axes = tuple(range(a.ndim - len(tuple(normalized_shape)), a.ndim))
    mu = mx.mean(a, axis=axes, keepdims=True)
    xc = a - mu
    rstd = 1.0 / mx.sqrt(mx.mean(xc * xc, axis=axes, keepdims=True) + eps)
    out = xc * rstd
    if w is not None:
        out = out * w
    if b is not None:
        out = out + b
    return (out, mu, rstd)

def _expand(a, *s, **k):
    if len(s) == 1 and isinstance(s[0], (list, tuple)):
        s = tuple(s[0])
    off = len(s) - a.ndim
    if off < 0:
        raise ValueError(f"expand: {len(s)} sizes < {a.ndim} dims")
    def _d(i, x):
        if x != -1:
            return x
        j = i - off  # -1 sentinel: keep input dim, right-aligned
        if j < 0:
            raise ValueError(f"expand: -1 in leading dim {i} has no input dim")
        return a.shape[j]
    return mx.broadcast_to(a, tuple(_d(i, x) for i, x in enumerate(s)))

def _pad_width(a, p):
    pairs = [(0, 0)] * a.ndim
    dims = len(p) // 2
    for i in range(dims):
        pairs[a.ndim - 1 - i] = (int(p[2 * i]), int(p[2 * i + 1]))
    return tuple(pairs)

def _shape_of(a, k):
    if len(a) == 1:
        return tuple(a[0]) if isinstance(a[0], (list, tuple)) else (a[0],)
    return tuple(a) if a else tuple(k.get("size", ()))

def _cat_impl(t, dim=0):
    xs = [_mx(x) for x in t]
    nd = max(x.ndim for x in xs)
    if not -nd <= dim < nd:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-nd}, {nd - 1}], but got {dim})")
    keep = [x for x in xs if x.size != 0 or x.ndim == nd]
    if not keep:
        keep = xs
    if len(keep) == 1:
        return keep[0]
    return mx.concatenate(keep, axis=dim % nd)

def _mean(a, dim=None, keepdim=False, dtype=None, **k):
    a = _mx(a)
    # aten casts the INPUT to dtype before the reduction
    if dtype is not None:
        if dtype not in _MX_DT:
            return _raise(f"mean dtype {dtype}")
        a = a.astype(_MX_DT[dtype])
    # torch tolerates dim 0/-1 on a 0-d tensor and reduces nothing
    if a.ndim == 0:
        return a
    axis = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    # an empty dim list reduces all dims in aten
    return mx.mean(a, axis=None if axis == () else axis, keepdims=keepdim)

def _red_axis(a, dim):
    axis = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    return None if axis == () else axis

def _var_impl(a, axis, keepdims, c):
    ax = (axis,) if isinstance(axis, int) else (axis if axis is not None else tuple(range(a.ndim)))
    n = 1
    for d in ax:
        n *= a.shape[d % a.ndim]
    dof = n - float(c)
    if dof <= 0:
        # torch: positive ssq / nonpositive dof -> +inf, zero ssq -> nan
        m0 = mx.mean(a, axis=axis, keepdims=True)
        ssq = mx.sum(mx.square(a - m0), axis=axis, keepdims=keepdims)
        return mx.where(ssq > 0, mx.array(float("inf"), dtype=ssq.dtype),
                        mx.array(float("nan"), dtype=ssq.dtype))
    if float(c).is_integer() and float(c) >= 0:
        return mx.var(a, axis=axis, keepdims=keepdims, ddof=int(c))
    # fractional or negative correction: ssq / (n - c)
    m = mx.mean(a, axis=axis, keepdims=True)
    return mx.sum(mx.square(a - m), axis=axis, keepdims=keepdims) / dof

def _unbias(dim, correction, unbiased):
    # the var/std overloads spell the ddof three ways: .default binds a bool
    # `unbiased` as arg 2 where .dim/.correction bind dim; .dim also takes an
    # `unbiased` kwarg; .correction takes `correction` — normalize to (dim, c)
    if isinstance(dim, bool):
        return None, 1 if dim else 0
    if unbiased is not None and correction is None:
        correction = 1 if unbiased else 0
    return dim, correction

def _var(a, dim=None, correction=None, keepdim=False, unbiased=None, **k):
    a = _mx(a)
    dim, correction = _unbias(dim, correction, unbiased)
    axis = _red_axis(a, dim)
    if a.ndim == 0 or _red_n(a, axis) == 0:
        # scalar/empty inputs: inf/nan scalar and empty-reduce semantics are
        # subtle enough to leave to the CPU fallback (w74 review)
        return _raise(f"var scalar/empty input")
    return _var_impl(a, axis, keepdim, 1 if correction is None else correction)

def _red_n(a, axis):
    if a.ndim == 0:
        return 1
    ax = (axis,) if isinstance(axis, int) else (axis if axis is not None else tuple(range(a.ndim)))
    n = 1
    for d in ax:
        n *= a.shape[d % a.ndim]
    return n

def _std(a, dim=None, correction=None, keepdim=False, unbiased=None, **k):
    a = _mx(a)
    dim, correction = _unbias(dim, correction, unbiased)
    axis = _red_axis(a, dim)
    if a.ndim == 0 or _red_n(a, axis) == 0:
        return _raise(f"std scalar/empty input")
    c = 1 if correction is None else correction
    if float(c).is_integer() and float(c) >= 0 and _red_n(a, axis) - float(c) > 0:
        return mx.std(a, axis=axis, keepdims=keepdim, correction=int(c))
    return mx.sqrt(_var_impl(a, axis, keepdim, c))

def _var_mean(a, dim=None, correction=None, keepdim=False, unbiased=None, **k):
    dim, correction = _unbias(dim, correction, unbiased)
    v = _var(a, dim, correction, keepdim)
    m = _mean(a, dim, keepdim)
    return v, m

def _std_mean(a, dim=None, correction=None, keepdim=False, unbiased=None, **k):
    dim, correction = _unbias(dim, correction, unbiased)
    s = _std(a, dim, correction, keepdim)
    m = _mean(a, dim, keepdim)
    return s, m

def _sum(a, dim=None, keepdim=False, dtype=None, **k):
    a = _mx(a)
    # aten casts the INPUT to dtype before the reduction
    if dtype is not None:
        if dtype not in _MX_DT:
            return _raise(f"sum dtype {dtype}")
        a = a.astype(_MX_DT[dtype])
    # torch tolerates dim 0/-1 on a 0-d tensor and reduces nothing
    if a.ndim == 0:
        return a
    axis = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    # an empty dim list reduces all dims in aten
    return mx.sum(a, axis=None if axis == () else axis, keepdims=keepdim)

_INT_DT = (mx.int8, mx.int16, mx.int32, mx.int64,
           mx.uint8, mx.uint16, mx.uint32, mx.uint64)


def _embedding(a, b, *a_, **k_):
    # F.embedding emits (indices, weight); aten.embedding emits (weight,
    # indices). Indices are integer dtype, weight is float.
    if _has_bool_index((a, b)):
        return _raise("aten.embedding: a bool index tensor is a mask in aten, not an int index")
    am, bm = _mx(a), _mx(b)
    idx, w = (am, bm) if am.dtype in _INT_DT else (bm, am)
    return w[idx.astype(mx.int32)]

def _softmax_fwd(a, dim=-1, half_to_float=False, dtype=None, *, log=False):
    a = _mx(a)
    if dtype is not None:
        if dtype not in _MX_DT:
            return _raise(f"softmax dtype {dtype}")
        a = a.astype(_MX_DT[dtype])
    if not mx.issubdtype(a.dtype, mx.floating):
        return _raise(f"softmax dtype {a.dtype}")

    use_fp32 = a.dtype in (mx.float16, mx.bfloat16)
    out_dtype = mx.float32 if half_to_float and use_fp32 else a.dtype
    if a.size == 0 and -a.ndim <= dim < a.ndim:
        return a.astype(out_dtype)
    if use_fp32:
        a = a.astype(mx.float32)
    shifted = a - mx.max(a, axis=dim, keepdims=True)
    numer = mx.exp(shifted)
    denom = mx.sum(numer, axis=dim, keepdims=True)
    result = shifted - mx.log(denom) if log else numer / denom
    return result.astype(out_dtype)

def _log_softmax_fwd(a, dim, half_to_float=False, dtype=None):
    return _softmax_fwd(a, dim, half_to_float, dtype=dtype, log=True)

def _log_softmax_bwd(g, out, dim, input_dtype=None):
    # d/dx log_softmax: g - softmax(x) * sum(g, dim)  with softmax(x)=exp(out)
    g, out = _mx(g), _mx(out)
    r = g - mx.exp(out) * mx.sum(g, axis=dim, keepdims=True)
    if input_dtype is not None:
        if input_dtype not in _MX_DT:
            return _raise(f"_log_softmax_backward_data input_dtype {input_dtype}")
        r = r.astype(_MX_DT[input_dtype])
    return r

def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    kw = {}
    # mlx 0.32's fast binding requires scale explicitly (no default).
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    kw["scale"] = scale
    if attn_mask is not None:
        kw["mask"] = _mx(attn_mask)
    elif is_causal:
        T, S = q.shape[-2], k.shape[-2]
        if T == S:
            kw["mask"] = "causal"
        else:
            kw["mask"] = mx.arange(S)[None, :] <= mx.arange(T)[:, None]
    return mx.fast.scaled_dot_product_attention(q, k, v, **kw)

def _sdpa_eff(q, k, v, attn_bias, compute_log_sumexp, dropout_p=0.0,
              is_causal=False, scale=None):
    # aten._scaled_dot_product_efficient_attention ->
    # (output, logsumexp, philox_seed, philox_offset). is_causal maps to
    # CausalFromTopLeft (causal_off=0).
    if dropout_p not in (0, 0.0):
        return _raise("efficient_attention dropout")
    q, k, v = _mx(q), _mx(k), _mx(v)
    k = _gqa_expand(k, q.shape[-3])
    v = _gqa_expand(v, q.shape[-3])
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    bias = _sdpa_bias(q, k, is_causal, attn_bias)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=s,
                                                  mask=None if bias is None else bias.astype(q.dtype))
    scores = s * mx.matmul(q.astype(mx.float32),
                           mx.swapaxes(k.astype(mx.float32), -2, -1))
    if bias is not None:
        scores = scores + _mx(bias).astype(mx.float32)
    lse = mx.logsumexp(scores, axis=-1)
    z = mx.zeros((1,), mx.int64)
    if not compute_log_sumexp:
        lse = mx.zeros(q.shape[:-1] + (0,), mx.float32)
    return (out, lse, z, z)

def _sdpa_eff_fwd(q, k, v, bias, cu_seqlens_q, cu_seqlens_k,
                  max_seqlen_q, max_seqlen_k, dropout_p, custom_mask_type,
                  compute_log_sumexp=False, scale=None, seqlen_k=None,
                  window_size=None):
    # aten._efficient_attention_forward -> 6-tuple
    # (output[BSHD], logsumexp[BHS], philox_seed, philox_offset,
    #  max_seqlen_batch_q:int, max_seqlen_batch_k:int).
    # Unlike _scaled_dot_product_efficient_attention (BHSD), this op's
    # q/k/v are BSHD (attention.cu transposes 1<->2 in and out; lse stays
    # in BHS layout).
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        return _raise("efficient_attention_forward cu_seqlens")
    if seqlen_k is not None or window_size is not None:
        return _raise("efficient_attention_forward seqlen_k/window_size")
    if dropout_p not in (0, 0.0):
        return _raise("efficient_attention_forward dropout")
    if bias is not None and custom_mask_type != 0:
        return _raise("efficient_attention_forward bias+mask_type")
    if custom_mask_type not in (0, 1, 2):
        return _raise(f"efficient_attention_forward mask_type={custom_mask_type}")
    q, k, v = _mx(q), _mx(k), _mx(v)
    sq, sk = q.shape[1], k.shape[1]
    # BSHD -> BHSD for the mx sdpa / scores below.
    q, k, v = (mx.swapaxes(t, 1, 2) for t in (q, k, v))
    k = _gqa_expand(k, q.shape[-3])
    v = _gqa_expand(v, q.shape[-3])
    # bias is already (B, H, Sq, Sk) per attention.cu; only q/k/v/out are BSHD.
    bias = _mx(bias)
    is_causal = custom_mask_type in (1, 2)
    off = 0 if custom_mask_type == 1 else (sk - sq if custom_mask_type == 2 else 0)
    s = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    b = _sdpa_bias(q, k, is_causal, bias, causal_off=off)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=s,
                                                  mask=None if b is None else b.astype(q.dtype))
    out = mx.swapaxes(out, 1, 2)  # BHSD -> BSHD
    scores = s * mx.matmul(q.astype(mx.float32),
                           mx.swapaxes(k.astype(mx.float32), -2, -1))
    if b is not None:
        scores = scores + b.astype(mx.float32)
    lse = mx.logsumexp(scores, axis=-1)
    if not compute_log_sumexp:
        lse = mx.zeros(q.shape[:-1] + (0,), mx.float32)
    z = mx.zeros((1,), mx.int64)
    msq = int(max_seqlen_q) if max_seqlen_q is not None else sq
    msk = int(max_seqlen_k) if max_seqlen_k is not None else sk
    return (out, lse, z, z, msq, msk)

def _conv2d(a, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    # torch: a NCHW, w OIHW ; mlx.conv2d: x NHWC, w (H, W, I/groups, O)
    if groups != 1:
        return _raise("grouped conv2d")
    x = _mx(a).transpose(0, 2, 3, 1)
    wt = _mx(w).transpose(0, 2, 3, 1)  # OIHW -> O,H,W,I (mlx conv2d weight layout)
    p = tuple(padding) if isinstance(padding, (list, tuple)) else (padding, padding)
    st = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride)
    dl = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation, dilation)
    y = mx.conv2d(x, wt, stride=st, padding=p, dilation=dl)
    if b is not None:
        y = y + _mx(b).reshape(1, 1, 1, -1)
    return y.transpose(0, 3, 1, 2)

def _tup(v, n):
    return tuple(v) if isinstance(v, (list, tuple)) else (v,) * n


def _nhwc(a):
    return _mx(a).transpose(0, *range(2, a.ndim), 1)


def _nchw(a):
    return a.transpose(0, a.ndim - 1, *range(1, a.ndim - 1))


def _nll_loss_forward(a, target, weight=None, reduction=1, ignore_index=-100):
    if weight is not None:
        return _raise("nll_loss weight")
    a, t = _mx(a), _mx(target).astype(mx.int32)
    n = a.shape[0]
    lp = a[mx.arange(n), t]
    m = t != ignore_index
    tw = mx.sum(m.astype(mx.float32))
    if reduction == 0:
        return (mx.where(m, -lp, 0.0), tw)
    return (-mx.sum(mx.where(m, lp, 0.0)) / (tw if reduction == 1 else 1.0), tw)


def _gelu_backward(g, a, approximate="none"):
    # aten runs this backward in fp32 opmath and rounds to the input dtype.
    d = a.dtype
    g, a = _mx(g).astype(mx.float32), _mx(a).astype(mx.float32)
    if approximate == "tanh":
        u = math.sqrt(2.0 / math.pi) * (a + 0.044715 * a * a * a)
        t = mx.tanh(u)
        out = _mx(g) * (0.5 * (1.0 + t) + 0.5 * a * (1.0 - t * t) * math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * a * a))
    else:
        out = _mx(g) * (0.5 * (1.0 + mx.erf(a / math.sqrt(2.0))) + a * mx.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi))
    return out.astype(d)


def _tanh_backward(g, out):
    # aten tanh_backward: grad * (1 - output^2) in fp32 opmath.
    d = _mx(g).dtype
    g32, o32 = _mx(g).astype(mx.float32), _mx(out).astype(mx.float32)
    return (g32 * (1.0 - o32 * o32)).astype(d)


def _nll_loss_backward(gout, a, target, weight, reduction, ignore_index, tw):
    if weight is not None:
        return _raise("nll_loss_backward weight")
    a, t = _mx(a), _mx(target).astype(mx.int32)
    n = a.shape[0]
    m = (t != ignore_index).astype(a.dtype)
    if reduction == 1:
        g = -_mx(gout) * m / mx.maximum(_mx(tw).astype(a.dtype), 1e-12)
    else:
        g = -_mx(gout).reshape(-1)[:n] * m
    onehot = mx.equal(t.reshape(n, 1), mx.arange(a.shape[1]).reshape(1, -1))
    return mx.where(onehot, g.reshape(n, 1), mx.zeros_like(a))


def _convolution(x, w, b=None, stride=1, padding=0, dilation=1,
                 transposed=False, output_padding=0, groups=1):
    # aten.convolution: torch layouts NCHW / OIHW; mx weight (O,H,W,I), x NHWC.
    if transposed:
        return _raise("transposed convolution")
    if isinstance(output_padding, (list, tuple)) and any(output_padding):
        return _raise("convolution output_padding")
    nd = x.ndim - 2
    y = mx.conv_general(_nhwc(x), _nhwc(w), stride=_tup(stride, nd),
                        padding=_tup(padding, nd),
                        kernel_dilation=_tup(dilation, nd), groups=groups)
    if b is not None:
        y = y + _mx(b).reshape((1,) * (y.ndim - 1) + (-1,))
    return _nchw(y)


def _convolution_backward(gout, x, w, bias_sizes, stride, padding, dilation,
                          transposed, output_padding, groups, output_mask):
    # grad_input: transposed conv = conv_general(input_dilation=stride, flip=True)
    # with padding d(k-1)-p, plus output_padding folded into the hi pad.
    # grad_weight: batch->groups trick - conv(x^T [batch=Ci, ch=N],
    # gout^T [out=Co, in=N, kernel=(Ho,Wo)], stride=d, kernel_dilation=s),
    # then slice to (kh,kw). Verified vs torch to ~1e-6 (stride 2, pad 1).
    if transposed or groups != 1:
        return _raise("convolution_backward transposed/groups")
    if isinstance(output_padding, (list, tuple)) and any(output_padding):
        return _raise("convolution_backward output_padding")
    nd = x.ndim - 2
    st, pd, dl = _tup(stride, nd), _tup(padding, nd), _tup(dilation, nd)
    gm = _nhwc(gout)
    mask = output_mask if output_mask is not None else (True, True, True)
    gi = gw = gb = None
    if mask[0]:
        wt_t = _mx(w).transpose(1, *range(2, w.ndim), 0)     # (Ci,K...,Co)
        pe = [dl[i] * (w.shape[2 + i] - 1) - pd[i] for i in range(nd)]
        if min(pe) < 0:
            return _raise("convolution_backward neg pad")
        op = [x.shape[2 + i] - ((gout.shape[2 + i] - 1) * st[i]
                                + dl[i] * (w.shape[2 + i] - 1) - 2 * pd[i] + 1)
              for i in range(nd)]
        if min(op) < 0:
            return _raise("convolution_backward neg op")
        gi = _nchw(mx.conv_general(gm, wt_t, input_dilation=st,
                                   kernel_dilation=dl,
                                   padding=(pe, [pe[i] + op[i] for i in range(nd)]),
                                   flip=True))
    if mask[1]:
        xt = _mx(x).transpose(1, *range(2, x.ndim), 0)       # (Ci,S...,N)
        gt = _mx(gout).transpose(1, *range(2, gout.ndim), 0)  # (Co,S'...,N)
        c = mx.conv_general(xt, gt, stride=dl, kernel_dilation=st, padding=pd)
        c = c[tuple([slice(None)] + [slice(0, w.shape[2 + i]) for i in range(nd)]
                    + [slice(None)])]
        gw = c.transpose(c.ndim - 1, 0, *range(1, c.ndim - 1))  # (Co,Ci,K...)
    if mask[2] and bias_sizes:
        gb = mx.sum(gm, axis=(0,) + tuple(range(1, gm.ndim - 1)))
    return (gi, gw, gb)


def _conv1d(a, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    # torch: a NCL or CL (unbatched), w (O, I/groups, K) ; mlx.conv1d: x NLC, w (O, K, I/groups)
    x = _mx(a)
    unbatched = x.ndim == 2
    if unbatched:
        x = x[None]
    x = x.transpose(0, 2, 1)
    wt = _mx(w).transpose(0, 2, 1)  # torch O,I/groups,K -> mlx O,K,I/groups
    st = stride[0] if isinstance(stride, (list, tuple)) else stride
    p = padding[0] if isinstance(padding, (list, tuple)) else padding
    dl = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
    y = mx.conv1d(x, wt, stride=st, padding=p, dilation=dl, groups=groups)
    if b is not None:
        y = y + _mx(b).reshape(1, 1, -1)
    y = y.transpose(0, 2, 1)
    return y[0] if unbatched else y

def _diff(a, n=1, dim=-1, prepend=None, append=None):
    x = _mx(a)
    d = dim % x.ndim
    if prepend is not None:
        x = mx.concatenate([_mx(prepend), x], axis=d)
    if append is not None:
        x = mx.concatenate([x, _mx(append)], axis=d)

    def _s(t, lo, hi):
        idx = [slice(None)] * t.ndim
        idx[d] = slice(lo, hi)
        return t[tuple(idx)]

    for _ in range(n):
        x = _s(x, 1, x.shape[d]) - _s(x, 0, x.shape[d] - 1)
    return x

def _group_norm(a, num_groups, w=None, b=None, eps=1e-5):
    # torch NCHW; mlx 0.32 has no fast.group_norm - normalize per group.
    x = _mx(a)
    n, c = x.shape[0], x.shape[1]
    xg = x.reshape(n, num_groups, -1)
    mean = xg.mean(axis=-1, keepdims=True)
    var = ((xg - mean) ** 2).mean(axis=-1, keepdims=True)
    xn = ((xg - mean) / mx.sqrt(var + eps)).reshape(x.shape)
    if w is not None:
        xn = xn * _mx(w).reshape(1, -1, 1, 1)
    if b is not None:
        xn = xn + _mx(b).reshape(1, -1, 1, 1)
    return xn

def _interpolate(a, size=None, scale_factor=None, mode="nearest",
                 align_corners=None, recompute_scale_factor=None, **k):
    # torch NCHW; nearest-neighbor upsample (the SD UNet upsampler mode).
    if mode != "nearest":
        return _raise("interpolate mode " + str(mode))
    x = _mx(a).transpose(0, 2, 3, 1)  # NHWC
    n, h, w, c = x.shape
    if size is not None:
        th, tw = (size if isinstance(size, (list, tuple)) else (size, size))
    else:
        sf = scale_factor if isinstance(scale_factor, (list, tuple)) else (scale_factor, scale_factor)
        th, tw = int(h * sf[0]), int(w * sf[1])
    yi = (mx.arange(th) * (h / th)).astype(mx.int32)
    xi = (mx.arange(tw) * (w / tw)).astype(mx.int32)
    return x[:, yi][:, :, xi].transpose(0, 3, 1, 2)


def _topk(a, k, dim=-1, largest=True, sorted=True):
    """torch.topk parity: returns (values, indices) sorted along dim."""
    a = _mx(a)
    dim = (dim if dim is not None else -1) % a.ndim
    perm = [i for i in range(a.ndim) if i != dim] + [dim]
    at = mx.transpose(a, axes=perm)
    idx = mx.argsort(-at if largest else at, axis=-1)[..., :k]
    vals = mx.take_along_axis(at, idx, axis=-1)
    inv = [perm.index(i) for i in range(a.ndim)]
    return (mx.transpose(vals, axes=inv), mx.transpose(idx, axes=inv).astype(mx.int64))

def _mode(a, dim=-1, keepdim=False):
    """torch.mode parity: (values, indices) of the row-wise most frequent
    value, ties resolving to the smallest such value (torch CPU behavior),
    indices first-occurrence. Sort + per-row equality count: the count of
    each sorted position's value, argmax picks the first maximum, which in
    an ascending sort is the smallest modal value."""
    a = _mx(a)
    dim = (dim if dim is not None else -1) % a.ndim
    perm = [i for i in range(a.ndim) if i != dim] + [dim]
    at = mx.transpose(a, axes=perm)
    # the omarchy backend has no int64/integral argsort kernel: integral
    # sorts run on an explicit CPU stream (small ranking widths, correctness
    # first); float sorts dispatch on the device stream as usual
    if mx.issubdtype(at.dtype, mx.integer):
        with mx.stream(mx.cpu):
            idx = mx.argsort(at, axis=-1)
            v = mx.take_along_axis(at, idx, axis=-1)
    else:
        idx = mx.argsort(at, axis=-1)
        v = mx.take_along_axis(at, idx, axis=-1)
    eq = mx.expand_dims(v, -1) == mx.expand_dims(v, -2)
    counts = mx.sum(eq, axis=-1)
    # torch.mode picks the smallest modal value and its index is that
    # value's last (rightmost) occurrence. Deterministic tie handling without
    # relying on argmax tie order: maximize count, break ties toward the
    # larger sorted slot (key = count * (n+1) - slot).
    fm = mx.argmax(counts, axis=-1)
    maxc = mx.max(counts, axis=-1)
    pos = fm + maxc - 1
    pos_ax = mx.expand_dims(pos, -1) if pos.ndim < v.ndim else pos
    vals = mx.take_along_axis(v, pos_ax, axis=-1)
    index = mx.take_along_axis(idx, pos_ax, axis=-1)
    inv = [perm.index(i) for i in range(a.ndim)]
    vals = mx.transpose(vals, axes=inv)
    index = mx.transpose(index, axes=inv)
    # keepdim is the natural shape: the size-1 axis already sits at dim
    if not keepdim:
        vals = mx.squeeze(vals, dim)
        index = mx.squeeze(index, dim)
    return (vals.astype(a.dtype), index.astype(mx.int64))


def _kthvalue(a, k, dim=-1, keepdim=False):
    """torch.kthvalue parity: the k-th smallest value along dim plus its
    index. Sort + index: the ascending sort's k-1 slot. torch CPU's own
    kthvalue index is a quickselect artifact that need not point at the
    returned value (verified: value -2.136 at index 2, torch's index 7 holds
    -1.24); this lowering guarantees the index points at a position holding
    the returned value."""
    a = _mx(a)
    dim = (dim if dim is not None else -1) % a.ndim
    perm = [i for i in range(a.ndim) if i != dim] + [dim]
    at = mx.transpose(a, axes=perm)
    # integral sorts run on an explicit CPU stream (no int64 argsort kernel
    # in the omarchy backend); see _mode
    if mx.issubdtype(at.dtype, mx.integer):
        with mx.stream(mx.cpu):
            idx = mx.argsort(at, axis=-1)
    else:
        idx = mx.argsort(at, axis=-1)
    # sel is itself the original position of the k-th smallest (a slot value
    # of the argsort output); taking idx at sel would double-indirect
    sel = idx[..., k - 1: k]
    vals = mx.take_along_axis(at, sel, axis=-1)
    sel_idx = sel
    inv = [perm.index(i) for i in range(a.ndim)]
    vals = mx.transpose(vals, axes=inv)
    sel_idx = mx.transpose(sel_idx, axes=inv)
    # keepdim is the natural shape: the size-1 axis already sits at dim
    if not keepdim:
        vals = mx.squeeze(vals, dim)
        sel_idx = mx.squeeze(sel_idx, dim)
    return (vals.astype(a.dtype), sel_idx.astype(mx.int64))


def _is_integral_dtype(dt):
    return mx.issubdtype(dt, mx.integer) or mx.issubdtype(dt, mx.bool_)


def _is_floatish_dtype(dt):
    return (mx.issubdtype(dt, mx.floating)
            or mx.issubdtype(dt, mx.complexfloating))


def _dtype_of(v):
    # dtype of a resolved operand: mx arrays carry theirs; torch Scalar
    # arguments arrive as python numbers and promote like wrapped scalars
    # (float scalar -> default float, int scalar stays integral).
    if isinstance(v, mx.array):
        return v.dtype
    if isinstance(v, bool):
        return mx.bool_
    if isinstance(v, int):
        return mx.int64
    if isinstance(v, float):
        return mx.float32
    return getattr(v, "dtype", mx.float32)


def _torch_comp_dtype(dd, dr, md, mr):
    # torch type promotion (docs "type promotion dtypes"): keep the highest
    # category (float over int; bool is rejected before this runs), then
    # promote among the DIMENSIONED participants of that category when any
    # exist, falling back to the 0-dim ones only when none do. A rank-zero
    # fp32 rhs therefore does not lift a dimensioned fp16 dest (torch
    # computes fp16 and can produce -inf), while a shape-[1] fp32 rhs
    # promotes to fp32.
    pair = ((dd, dr), (md, mr))
    if _is_floatish_dtype(dd) or _is_floatish_dtype(md):
        use = [(t, r) for t, r in pair if _is_floatish_dtype(t)]
        dim = [(t, r) for t, r in use if r > 0]
        use = dim or use
    else:
        dim = [(t, r) for t, r in pair if r > 0]
        use = dim or list(pair)
    dts = [t for t, _ in use]
    return dts[0] if len(dts) == 1 else mx.result_type(*dts)


def _sub_inplace(a, b, alpha=1, **k):
    # aten.sub_.Tensor(self, other, alpha) eager contract: torch promotes,
    # computes, then requires the result to fit the destination shape and
    # cast back to the destination dtype (_inplace_writeback raises the
    # torch-shaped errors for growth / float->int, which surface as a CPU
    # fallback running torch's own check). Reject bool operands first, as
    # the compiled path does. Scale in the destination width for integral
    # operands: (d - k*rhs) mod 2^n equals torch's promote-then-wrap for
    # integral alpha, so a narrow rhs times a wide alpha cannot wrap early;
    # a float alpha promotes the computation to float and the writeback
    # rejects the float->int cast-back, matching torch.
    if isinstance(alpha, bool):
        # torch: boolean alpha is only valid for boolean results, and
        # boolean subtraction is rejected below
        raise RuntimeError("Boolean alpha only supported for Boolean results.")
    d = _mx(a)
    mb = _mx(b)
    # Checks run on the converted mlx dtypes (torch-side _dtype_of maps to
    # torch dtypes, so the earlier torch-dtype comparisons never matched):
    # bool operands are rejected like the compiled path does, and complex
    # operands stay on the torch path - torch rejects a complex result cast
    # into a real in-place destination, and the lowering does not implement
    # torch's real/complex promotion widths; this build has no complex128.
    if d.dtype == mx.bool_ or mb.dtype == mx.bool_:
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is "
            "not supported. If you are trying to invert a mask, "
            "use the `~` or `logical_not()` operator instead.")
    if d.dtype == mx.complex64 or mb.dtype == mx.complex64:
        raise RuntimeError("Complex operands are not lowered for in-place subtract.")
    # Compute in the torch operation dtype: both operands promoted, a
    # python float alpha lifting an integral pair to the default float
    # (a python int alpha is weak and changes nothing). Cast both sides
    # there BEFORE scaling so a narrow operand (int8, float16) cannot
    # wrap or overflow early - fp16 rhs 70000 * 0.5 stays finite in the
    # fp32 computation, int8 100 * 2 stays 200 instead of wrapping. The
    # destination only rules the final writeback: shape check, then the
    # comp->dest cast with float->int rejected (torch's in-place error).
    comp = _torch_comp_dtype(d.dtype, d.ndim, mb.dtype, mb.ndim)
    if (not _is_floatish_dtype(d.dtype) and not _is_floatish_dtype(mb.dtype)
            and isinstance(alpha, float)):
        comp = mx.float32
    dc = d.astype(comp) if comp != d.dtype else d
    mb = mb.astype(comp) if comp != mb.dtype else mb
    return _inplace_writeback(d, dc - mb * alpha)


def _rsub(a, b, alpha=1, **k):
    # aten.rsub.Scalar(self, other, alpha): other - alpha * self. Torch
    # rejects subtraction on bool operands, and a floating alpha only when
    # the promoted operand type is integral (int64 self + float other is
    # legal and returns float).
    da, db = _dtype_of(a), _dtype_of(b)
    if da == mx.bool_ or db == mx.bool_:
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not "
            "supported. If you are trying to invert a mask, use the `~` or "
            "`logical_not()` operator instead.")
    if isinstance(alpha, float) and _is_integral_dtype(mx.result_type(da, db)):
        raise RuntimeError(
            "For integral input tensors, the argument alpha must not be a "
            "floating point number")
    return _mx(b) - _mx(a) * alpha


def _alpha_arith(a, b, alpha, sign):
    # aten.add/sub (self, other, *, alpha=1): self +/- alpha*other. Torch
    # rejects sub on bool operands and a floating alpha when the promoted
    # operand type is integral (int64 self + float other is legal and
    # returns float).
    da, db = _dtype_of(a), _dtype_of(b)
    if sign < 0 and (da == mx.bool_ or db == mx.bool_):
        raise RuntimeError(
            "Subtraction, the `-` operator, with a bool tensor is not "
            "supported. If you are trying to invert a mask, use the `~` or "
            "`logical_not()` operator instead.")
    rt = mx.result_type(da, db)
    if isinstance(alpha, float) and _is_integral_dtype(rt):
        raise RuntimeError(
            "For integral input tensors, the argument alpha must not be a "
            "floating point number")
    # Compute in the promoted dtype: alpha*other in other's own dtype can
    # overflow before promotion (f16 40000 * 2 = inf vs f32 80000).
    ma, mb = _mx(a), _mx(b)
    ma = ma.astype(rt) if isinstance(ma, mx.array) else mx.array(ma, dtype=rt)
    mb = mb.astype(rt) if isinstance(mb, mx.array) else mx.array(mb, dtype=rt)
    return _ar(ma + sign * (alpha * mb))


def _ar(v):
    # Scalar-only calls (aten.mul.default(2, 3)) reach the specs with no mx
    # operand; a bare Python number would leave the graph output unboxed.
    # Box with torch's scalar dtypes (int64 for ints) — assert_close checks.
    if isinstance(v, mx.array):
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        return mx.array(v, dtype=mx.int64)
    return mx.array(v)


def _atan2(y, x):
    # IEEE signed-zero quadrants: atan2(+0, -0) = +pi, atan2(-0, -0) = -pi.
    # The Vulkan atan2 kernel drops the sign of a zero x; 1/+-0 = +-inf
    # recovers it (no mx.signbit in the wheel). Only the (+/-0, -0) pairs
    # are wrong: nonzero and NaN y stay on mx.atan2.
    y, x = _mx(y), _mx(x)
    r = mx.atan2(y, x)
    x_neg0 = (x == 0) & (1.0 / x < 0)
    fix = mx.where(1.0 / y < 0, -math.pi, math.pi).astype(r.dtype)
    return mx.where(x_neg0 & (y == 0), fix, r)


def _astype(v, dt):
    v = _mx(v)
    return v.astype(dt) if isinstance(v, mx.array) else v


def _has_0d_mixed(*vs):
    # torch promotes 0-dim tensors weakly — they cannot raise the result
    # above a non-0-dim operand's dtype. Mixed-dtype inputs with a 0-dim
    # operand follow rules subtle enough to leave to the torch fallback.
    ts = [v for v in vs if isinstance(v, (mx.array, torch.Tensor))]
    if not any(v.ndim == 0 for v in ts):
        return False
    return len({_dtype_of(v) for v in ts}) > 1


def _addcmul(a, t1, t2, value=1, **k):
    # aten.addcmul(input, t1, t2, *, value=1): input + value*t1*t2. Bool
    # has no kernel; on integral operands the wrapped value scalar takes
    # the tensor dtype (fractional value truncates). Operands evaluate at
    # the common dtype — fp32 opmath for the low-precision floats — so
    # neither narrow ints nor fp16 intermediates overflow before
    # promotion; value applies to t1 first like torch.
    if _has_0d_mixed(a, t1, t2):
        return _raise("addcmul 0-dim mixed-dtype operand")
    dt = mx.result_type(mx.result_type(_dtype_of(a), _dtype_of(t1)),
                        _dtype_of(t2))
    if dt == mx.bool_:
        raise NotImplementedError(
            '"addcmul" not implemented for \'Bool\'')
    if isinstance(value, float) and _is_integral_dtype(dt):
        value = int(value)
    cd = mx.float32 if (dt == mx.float16 or dt == mx.bfloat16) else dt
    return (_astype(a, cd) + (value * _astype(t1, cd))
            * _astype(t2, cd)).astype(dt)


def _addcdiv(a, t1, t2, value=1, **k):
    # aten.addcdiv(input, t1, t2, *, value=1): input + value*t1/t2. Torch
    # rejects integral division operands regardless of self's dtype
    # (integer division unsupported); same common-dtype eval as addcmul.
    if _has_0d_mixed(a, t1, t2):
        return _raise("addcdiv 0-dim mixed-dtype operand")
    if _is_integral_dtype(
            mx.result_type(_dtype_of(t1), _dtype_of(t2))):
        raise RuntimeError(
            "Integer division with addcdiv is no longer supported, and in "
            "a future release addcdiv will perform a true division like "
            "tensor1.true_divide(tensor2)")
    dt = mx.result_type(mx.result_type(_dtype_of(a), _dtype_of(t1)),
                        _dtype_of(t2))
    cd = mx.float32 if (dt == mx.float16 or dt == mx.bfloat16) else dt
    return (_astype(a, cd) + (value * _astype(t1, cd))
            / _astype(t2, cd)).astype(dt)


def _arange(*a, **k):
    # torch.arange(end) | (start, end[, step]); dynamo may pass everything
    # as kwargs (start/end/step/dtype/device) — fold them into positional.
    args = list(a)
    if not args and "start" in k:
        args.append(k["start"])
    if "end" in k and len(args) < 2:
        args.append(k["end"])
    if "step" in k and len(args) < 3:
        if len(args) == 1:
            args.insert(0, 0)
        args.append(k["step"])
    dtype = _MX_DT.get(k.get("dtype"))
    if dtype is None and all(type(v) is int for v in args):
        dtype = mx.int64   # torch.arange defaults to int64 for integral args
    return mx.arange(*args, dtype=dtype)



def _histc(a, bins=100, min=None, max=None, **kw):
    """torch.histc parity: equal-width bins over [min, max], out-of-range dropped,
    the right edge inclusive. min==0 and max==0 (or both None) = the input min/max."""
    x = mx.flatten(_mx(a).astype(mx.float32))
    nb = int(bins)
    if nb <= 0:
        raise ValueError("histc needs bins >= 1")
    auto = (min is None or min == 0) and (max is None or max == 0)
    lo = float(mx.min(x).item()) if auto else float(min)
    hi = float(mx.max(x).item()) if auto else float(max)
    if hi < lo:
        raise ValueError("histc needs max >= min")
    if hi == lo:
        # torch widens a collapsed range to [v-0.5, v+0.5]; without this the
        # width is 0 and (x - lo) / width puts the whole mass in bin 0
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / nb
    idx = mx.clip(mx.floor((x - lo) / width).astype(mx.int32), 0, nb - 1)
    keep = ((x >= lo) & (x <= hi)).astype(mx.float32)
    return (mx.eye(nb, nb)[idx].astype(mx.float32) * keep[:, None]).sum(axis=0)


def _histogram(a, bins=10, range=None, weight=None, density=False, **kw):
    """torch.histogram parity: returns (hist, bin_edges). An int bin count builds
    equal-width edges over the (lo, hi) range (default = the input min/max); a list
    of edges is used as-is. Per-sample weights replace counts; density=True
    normalizes so mass * width integrates to 1. Out-of-range samples drop."""
    x = mx.flatten(_mx(a).astype(mx.float32))
    if isinstance(bins, (list, tuple)):
        edges = mx.array([float(e) for e in bins])
        nb = len(bins) - 1
    else:
        nb = int(bins)
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo = float(mx.min(x).item())
            hi = float(mx.max(x).item())
        edges = lo + (hi - lo) * (mx.arange(nb + 1, dtype=mx.float32) / nb)
    if nb <= 0:
        raise ValueError("histogram needs bins >= 1")
    lo = float(edges[0].item())
    hi = float(edges[-1].item())
    if hi < lo:
        raise ValueError("histogram needs a rising edge sequence")
    width = (hi - lo) / nb
    keep = ((x >= lo) & (x <= hi)).astype(mx.float32)
    interior = edges[1:-1]
    if interior.shape[0] > 0:
        idx = (x[:, None] >= interior[None, :]).astype(mx.int32).sum(axis=1)
    else:
        idx = mx.zeros_like(x).astype(mx.int32)
    onehot = mx.eye(nb, nb)[idx].astype(mx.float32) * keep[:, None]
    if weight is not None:
        w = _mx(weight).astype(mx.float32) * keep
        hist = (onehot * w[:, None]).sum(axis=0)
    else:
        hist = onehot.sum(axis=0)
    if density:
        # torch normalizes by the PER-BIN widths, not one uniform width —
        # a non-uniform (list) edge sequence has unequal bin widths
        widths = edges[1:] - edges[:-1]
        hist = hist / (hist.sum() * widths)
    return hist.astype(_mx(a).dtype), edges



def _multinomial(probs, num_samples, replacement=False, **kw):
    """torch.multinomial parity: draws from probability rows along the last dim.
    replacement=False is top-k of Gumbel-perturbed log-probs (exact);
    replacement=True is independent categorical draws on the logits."""
    p = _mx(probs)
    lp = mx.log(p)
    if replacement:
        return mx.random.categorical(lp, num_samples=num_samples).astype(mx.int64)
    return mx.argsort(-(lp + mx.random.gumbel(p.shape)), axis=-1)[..., :num_samples].astype(mx.int64)

def _bincount(a, weights=None, minlength=0):
    """aten.bincount: out[i] = count (or sum of weights) of positions where
    a == i. Output length = max(minlength, max(a) + 1); int64 out unless
    weights is given, then the weights dtype.
    The backend's Scatter::Sum has no 64-bit accumulator, so counts
    accumulate in int32 (exact: counts are bounded by the input length)
    and cast to the torch output dtype."""
    a = _mx(a)
    if a.ndim != 1:
        return _raise("bincount ndim>1")
    n = int(minlength)
    if a.size:
        if int(np.array(mx.min(a)).item()) < 0:
            raise ValueError("bincount only supports 1-d non-negative integral inputs")
        n = max(n, int(np.array(mx.max(a)).item()) + 1)
    if weights is None:
        w, dt = mx.ones(a.shape, dtype=mx.int32), mx.int64
    else:
        w, dt = _mx(weights), _mx(weights).dtype
        if w.dtype == mx.float64:
            # the GPU wheel has no float64: mx silently casts an f64 array
            # to float32, so the only torch-exact route is the fallback
            # (torch accumulates in true f64 and returns f64)
            return _raise("bincount float64 weights fall back to torch for exact f64 sums")
        elif w.dtype not in (mx.float32, mx.float16, mx.bfloat16):
            # every integer weight dtype accumulates through the int32
            # Scatter::Sum: refuse when the SUM can leave int32
            # (max|w| * a.size >= 2**31) — the old catch-all cast truncated
            # uint64 to 0 and wrapped int8 sums, and the int64-only guard
            # missed the same overflow on int32/uint32 weights. _raise
            # falls back to torch, which returns exact sums for any width.
            if w.size:
                lo = int(np.array(mx.min(w)).item())
                hi = int(np.array(mx.max(w)).item())
                if max(abs(lo), abs(hi)) * a.size >= 2 ** 31:
                    return _raise("bincount integral weights exceed the int32 scatter sum range")
            w = w.astype(mx.int32)
    out = mx.zeros((n,), dtype=w.dtype)
    if a.size:
        out = out.at[a.astype(mx.int32)].add(w)
    return out.astype(dt)

def _bitwise_or_logical(a, b, bitop, logop):
    a, b = _mx(a), _mx(b)
    _float_dts = (mx.float32, mx.float16, mx.bfloat16)
    if a.dtype in _float_dts or b.dtype in _float_dts:
        _b = getattr(mx, "bool_", None) or mx.uint8
        return logop(a.astype(_b), b.astype(_b))
    return bitop(a, b)


_KIND_GROUPS = None
def _dtype_kind(dt):
    global _KIND_GROUPS
    if _KIND_GROUPS is None:
        g = {"bool": ("bool_",),
             "int": tuple(p + str(w) for p in ("int", "uint") for w in (8, 16, 32, 64)),
             "float": ("float16", "bfloat16", "float32", "float64"),
             "complex": ("complex64", "complex128")}
        _KIND_GROUPS = {k: frozenset(getattr(mx, n) for n in names if hasattr(mx, n))
                        for k, names in g.items()}
    return next((k for k, dts in _KIND_GROUPS.items() if dt in dts), str(dt))


def _inplace_result(upd, dst):
    # Torch in-place contract: the broadcast result may not grow the
    # destination shape, and its dtype must cast back to the destination's
    # kind (int64->int8 stays, int64->bool raises). Refusals leave the op to
    # the torch fallback, which raises the same error.
    dst = _mx(dst)
    if tuple(upd.shape) != tuple(dst.shape):
        _raise(f"in-place result shape {tuple(upd.shape)} vs dest {tuple(dst.shape)}")
    if upd.dtype != dst.dtype:
        if _dtype_kind(upd.dtype) != _dtype_kind(dst.dtype):
            _raise(f"in-place result {upd.dtype} cannot cast to dest {dst.dtype}")
        upd = upd.astype(dst.dtype)
    return upd


# ---------------- pooling (PT-P) ----------------
# Pure-mx lowerings: fwd via strided window-stack, bwd via zero-insert
# (fractional stride) + per-offset shift/accumulate. Static shapes only.

def _pool_hw(v, default=None):
    if v is None or (isinstance(v, (list, tuple)) and len(v) == 0):
        return default
    t = tuple(v) if isinstance(v, (list, tuple)) else (v, v)
    return (int(t[0]), int(t[1]))


def _pool_out(in_sz, k, s, p, dil, ceil_mode):
    o = (in_sz + 2 * p - dil * (k - 1) - 1 + (s - 1 if ceil_mode else 0)) // s + 1
    if (o - 1) * s >= in_sz + p:
        o -= 1
    return max(int(o), 0)


def _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw, pad_value=0.0):
    # (N,C,Hp,Wp) -> (N,C,oh,ow,K), K = kh*kw row-major window offsets.
    # ceil_mode lets the last window start inside the pad and overrun the
    # padded extent; grow the array on the bottom/right so those positions
    # read pad_value instead of slicing ragged.
    need_h = (oh - 1) * sh + (kh - 1) * dh + 1
    need_w = (ow - 1) * sw + (kw - 1) * dw + 1
    if need_h > xp.shape[2] or need_w > xp.shape[3]:
        xp = mx.pad(xp, ((0, 0), (0, 0),
                         (0, max(need_h - xp.shape[2], 0)),
                         (0, max(need_w - xp.shape[3], 0))),
                    constant_values=pad_value)
    cols = [xp[:, :, i * dh: i * dh + (oh - 1) * sh + 1: sh,
               j * dw: j * dw + (ow - 1) * sw + 1: sw]
            for i in range(kh) for j in range(kw)]
    return mx.stack(cols, axis=-1)


def _unstride(g, oh, ow, sh, sw):
    # (N,C,oh,ow) -> (N,C,(oh-1)*sh+1,(ow-1)*sw+1): values at strided spots,
    # zeros between (insert s-1 zeros after each element along each axis)
    n, c = g.shape[0], g.shape[1]
    g = mx.pad(mx.reshape(g, (n, c, oh, ow, 1)),
               ((0, 0), (0, 0), (0, 0), (0, 0), (0, sw - 1)))
    g = mx.reshape(g, (n, c, oh, ow * sw))
    g = mx.pad(mx.reshape(g, (n, c, oh, 1, ow * sw)),
               ((0, 0), (0, 0), (0, 0), (0, sh - 1), (0, 0)))
    g = mx.reshape(g, (n, c, oh * sh, ow * sw))
    return g[:, :, : (oh - 1) * sh + 1, : (ow - 1) * sw + 1]


def _pool_scatter(contrib, hp, wp, kh, kw, sh, sw, dh, dw):
    # contrib (N,C,oh,ow,K) -> (N,C,hp,wp): each offset's field shifted into
    # its window position and summed (handles overlap stride < kernel)
    n, c, oh, ow = contrib.shape[0], contrib.shape[1], contrib.shape[2], contrib.shape[3]
    acc = mx.zeros((n, c, hp, wp), dtype=contrib.dtype)
    for i in range(kh):
        for j in range(kw):
            g = _unstride(contrib[..., i * kw + j], oh, ow, sh, sw)
            g = mx.pad(g, ((0, 0), (0, 0), (i * dh, hp), (j * dw, wp)))
            acc = acc + g[:, :, :hp, :wp]
    return acc


def _pool_shape(a, kernel_size, stride, padding, dilation, ceil_mode):
    kh, kw = _pool_hw(kernel_size)
    sh, sw = _pool_hw(stride, (kh, kw))
    ph, pw = _pool_hw(padding, (0, 0))
    dh, dw = _pool_hw(dilation, (1, 1))
    n, c, h, w = a.shape
    oh = _pool_out(h, kh, sh, ph, dh, ceil_mode)
    ow = _pool_out(w, kw, sw, pw, dw, ceil_mode)
    return (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw)


def _max_pool2d(a, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False):
    x = _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, dilation, ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-mx.inf)
    win = _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw,
                        pad_value=-mx.inf)
    out = win.max(axis=-1)
    if not return_indices:
        return out
    idx = win.argmax(axis=-1)
    ii, jj = idx // kw, idx % kw
    hi = mx.arange(oh).reshape(1, 1, oh, 1) * sh + ii * dh - ph
    wi = mx.arange(ow).reshape(1, 1, 1, ow) * sw + jj * dw - pw
    return out, (hi * w + wi).astype(mx.int64)


def _max_pool2d_bwd(grad_output, a, kernel_size, stride=None, padding=0,
                    dilation=1, ceil_mode=False, indices=None):
    g, x = _mx(grad_output), _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, dilation, ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-mx.inf)
    win = _pool_windows(xp, oh, ow, kh, kw, sh, sw, dh, dw,
                        pad_value=-mx.inf)
    idx = win.argmax(axis=-1)
    mask = mx.equal(idx[..., None], mx.arange(kh * kw)).astype(g.dtype)
    contrib = g[..., None] * mask
    acc = _pool_scatter(contrib, h + 2 * ph, w + 2 * pw, kh, kw, sh, sw, dh, dw)
    return acc[:, :, ph:ph + h, pw:pw + w]


def _avg_pool2d(a, kernel_size, stride=None, padding=0, ceil_mode=False,
                count_include_pad=True, divisor_override=None):
    x = _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, dh, dw) = \
        _pool_shape(x, kernel_size, stride, padding, (1, 1), ceil_mode)
    xp = mx.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    s = _pool_windows(xp, oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    if divisor_override is not None:
        return s / float(divisor_override)
    # Divisor = window positions inside the padded input; ceil overrun
    # never counts. include_pad counts the p-pad region itself.
    ones = mx.ones((n, c, h, w), dtype=x.dtype)
    cnt = _pool_windows(
        mx.pad(ones, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
               constant_values=1.0 if count_include_pad else 0.0),
        oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    return s / cnt


def _avg_pool2d_backward(grad_output, a, kernel_size, stride=None, padding=0,
                         ceil_mode=False, count_include_pad=True,
                         divisor_override=None):
    g, x = _mx(grad_output), _mx(a)
    (n, c, h, w, oh, ow), (kh, kw, sh, sw, ph, pw, _, _) = \
        _pool_shape(x, kernel_size, stride, padding, (1, 1), ceil_mode)
    if divisor_override is not None:
        div = float(divisor_override)
    else:
        ones = mx.ones((n, c, h, w), dtype=x.dtype)
        div = _pool_windows(
            mx.pad(ones, ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                   constant_values=1.0 if count_include_pad else 0.0),
            oh, ow, kh, kw, sh, sw, 1, 1).sum(axis=-1)
    contrib = mx.broadcast_to((g / div)[..., None],
                              (n, c, oh, ow, kh * kw))
    acc = _pool_scatter(contrib, h + 2 * ph, w + 2 * pw, kh, kw, sh, sw, 1, 1)
    return acc[:, :, ph:ph + h, pw:pw + w]


def _adaptive_avg_pool2d(a, output_size):
    x = _mx(a)
    n, c, h, w = x.shape
    if output_size is None:
        output_size = (h, w)
    elif isinstance(output_size, (list, tuple)):
        output_size = tuple(h if s is None else s for s in output_size[:1]) + \
            tuple(w if s is None else s for s in output_size[1:2])
    oh, ow = _pool_hw(output_size)
    if oh == h and ow == w:
        return x
    rows = []
    for i in range(oh):
        hs = i * h // oh
        he = (i + 1) * h // oh + (1 if (i + 1) * h % oh else 0)  # ceil
        cols = []
        for j in range(ow):
            ws = j * w // ow
            we = (j + 1) * w // ow + (1 if (j + 1) * w % ow else 0)
            cols.append(x[:, :, hs:he, ws:we].mean(axis=(-2, -1)))
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2)


def _adaptive_avg_pool2d_backward(grad_output, a):
    g, x = _mx(grad_output), _mx(a)
    n, c, h, w = x.shape
    oh, ow = g.shape[2], g.shape[3]
    if oh == h and ow == w:
        return g
    acc = mx.zeros((n, c, h, w), dtype=g.dtype)
    for i in range(oh):
        hs = i * h // oh
        he = (i + 1) * h // oh + (1 if (i + 1) * h % oh else 0)
        for j in range(ow):
            ws = j * w // ow
            we = (j + 1) * w // ow + (1 if (j + 1) * w % ow else 0)
            cnt = float((he - hs) * (we - ws))
            mask = mx.pad(mx.ones((1, 1, he - hs, we - ws), dtype=g.dtype),
                          ((0, 0), (0, 0), (hs, h - he), (ws, w - we)))
            acc = acc + g[:, :, i, j][:, :, None, None] * (mask / cnt)
    return acc


def _bn3(x, w, b, rm, rv, training, momentum, eps):
    """batch-norm forward, channel dim 1 (N,C,... layout). Training
    normalizes with the biased batch variance (torch semantics); eval
    reads the running stats. Returns (out, save_mean, save_invstd).

    The normalize runs in an (N*H*W, C) layout: multi-axis mean/var over
    a 4-D tensor produced wrong values on the Vulkan backend (PAR-41),
    while the 2-D column reductions are exact."""
    x = _mx(x)
    C = x.shape[1]
    N = x.shape[0]
    x2 = mx.moveaxis(x, 1, -1).reshape(-1, C)
    if training:
        mean = mx.mean(x2, axis=0)
        var = mx.var(x2, axis=0)
    else:
        mean = mx.reshape(_mx(rm), (-1,))
        var = mx.reshape(_mx(rv), (-1,))
    invstd = mx.rsqrt(var + eps)
    y = (x2 - mean) * invstd
    if w is not None:
        y = y * mx.reshape(_mx(w), (1, C))
    if b is not None:
        y = y + mx.reshape(_mx(b), (1, C))
    y = mx.moveaxis(y.reshape(N, *x.shape[2:], C), -1, 1)
    return y, mean, invstd

def _bn_legit_functional(x, w, b, rm, rv, training, momentum, eps):
    """aten._native_batch_norm_legit_functional (aot train path): same core as
    _bn3 but returns updated running stats: (out, save_mean, save_invstd,
    new_running_mean, new_running_var). Torch updates running_var with the
    UNBIASED batch var while normalizing with the biased one. Uses the same
    (N*H*W, C) layout as _bn3 (PAR-41)."""
    x = _mx(x)
    out, sm, si = _bn3(x, w, b, rm, rv, training, momentum, eps)
    rmx, rvx = _mx(rm), _mx(rv)
    if training:
        C = x.shape[1]
        x2 = mx.moveaxis(x, 1, -1).reshape(-1, C)
        n = x2.shape[0]
        var_b = mx.var(x2, axis=0)
        unbias = var_b * (n / (n - 1)) if n > 1 else var_b
        new_rm = (1.0 - momentum) * rmx + momentum * sm
        new_rv = (1.0 - momentum) * rvx + momentum * unbias
    else:
        new_rm, new_rv = rmx, rvx
    return out, sm, si, new_rm, new_rv


def _select_backward(g, sizes, dim, index):
    """aten.select_backward: grad scattered into zeros shaped like the input."""
    g = _mx(g)
    sizes = [int(s) for s in sizes]
    dim = int(dim) % len(sizes)
    index = int(index) % sizes[dim]
    shape = [1] * len(sizes)
    shape[dim] = sizes[dim]
    mask = (mx.arange(sizes[dim]) == index).reshape(shape)
    return mx.where(mask, mx.expand_dims(g, dim), mx.zeros((), dtype=g.dtype))


def _bn_backward(dy, x, w, rm, rv, save_mean, save_invstd, training, eps, out_mask):
    """aten.native_batch_norm_backward: (dx, dw, db) honoring output_mask.
    Same (N*H*W, C) layout as _bn3 (PAR-41). ATen accumulates in the
    stats dtype (fp32 for 16-bit inputs), so compute there and cast
    outputs back; dw/db are output whenever the mask asks, weight or not."""
    dy, x = _mx(dy), _mx(x)
    C = x.shape[1]
    N = x.shape[0]
    acc = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
    pdt = _mx(w).dtype if w is not None else x.dtype
    dy2 = mx.moveaxis(dy, 1, -1).reshape(-1, C).astype(acc)
    x2 = mx.moveaxis(x, 1, -1).reshape(-1, C).astype(acc)
    if training:
        mean = _mx(save_mean).astype(acc)
        invstd = _mx(save_invstd).astype(acc)
    else:
        mean = mx.reshape(_mx(rm), (-1,)).astype(acc)
        invstd = mx.rsqrt(mx.reshape(_mx(rv), (-1,)).astype(acc) + eps)
    wb = mx.reshape(_mx(w), (1, C)).astype(acc) if w is not None else 1.0
    xh = (x2 - mean) * invstd
    if training:
        dx2 = (dy2 - mx.mean(dy2, axis=0)
               - xh * mx.mean(dy2 * xh, axis=0)) * invstd * wb
    else:
        dx2 = dy2 * invstd * wb
    dw = mx.sum(dy2 * xh, axis=0).astype(pdt) if out_mask[1] else None
    db = mx.sum(dy2, axis=0).astype(pdt) if out_mask[2] else None
    dx = (mx.moveaxis(dx2.reshape(N, *x.shape[2:], C), -1, 1).astype(x.dtype)
          if out_mask[0] else None)
    return dx, dw, db

# ---------------- lowering tables ----------------

# torch-level function objects -> lowering. Built lazily below so missing
# attributes on any given torch build can't break the import.
def _embedding_any(a, b, *a_, **k_):
    """Order-agnostic embedding: the weight is the float 2-D tensor, the
    indices are the integer tensor - F.embedding and aten.embedding name
    them in opposite orders, so take them by dtype."""
    wa, wb = _mx(a), _mx(b)
    if _has_bool_index((wa, wb)):
        return _raise("aten.embedding: a bool index tensor is a mask in aten, not an int index")
    idx_is_a = wa.dtype in (mx.int32, mx.int64, mx.uint32)
    w = wb if idx_is_a else wa
    idx = wa if idx_is_a else wb
    return w[idx.astype(mx.int32)]

def _embedding(weight, idx):
    """aten.embedding(weight, indices) order."""
    return _mx(weight)[_int_index(idx, "aten.embedding")]

def _embedding_f(input, weight, *a_, **k_):
    """F.embedding(input, weight) order."""
    return _embedding(weight, input)

def _coerce_index(v):
    # fx graphs under dynamic shapes carry torch.SymInt/SymBool indices;
    # mx slicing needs concrete python scalars. Coerce recursively through
    # slices and tuples; tensors/arrays pass through for advanced indexing.
    if isinstance(v, torch.SymInt):
        return int(v)
    if isinstance(v, torch.SymBool):
        return bool(v)
    if isinstance(v, torch.SymFloat):
        return float(v)
    if isinstance(v, slice):
        return slice(_coerce_index(v.start), _coerce_index(v.stop),
                     _coerce_index(v.step))
    if isinstance(v, (tuple, list)):
        return type(v)(_coerce_index(x) for x in v)
    return v


def _has_bool_index(idx):
    # aten reads a bool index tensor as a mask (masked positions), never as
    # 0/1 integers - and the backend's gather path treats a bool array as
    # 0/1 silently, so every index-family site defers on bool rather than
    # read or write wrong positions.
    if isinstance(idx, (tuple, list)):
        return any(_has_bool_index(v) for v in idx)
    if isinstance(idx, mx.array):
        return mx.issubdtype(idx.dtype, mx.bool_)
    if isinstance(idx, torch.Tensor):
        return idx.dtype == torch.bool
    return False


def _int_index(i, name):
    # an int-typed index for take/gather/index_select-family ops: defer a
    # bool tensor instead of casting a mask to 0/1.
    if _has_bool_index(i):
        return _raise(f"{name}: a bool index tensor is a mask in aten, not an int index")
    return _mx(i).astype(mx.int32)


def _bool_mask(m, name):
    # masked_fill's mask is bool by contract; a non-bool arg defers to the
    # fallback rather than silently astype (aten raises on non-bool masks).
    mm = _mx(m)
    if not mx.issubdtype(mm.dtype, mx.bool_):
        return _raise(f"{name}: mask must be a bool tensor")
    return mm


def _aten_index(a, idx):
    # aten.index(self, indices): a bool entry is a mask; the backend's tuple
    # gather would read it as 0/1 positions, so defer.
    if _has_bool_index(idx):
        return _raise("aten.index: a bool index tensor is a mask in aten, not an int index")
    return a[tuple(_mx(i) if isinstance(i, (torch.Tensor, list)) else i
                   for i in (idx if isinstance(idx, (list, tuple)) else (idx,)))]


def _local_scalar_dense(a):
    """aten._local_scalar_dense: the .item() graph op (the whisper
    getitem-adjacent family) - hand back the Python scalar a 0-dim tensor
    carries so scalar branches stay Python without a torch round-trip."""
    a = _mx(a)
    if a.ndim != 0 and a.size != 1:
        _raise(f"_local_scalar_dense on shape {tuple(a.shape)}")
    return a.item()


def _float_scalar(t, *a, **k):
    """aten.Float (the Scalar and Tensor overloads share the opname): torch
    .float() semantics - a Python number becomes a 0-dim fp32 tensor, any
    tensor (0-dim or not, any dtype) is cast to fp32. Casting unconditionally
    matters: int64 tensors previously slipped through uncast."""
    if isinstance(t, (int, float)):
        return mx.array(float(t))
    return _mx(t).astype(mx.float32)


def _odd_passthrough(f, a):
    # odd unary f on a backend that drops edge cases: pass the input through
    # at zeros (keeps -0's sign) and infinities (keeps the signed inf).
    a = _mx(a)
    return mx.where((a == 0) | mx.isinf(a), a, f(a))


def _sinh(a):
    return _odd_passthrough(mx.sinh, a)


def _asinh(a):
    return _odd_passthrough(mx.asinh, a)


def _cosh(a):
    # even function: cosh(+/-inf) = +inf; zeros need no sign guard.
    a = _mx(a)
    return mx.where(mx.isinf(a), mx.abs(a), mx.cosh(a))


def _acosh(a):
    # acosh(+inf) = +inf; x < 1 stays NaN like torch.
    a = _mx(a)
    return mx.where(a == float("inf"), a, mx.acosh(a))


# aten sampling family (uniform/cauchy/exponential/geometric/
# log_normal/normal.*) deliberately stays on torch: mx.random cannot
# reproduce torch's generator-state consumption, exclusive-bound
# semantics, or weak promotion — the CPU fallback keeps the RNG
# contract exact. test_random_family pins the deferral.


def _blas_beta(b, beta):
    # aten beta=0 ignores `input` entirely, NaN/inf included; a bare
    # `beta * input` would propagate that payload into the result.
    b = _mx(b)
    if beta == 0:
        return mx.zeros(b.shape, b.dtype)
    return beta * b


def _gn_native(a, w, b, N, C, HxW, group, eps):
    """aten.native_group_norm: normalize over (C/group * HxW) per
    (N, group); returns (out shaped like a, mean(N,group),
    rstd(N,group)) like aten."""
    x = _mx(a).reshape(N, group, (C // group) * HxW)
    mu = mx.mean(x, axis=2, keepdims=True)
    xc = x - mu
    rstd = 1.0 / mx.sqrt(mx.mean(xc * xc, axis=2, keepdims=True) + eps)
    out = (xc * rstd).reshape(a.shape)
    if w is not None:
        out = out * _mx(w).reshape(1, C, *(1,) * (a.ndim - 2))
    if b is not None:
        out = out + _mx(b).reshape(1, C, *(1,) * (a.ndim - 2))
    return out, mu.reshape(N, group), rstd.reshape(N, group)


def _xlogy(a, b):
    # aten.xlogy(self, other): NaN in either operand -> NaN first (the aten
    # kernel checks isnan before the zero rule), then self == 0 -> 0, which
    # also masks the NaN that self * log(other) produces for other <= 0.
    # The wheel has no fused xlogy; build the mask with where.
    a, b = _mx(a), _mx(b)
    r = a * mx.log(b)
    nan = mx.array(float("nan"), dtype=r.dtype)
    return _ar(mx.where(mx.equal(a, 0), mx.where(mx.isnan(b), nan, mx.zeros_like(r)), r))


def _xlog1py(a, b):
    # aten.xlog1py(self, other): self * log1p(other) with the same NaN-first,
    # then self==0 mask (KL-divergence / custom-loss graphs use both forms).
    a, b = _mx(a), _mx(b)
    r = a * mx.log1p(b)
    nan = mx.array(float("nan"), dtype=r.dtype)
    return _ar(mx.where(mx.equal(a, 0), mx.where(mx.isnan(b), nan, mx.zeros_like(r)), r))


def _opmath_f32(a):
    """aten elementwise ops on fp16/bf16 inputs compute in fp32 opmath and
    round once at the end; the lowering does the same or the narrow-dtype
    tail values drift. Returns (original, fp32-view)."""
    a = _mx(a)
    return a, (a.astype(mx.float32) if a.dtype in (mx.bfloat16, mx.float16) else a)


def _softplus(a, beta=1, threshold=20):
    # aten.softplus(input, beta=1, threshold=20):
    #   log1p(exp(beta*x)) / beta, and the linear x when beta*x > threshold
    #   (strictly greater — the equality point keeps the log branch). The
    #   linear tail also keeps exp(beta*x) finite for large inputs.
    a, f = _opmath_f32(a)
    bx = f * beta
    return mx.where(bx > threshold, f,
                    mx.log1p(mx.exp(bx)) / beta).astype(a.dtype)


def _elu(a, alpha=1.0, scale=1.0, input_scale=1.0):
    # aten.elu(input, alpha=1, scale=1, input_scale=1):
    #   scale*x when x > 0, else alpha*scale*(exp(input_scale*x) - 1)
    a, f = _opmath_f32(a)
    out = mx.where(mx.greater(f, 0.0), scale * f,
                   scale * alpha * (mx.exp(input_scale * f) - 1.0))
    return out.astype(a.dtype)


def _selu(a):
    # aten.selu is elu with the published constants folded in:
    #   scale=1.0507009873554805, alpha=1.6732632423543772, input_scale=1
    return _elu(a, alpha=1.6732632423543772, scale=1.0507009873554805)


def _celu(a, alpha=1.0):
    # aten.celu(input, alpha=1): max(0, x) + min(0, alpha*(exp(x/alpha) - 1))
    a, f = _opmath_f32(a)
    out = mx.maximum(f, 0.0) + mx.minimum(alpha * (mx.exp(f / alpha) - 1.0), 0.0)
    return out.astype(a.dtype)


def _mish(a, inplace=False):
    # aten.mish(input): x * tanh(softplus(x)) under softplus' defaults
    # (beta=1, threshold=20).
    a, f = _opmath_f32(a)
    sp = mx.where(f > 20, f, mx.log1p(mx.exp(f)))
    return (f * mx.tanh(sp)).astype(a.dtype)


def _glu(a, dim=-1):
    # aten.glu(input, dim=-1): halves along dim, a * sigmoid(b). A scalar
    # or an odd-sized dim is an error in aten too.
    a, f = _opmath_f32(a)
    if a.ndim == 0 or a.shape[dim % a.ndim] % 2:
        return _raise("glu")
    lo, hi = mx.split(f, 2, axis=dim % a.ndim)
    return (lo * mx.sigmoid(hi)).astype(a.dtype)
def _frac(a):
    # aten.frac(input): x - trunc(x); the sign keeps with x (-2.3 -> -0.3,
    # NOT x - floor(x) which would give +0.7).
    a = _mx(a)
    return a - mx.trunc(a)


def _l2_clamped(a, dim, eps):
    return mx.maximum(mx.sqrt(mx.sum(mx.square(_mx(a)), axis=dim)), eps)


def _linalg_norm(a, ord=None, dim=None, keepdim=False, dtype=None):
    # aten.linalg_norm: mx.linalg.norm shares the vector/matrix
    # ord/axis/keepdims contract (ord=None + axis=None flattens, a
    # 2-axis dim is a matrix norm); dtype is an output cast.
    a = _mx(a)
    out = mx.linalg.norm(a, ord=ord, axis=dim, keepdims=keepdim)
    if dtype is not None:
        if dtype not in _MX_DT:
            return _raise("linalg_norm dtype")
        out = out.astype(_MX_DT[dtype])
    return out


def _linalg_vector_norm(a, ord=2, dim=None, keepdim=False, dtype=None):
    # aten.linalg_vector_norm always means a VECTOR norm — flattening
    # when dim is None and reducing every listed dim. mx.linalg.norm
    # reads ord+axis=None or a 2-axis dim as a MATRIX norm, so compose
    # from power/sum instead.
    a = _mx(a)
    ax = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    if ax is None:
        ax = tuple(range(a.ndim))
    o = float(ord)
    if o == float("inf"):
        out = mx.max(mx.abs(a), axis=ax, keepdims=keepdim)
    elif o == float("-inf"):
        out = mx.min(mx.abs(a), axis=ax, keepdims=keepdim)
    elif o == 0:
        out = mx.sum((a != 0).astype(mx.int32), axis=ax,
                     keepdims=keepdim).astype(a.dtype)
    else:
        out = mx.power(
            mx.sum(mx.power(mx.abs(a), o), axis=ax, keepdims=keepdim),
            1.0 / o)
    if dtype is not None:
        if dtype not in _MX_DT:
            return _raise("linalg_vector_norm dtype")
        out = out.astype(_MX_DT[dtype])
    return out


def _pnorm(a, p=2, dim=None, keepdim=False):
    # aten.norm: sum(|x|^p)^(1/p); inf/-inf take the max/min magnitude,
    # not the 2-norm.
    a = _mx(a)
    ax = tuple(dim) if isinstance(dim, (list, tuple)) else dim
    if str(p) in ("inf", "-inf"):
        m = mx.max if str(p) == "inf" else mx.min
        return m(mx.abs(a), axis=ax, keepdims=keepdim)
    return mx.power(mx.sum(mx.power(mx.abs(a), float(p)), axis=ax,
                                  keepdims=keepdim), 1.0 / float(p))


def _cdist(a, b, p=2.0, compute_mode=None):
    # aten.cdist(x1, x2, p, compute_mode): pairwise p-norm distances
    # between the rows of [.., N, D] and [.., M, D]. Broadcast-
    # difference covers every p; compute_mode only picks a numerics
    # strategy in aten, the values are unchanged.
    a, b = _mx(a), _mx(b)
    d = a[..., :, None, :] - b[..., None, :, :]
    fp = float(p)
    if fp == float("inf"):
        return mx.max(mx.abs(d), axis=-1)
    if fp == 0:
        return mx.sum((d != 0).astype(mx.int32), axis=-1).astype(a.dtype)
    return mx.power(mx.sum(mx.power(mx.abs(d), fp), axis=-1), 1.0 / fp)


def _pdist(a, p=2.0):
    # aten.pdist(self, p): condensed i<j pairwise distances of the N
    # row vectors. mx has no compress/nonzero, so the condense order is
    # built arithmetically: flat positions with jj>ii key below n*n and
    # sort first; the keys are unique so the order is exact.
    a = _mx(a)
    n = a.shape[-2]
    d = _cdist(a, a, p)
    flat = mx.arange(n * n)
    sel = flat % n > flat // n
    key = mx.logical_not(sel).astype(mx.int32) * (n * n) + flat
    m = n * (n - 1) // 2
    return mx.flatten(d)[mx.argsort(key)[:m]]


def _repeat_interleave(*args, dim=None, output_size=None):
    # aten.repeat_interleave comes in three overloads: .Tensor (repeats
    # only — returns arange indices repeated), .self_int (scalar repeats)
    # and .self_Tensor (per-element repeats). The overloads share the
    # opname, so arg count picks the form; output_size is kwarg-only in
    # every schema. mx has no repeat_interleave, so the gather indices
    # are built arithmetically: out[j] = a[i] for the i whose inclusive
    # cumsum is first greater than j.
    if len(args) == 1:
        repeats = _mx(args[0]).reshape(-1)
        a, dim = mx.arange(repeats.shape[0]), 0
    else:
        a = _mx(args[0])
        repeats = args[1]
        if len(args) > 2:
            dim = args[2]
        if dim is None:
            a = a.reshape(-1)
            dim = 0
        elif dim < 0:
            dim += a.ndim
        if isinstance(repeats, int):
            repeats = mx.full((a.shape[dim],), repeats, dtype=mx.int32)
        else:
            repeats = _mx(repeats).reshape(-1)
    total = int(output_size) if output_size is not None else int(mx.sum(repeats))
    cums = mx.cumsum(repeats)
    idx = mx.sum(mx.arange(total)[None, :] >= cums[:, None], axis=0)
    return mx.take(a, idx, axis=dim)


def _roll(a, shifts, dims=None):
    # aten.roll spells the flatten form dims=[] (its schema default);
    # torch.roll spells it dims=None. mx.roll flattens on axis=None, so
    # both collapse to that; list/shift pairs go through as tuples.
    if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        dims = None
    if isinstance(shifts, (list, tuple)) and len(shifts) == 1:
        shifts = shifts[0]
    if isinstance(dims, (list, tuple)) and len(dims) == 1:
        dims = dims[0]
    return mx.roll(_mx(a), shifts, axis=dims)
def _flip(a, dims):
    """aten.flip(self, int[] dims): reverse along each listed dim. A
    negative-step slice is the same view torch returns (PT-Q11)."""
    m = _mx(a)
    for d in dims:
        m = _slice(m, int(d) % m.ndim, None, None, -1)
    return m


def _rot90(a, k=1, dims=(0, 1)):
    """aten.rot90(self, int k=1, int[] dims=[0,1]): k quarter-turns in the
    dims plane — the ATen decompositions flip(dims[1])+transpose / flip
    both / flip(dims[0])+transpose (PT-Q11)."""
    m = _mx(a)
    d0, d1 = int(dims[0]) % m.ndim, int(dims[1]) % m.ndim
    kk = int(k) % 4
    if kk == 0:
        return m
    if kk == 1:
        return mx.swapaxes(_flip(m, (d1,)), d0, d1)
    if kk == 2:
        return _flip(m, (d0, d1))
    return mx.swapaxes(_flip(m, (d0,)), d0, d1)
def _isin(elements, test_elements, assume_unique=False, invert=False):
    """aten.isin(elements, test_elements, *, assume_unique, invert): membership
    by broadcast compare — elements[..., None] vs the flattened test set,
    any() over the last axis. assume_unique is a perf hint only; membership
    is the same set either way. bool out, same shape as elements."""
    hit = mx.any(mx.equal(mx.expand_dims(_mx(elements), -1),
                          mx.flatten(_mx(test_elements))), axis=-1)
    return mx.logical_not(hit) if invert else hit


_FUNC_SPECS = {
    "linear": _linear,
    "matmul": lambda a, b: mx.matmul(_mx(a), _mx(b)).astype(_mx(a).dtype),
    "bmm": lambda a, b: mx.matmul(_mx(a), _mx(b)).astype(_mx(a).dtype),
    "mm": lambda a, b: mx.matmul(_mx(a), _mx(b)).astype(_mx(a).dtype),
    "addmm": lambda b, m1, m2, beta=1, alpha=1: (_blas_beta(b, beta) + alpha * mx.matmul(m1, m2)).astype(_mx(b).dtype),
    "baddbmm": lambda b, m1, m2, beta=1, alpha=1: (_blas_beta(b, beta) + alpha * mx.matmul(m1, m2)).astype(_mx(b).dtype),
    "mv": lambda a, v: mx.matmul(a, v),
    "dot": lambda a, b: mx.sum(_mx(a) * _mx(b)),
    "addmv": lambda b, m, v, beta=1, alpha=1: _blas_beta(b, beta) + alpha * mx.matmul(m, v),
    "addbmm": lambda b, b1, b2, beta=1, alpha=1: _blas_beta(b, beta) + alpha * mx.sum(mx.matmul(b1, b2), axis=0),
    "addr": lambda b, v1, v2, beta=1, alpha=1: _blas_beta(b, beta) + alpha * mx.outer(_mx(v1), _mx(v2)),
    "vdot": lambda a, b: mx.sum(mx.conjugate(_mx(a)) * _mx(b)),
    "reshape": _reshape, "view": _reshape, "_unsafe_view": _reshape,
    "unsafe_view": _reshape,
    "add": lambda a, b, alpha=1, *k: _alpha_arith(a, b, alpha, +1),
    "sub": lambda a, b, alpha=1, *k: _alpha_arith(a, b, alpha, -1),
    "mul": lambda a, b, *k: _ar(_mx(a) * _mx(b)),
    "mul_": lambda a, b, *k: _ar(_mx(a) * _mx(b)),
    "div": lambda a, b, *k: _ar(_mx(a) / _mx(b)),
    "truediv": lambda a, b, *k: _ar(_mx(a) / _mx(b)),
    "true_divide": lambda a, b, *k: _ar(_mx(a) / _mx(b)),
    # aten.remainder: numpy-style modulo (result takes the divisor's sign,
    # matching torch.remainder); mx.remainder has the same contract and the
    # omarchy backend implements it for float and int (Remainder eval_gpu).
    # Go through %, not mx.remainder: scalar-only calls (remainder(7, 3))
    # evaluate in Python first so _ar boxes int64 like the binop specs, and
    # a bare scalar stays weak beside a narrow-int tensor operand.
    "remainder": lambda a, b, *k: _ar(_mx(a) % _mx(b)),
    # aten.addcmul/addcdiv(input, t1, t2, *, value=1): input + value*t1*t2
    "addcmul": _addcmul,
    "addcdiv": _addcdiv,
    "neg": lambda a: -a,
    "pow": lambda a, b: mx.power(_mx(a), _mx(b)),
    "sqrt": mx.sqrt, "rsqrt": mx.rsqrt, "exp": mx.exp, "log": mx.log,
    # PT-AL-e: basic math ops used in normalization chains
    "reciprocal": mx.reciprocal, "frac": _frac,
    "reciprocal_": mx.reciprocal, "frac_": _frac,
    "sigmoid": mx.sigmoid, "tanh": mx.tanh, "erf": mx.erf,
    "transpose": lambda a, d0, d1: mx.transpose(a, axes=_perm_swap(a, d0, d1)),
    "permute": lambda a, *d: mx.transpose(a, axes=tuple(d[0]) if len(d) == 1 else tuple(d)),
    "flatten": lambda a, start=0, end=-1: mx.reshape(a, (*a.shape[:start], int(np.prod(a.shape[start:end + 1])), *a.shape[end + 1:]) if end != -1 else (*a.shape[:start], int(np.prod(a.shape[start:])))),
    "unsqueeze": lambda a, d: mx.expand_dims(a, axis=d),
    "squeeze": _squeeze, "squeeze_": _squeeze,
    "expand": _expand, "expand_as": lambda a, o: mx.broadcast_to(a, o.shape),
    "cat": _cat_impl,
    "concat": _cat_impl,
    "stack": lambda t, dim=0: mx.stack([_mx(x) for x in t], axis=dim),
    "split": lambda a, ss, dim=0: _split_even(a, ss, dim) if isinstance(ss, int) else _split_sizes(a, ss, dim),
    "split_with_sizes": lambda a, ss, dim=0: _split_sizes(a, ss, dim),
    "chunk": _chunk, "unbind": lambda a, dim=0: _unbind(a, dim),
    "meshgrid": _meshgrid,
    # torch's norm=None means "backward", same as mx's default
    "fft_fft": lambda a, n=None, dim=-1, norm=None: mx.fft.fft(_mx(a), n, axis=dim, norm=norm or "backward"),
    "fft_ifft": lambda a, n=None, dim=-1, norm=None: mx.fft.ifft(_mx(a), n, axis=dim, norm=norm or "backward"),
    # PT-AU-b: real<->complex transforms (Parakeet mel-spectrogram)
    "fft_rfft": lambda a, n=None, dim=-1, norm=None: mx.fft.rfft(_mx(a), n, axis=dim, norm=norm or "backward"),
    "fft_irfft": lambda a, n=None, dim=-1, norm=None: mx.fft.irfft(_mx(a), n, axis=dim, norm=norm or "backward"),
    # PT-AU-c: 2-D transforms (vision, spectrogram models)
    "fft_fft2": lambda a, s=None, dim=(-2, -1), norm=None: mx.fft.fft2(_mx(a), s, axes=dim, norm=norm or "backward"),
    "fft_ifft2": lambda a, s=None, dim=(-2, -1), norm=None: mx.fft.ifft2(_mx(a), s, axes=dim, norm=norm or "backward"),
    "select": lambda a, dim, i: _slice(a, dim, i % a.shape[dim % a.ndim], i % a.shape[dim % a.ndim] + 1, 1)[tuple(0 if d == dim % a.ndim else slice(None) for d in range(a.ndim))],
    "slice": _slice, "narrow": lambda a, dim, s, n: _slice(a, dim, s, s + n, 1),
    "mean": _mean, "sum": _sum,
    # PT-AL-g: aten.logsumexp(self, int[1] dim, keepdim) — the dim list
    # becomes the axis tuple; mx.logsumexp is the same stable reduction.
    "logsumexp": lambda a, dim, keepdim=False: mx.logsumexp(_mx(a), axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "max": lambda a, d=None, keepdim=False, *k: (mx.max(a, axis=d, keepdims=keepdim), mx.argmax(a, axis=d, keepdims=keepdim)) if d is not None else mx.max(a),
    "min": lambda a, d=None, keepdim=False, *k: (mx.min(a, axis=d, keepdims=keepdim), mx.argmin(a, axis=d, keepdims=keepdim)) if d is not None else mx.min(a),
    "where": lambda c, a, b: mx.where(_mx(c), _mx(a), _mx(b)),
    "zeros": lambda *a, **k: mx.zeros(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "zeros_like": lambda a, **k: mx.zeros(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "ones": lambda *a, **k: mx.ones(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "ones_like": lambda a, **k: mx.ones(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "full": lambda s, v, **k: mx.full(tuple(s), v, dtype=_MX_DT.get(k.get("dtype"))),
    "full_like": lambda a, v, **k: mx.full(a.shape, v, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "tril": lambda a, d=0: mx.tril(a, d) if hasattr(mx, "tril") else _raise("tril"),
    "getitem": lambda t, i: _raise("aten.getitem: a bool index tensor is a mask in aten, not an int index") if _has_bool_index(i) else t[_coerce_index(i)],
    "detach": lambda a, *a_, **k_: a, "clone": lambda a, *a_, **k_: a.clone() if isinstance(a, torch.Tensor) else a, "contiguous": lambda a, *a_, **k_: a,
    "alias": lambda a: a, "lift_fresh": lambda a: a,
    "lift_fresh_copy": lambda a, *a_, **k_: _mx(a),
    "to": lambda a, *a_, **k_: a,
    "_to_copy": lambda a, **k_: a, "size": lambda a, d=None: a.shape[d] if d is not None else a.shape,
    "numel": lambda a: int(a.size),
    "native_layer_norm": _ln_native,
    "dropout": lambda a, p=0.5, training=False, *k: a,
    # aten.batch_norm order: (x, weight, bias, running_mean, running_var, training, momentum, eps, cudnn)
    "batch_norm": lambda x, w, b, rm, rv, training, momentum, eps, *k: _bn3(x, w, b, rm, rv, training, momentum, eps)[0],
    "native_batch_norm": lambda x, w, b, rm, rv, training, momentum, eps: _bn3(x, w, b, rm, rv, training, momentum, eps),
    "_native_batch_norm_legit_functional": _bn_legit_functional,
    # the aot/functional EVAL graph lowers to this op; without an entry
    # every eval batch-norm through torch.compile is unlowerable (PAR-41)
    # aten order: (x, weight, bias, running_mean, running_var, momentum, eps)
    # - no training flag. Torch returns (out, empty, empty); MLX returns the
    # running stats in slots 2/3 like native_batch_norm.
    # aten order: (x, weight, bias, running_mean, running_var, momentum, eps)
    # - no training flag. Torch returns (out, empty, empty): the save stats
    # are typed length-0 tensors, so the lowering returns typed empties too
    # and keeps _bn3's eval math for the output only.
    "_native_batch_norm_legit_no_training": lambda x, w, b, rm, rv, momentum, eps, *k: (
        lambda r: (r[0].astype(_mx(x).dtype), _mx(rm).reshape(-1)[:0], _mx(rv).reshape(-1)[:0]))(_bn3(
            x, w, b, rm, rv, False, momentum, eps)),
    "native_batch_norm_backward": _bn_backward,
    "native_group_norm": _gn_native,
    "conv2d": _conv2d,
    "max_pool2d": lambda a, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False: _max_pool2d(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices),
    "max_pool2d_with_indices": lambda a, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False: _max_pool2d(a, kernel_size, stride, padding, dilation, ceil_mode, return_indices=True),
    "max_pool2d_with_indices_backward": _max_pool2d_bwd,
    "avg_pool2d": _avg_pool2d,
    "avg_pool2d_backward": _avg_pool2d_backward,
    "adaptive_avg_pool2d": _adaptive_avg_pool2d,
    "_adaptive_avg_pool2d": _adaptive_avg_pool2d,
    "adaptive_avg_pool2d_backward": _adaptive_avg_pool2d_backward,
    "_adaptive_avg_pool2d_backward": _adaptive_avg_pool2d_backward,
    "conv1d": _conv1d,
    "diff": _diff,
    "convolution": _convolution,
    "convolution_backward": _convolution_backward,
    "t": lambda a: a if a.ndim < 2 else mx.transpose(a, axes=(-1, -2)),
    "threshold_backward": lambda g, x, t=0.0: mx.where(mx.greater(_mx(x), t), _mx(g), mx.zeros_like(_mx(g))),
    "select_backward": _select_backward,
    "tanh_backward": _tanh_backward,
    "nll_loss_forward": _nll_loss_forward,
    "nll_loss_backward": _nll_loss_backward,
    "gelu_backward": _gelu_backward,
    "group_norm": _group_norm,
    "interpolate": _interpolate,
    "softmax": lambda a, dim=-1, dtype=None, **k: _softmax_fwd(a, dim, dtype=dtype),
    "_softmax": _softmax_fwd,
    "gelu": lambda a, approximate="none": (0.5 * a * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a * a * a))) if approximate == "tanh" else 0.5 * a * (1.0 + mx.erf(a / math.sqrt(2.0)))),
    "cross_entropy": lambda logits, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", label_smoothing=0.0, *a_, **k_: (
        _ce(logits, target, weight, ignore_index, label_smoothing, _red(reduction, size_average, reduce))),
    "nll_loss": lambda logp, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean", *a_, **k_: (
        _nll(logp, target, weight, ignore_index, _red(reduction, size_average, reduce))),
    "mse_loss": lambda a, b, size_average=None, reduce=None, reduction="mean": _mxnn.losses.mse_loss(a, b, reduction=_red(reduction, size_average, reduce)),
    "native_layer_norm_backward": _ln_backward,
    "_scaled_dot_product_flash_attention_for_cpu": _sdpa_flash_cpu,
    "_scaled_dot_product_flash_attention_for_cpu_backward": _sdpa_flash_cpu_bwd,
    "embedding_dense_backward": _embedding_dense_backward,
    "slice_backward": _slice_backward,
    "silu_backward": _silu_backward,
    "silu": lambda a, inplace=False: a * mx.sigmoid(a),
    "relu": lambda a, inplace=False: mx.maximum(a, 0.0),
    "softplus": _softplus,
    "mish": _mish,
    "relu_": lambda a: mx.maximum(a, 0.0),
    # PT-AL-a: the elu family (ViT, Conformer, speech models).
    "elu": _elu,
    "elu_": _elu,
    "selu": _selu,
    "selu_": _selu,
    "celu": _celu,
    "celu_": _celu,
    # PT-AL-d: aten.glu (GDN gate, Conformer)
    "glu": _glu,
    # PT-AL-b: the piecewise activation family (MobileNetV3 et al.)
    "hardsigmoid": lambda a, inplace=False: mx.minimum(mx.maximum(a + 3.0, 0.0), 6.0) / 6.0,
    "hardswish": lambda a, inplace=False: a * (mx.minimum(mx.maximum(a + 3.0, 0.0), 6.0) / 6.0),
    "hardsigmoid_": lambda a: mx.minimum(mx.maximum(a + 3.0, 0.0), 6.0) / 6.0,
    "hardswish_": lambda a: a * (mx.minimum(mx.maximum(a + 3.0, 0.0), 6.0) / 6.0),
    "hardshrink": lambda a, lambd=0.5, *k: mx.where(mx.greater(mx.abs(a), lambd), a, mx.zeros_like(a)),
    "softshrink": lambda a, lambd=0.5, *k: mx.where(mx.greater(a, lambd), a - lambd, mx.where(mx.less(a, -lambd), a + lambd, mx.zeros_like(a))),
    "tanhshrink": lambda a: a - mx.tanh(a),
    "softsign": lambda a: a / (1.0 + mx.abs(a)),
    "elu": lambda a, alpha=1.0, scale=1.0, input_scale=1.0, inplace=False: scale * mx.where(mx.greater(a, 0), a, alpha * mx.expm1(input_scale * a)),
    "selu": lambda a, inplace=False: 1.0507009873554805 * mx.where(mx.greater(a, 0), a, 1.6732632423543772 * mx.expm1(a)),
    "celu": lambda a, alpha=1.0, inplace=False: mx.where(mx.greater(a, 0), a, alpha * mx.expm1(a / alpha)),
    "scaled_dot_product_attention": _sdpa,
    "_scaled_dot_product_efficient_attention": _sdpa_eff,
    "_efficient_attention_forward": _sdpa_eff_fwd,
    "eq": lambda a, b: mx.equal(_mx(a), _mx(b)),
    "ne": lambda a, b: mx.not_equal(_mx(a), _mx(b)),
    "not_equal": lambda a, b: mx.not_equal(_mx(a), _mx(b)),
    "gt": lambda a, b: mx.greater(_mx(a), _mx(b)),
    "lt": lambda a, b: mx.less(_mx(a), _mx(b)),
    "cos": mx.cos, "sin": mx.sin, "log": mx.log, "log1p": mx.log1p,
    # PT-AL-f: the wheel table already carries these; the dev table
    # must match so traced graphs lower them too.
    "expm1": mx.expm1, "expm1_": mx.expm1,
    "logaddexp": lambda a, b: mx.logaddexp(_mx(a), _mx(b)),
    "xlogy": _xlogy, "xlog1py": _xlog1py,
    # aten registers torch.special.xlog1py as special_xlog1py
    "special_xlog1py": _xlog1py,
    "sinh": _sinh, "cosh": _cosh,
    "asinh": _asinh, "acosh": _acosh, "atanh": mx.atanh,
    "acos": mx.acos, "asin": mx.asin, "atan": mx.atan, "atan2": _atan2,
    "outer": mx.outer, "einsum": lambda *a: mx.einsum(*a),
    "argmax": lambda a, dim=None, keepdim=False: mx.argmax(a, axis=dim, keepdims=keepdim),
    "argmin": lambda a, dim=None, keepdim=False: mx.argmin(a, axis=dim, keepdims=keepdim),
    "topk": _topk,
    "multinomial": _multinomial,
    "histc": _histc,
    "histogram": _histogram,
    "mode": _mode,
    "kthvalue": _kthvalue,
    "take": lambda a, i, dim=None: mx.take(a, _int_index(i, "aten.take"), axis=dim),
    "index_select": lambda a, dim, i: mx.take(a, _int_index(i, "aten.index_select"), axis=dim),
    "gather": lambda a, dim, i: mx.take_along_axis(a, _int_index(i, "aten.gather"), axis=dim),
    "take_along_dim": lambda a, i, dim: mx.take_along_axis(a, _int_index(i, "aten.take_along_dim"), axis=dim),
    "index": _aten_index,
    # PT-AL-i: membership (aten.isin; in1d is its legacy alias in old graphs)
    "isin": _isin,
    "in1d": _isin,
    "bincount": _bincount,
    "tril": lambda a, k=0: mx.tril(a, k), "triu": lambda a, k=0: mx.triu(a, k),
    "pad": lambda a, p, mode="constant", value=0: mx.pad(a, _pad_width(a, p)),
    "tile": lambda a, d: mx.tile(a, tuple(d)),
    "repeat": lambda a, *d: mx.tile(a, tuple(d[0]) if len(d) == 1 and isinstance(d[0], (list, tuple)) else tuple(d)),
    # PT-AL-i: aten.repeat_interleave (self_int / self_Tensor / .Tensor
    # overloads) and aten.roll's dims=[] schema default.
    "repeat_interleave": _repeat_interleave,
    "movedim": lambda a, src, dst: mx.moveaxis(a, src, dst),
    "moveaxis": lambda a, src, dst: mx.moveaxis(a, src, dst),
    "swapaxes": lambda a, d0, d1: mx.swapaxes(a, d0, d1),
    "flip": _flip, "rot90": _rot90,
    "abs": mx.abs, "sign": mx.sign,
    "maximum": mx.maximum, "minimum": mx.minimum, "fmax": mx.maximum, "fmin": mx.minimum,
    "floor": mx.floor, "ceil": mx.ceil, "round": mx.round, "floor_divide": mx.floor_divide,
    "negative": mx.negative, "isnan": mx.isnan, "isinf": mx.isinf,
    "isfinite": lambda a: mx.logical_not(mx.isinf(a)),
    "linspace": lambda s, e, n, **k: mx.linspace(s, e, n),
    "arange": lambda *a, **k: _arange(*a, **k),
    "eye": lambda n, m=None, **k: mx.eye(n, m) if m else mx.eye(n),
    "ge": lambda a, b: mx.greater_equal(_mx(a), _mx(b)),
    "le": lambda a, b: mx.less_equal(_mx(a), _mx(b)),
    "greater": lambda a, b: mx.greater(_mx(a), _mx(b)),
    "less": lambda a, b: mx.less(_mx(a), _mx(b)),
    "logical_and": lambda a, b: mx.logical_and(_mx(a), _mx(b)),
    "bitwise_and": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_and, mx.logical_and),
    "bitwise_or": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_or, mx.logical_or),
    "bitwise_xor": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_xor, mx.bitwise_xor),
    "and_": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_and, mx.logical_and),
    "bitwise_and_": lambda a, b: _inplace_result(_bitwise_or_logical(a, b, mx.bitwise_and, mx.logical_and), a),
    "or_": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_or, mx.logical_or),
    "xor": lambda a, b: _bitwise_or_logical(a, b, mx.bitwise_xor, mx.bitwise_xor),
    "logical_or": lambda a, b: mx.logical_or(_mx(a), _mx(b)),
    "logical_not": lambda a: mx.logical_not(a),
    "invert": lambda a: mx.logical_not(a),
    "clamp": lambda a, min=None, max=None: mx.clip(a, _mx(min) if min is not None else -float("inf"), _mx(max) if max is not None else float("inf")),
    "clamp_": lambda a, min=None, max=None: mx.clip(a, _mx(min) if min is not None else -float("inf"), _mx(max) if max is not None else float("inf")),
    "clip": lambda a, min=None, max=None: mx.clip(a, _mx(min) if min is not None else -float("inf"), _mx(max) if max is not None else float("inf")),
    "clamp_min": lambda a, min: mx.maximum(a, _mx(min)),
    "clamp_min_": lambda a, min: mx.maximum(a, _mx(min)),
    "clamp_max": lambda a, max: mx.minimum(a, _mx(max)),
    "clamp_max_": lambda a, max: mx.minimum(a, _mx(max)),
    "rsub": _rsub,
    "sub_": _sub_inplace,
    "sub_.Tensor": _sub_inplace,
    "masked_fill": lambda a, m, v: mx.where(_bool_mask(m, "aten.masked_fill"), v, a),
    "masked_fill_": lambda a, m, v: mx.where(_bool_mask(m, "aten.masked_fill_"), v, a),
    "cumsum": lambda a, dim, dtype=None: mx.cumsum(a, axis=dim),
    "std": _std, "var": _var, "var_mean": _var_mean, "std_mean": _std_mean,
    "roll": _roll,
    "log_softmax": lambda a, dim=-1, dtype=None, **k: _log_softmax_fwd(a, dim, dtype=dtype),
    "_log_softmax": _log_softmax_fwd,
    "_log_softmax_backward_data": _log_softmax_bwd,
    "normalize": lambda x, p=2, dim=1, eps=1e-12: x / mx.maximum(mx.sum(mx.power(mx.abs(x), p), axis=dim, keepdims=True), eps),
    "norm": _pnorm,
    # PT-NORM-b: the linalg norm family (normalization, grad clipping)
    "linalg_vector_norm": lambda a, ord=2, dim=None, keepdim=False, *, dtype=None: _linalg_vector_norm(a, ord, dim, keepdim, dtype),
    "linalg_norm": lambda a, ord=None, dim=None, keepdim=False, *, dtype=None: _linalg_norm(a, ord, dim, keepdim, dtype),
    # PT-PAIR-a: pairwise distances used in metric learning.
    "cdist": lambda a, b, p=2.0, compute_mode=None: _cdist(a, b, p, compute_mode),
    "pdist": lambda a, p=2.0: _pdist(a, p),
    "rms_norm": lambda a, ns, w=None, eps=None: mx.fast.rms_norm(a, _mx(w) if w is not None else None, eps if eps is not None else 1e-6),
    # aten.cosine_similarity clamps each l2 norm at eps separately, not
    # their product (device-checked against torch 2.15 CPU).
    "cosine_similarity": lambda x1, x2, dim=1, eps=1e-8: mx.sum(_mx(x1) * _mx(x2), axis=dim) / (_l2_clamped(x1, dim, eps) * _l2_clamped(x2, dim, eps)),
    "embedding": _embedding_any,
    "all": lambda a, dim=None, keepdim=False: mx.all(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "any": lambda a, dim=None, keepdim=False: mx.any(a, axis=tuple(dim) if isinstance(dim, (list, tuple)) else dim, keepdims=keepdim),
    "count_nonzero": lambda a, dim=None: mx.sum((a != 0).astype(mx.int32), axis=dim),
    "empty": lambda *a, **k: mx.zeros(_shape_of(a, k), dtype=_MX_DT.get(k.get("dtype"))),
    "empty_like": lambda a, **k: mx.zeros(a.shape, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_zeros": lambda a, s, **k: mx.zeros(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_ones": lambda a, s, **k: mx.ones(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_empty": lambda a, s, **k: mx.zeros(tuple(s), dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "new_full": lambda a, s, v, **k: mx.full(tuple(s), v, dtype=_MX_DT.get(k.get("dtype"), _mx(a).dtype)),
    "scalar_tensor": lambda v, **k: mx.array(v, _MX_DT.get(k.get("dtype"), mx.int64 if type(v) is int else None)),
    "tensor": lambda d, **k: _mx(d) if isinstance(d, torch.Tensor) else mx.array(np.array(_torch(d)) if isinstance(d, (list, tuple)) else d),
    "as_tensor": lambda d, **k: _mx(d),
    "einsum_fn": mx.einsum,
    "_local_scalar_dense": _local_scalar_dense,
    "Float": _float_scalar,
}

def _raise(name):
    raise NotImplementedError(name)

FUNC_TABLE = {}
_FN_SOURCES = [torch, F, torch.special, operator]

def _resolve(dotted):
    parts = dotted.split(".")
    obj = globals()[parts[0]]
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _bd_inner(tgt, args, kwargs):
    """Resolve torch.nn.functional.boolean_dispatch.<locals>.fn wrappers
    (F.max_pool2d's return_indices arm and friends) to the inner torch
    function the call actually selects, so table lookups can hit; returns
    the wrapper unchanged when the closure cannot be read."""
    if not getattr(tgt, "__qualname__", "").startswith("boolean_dispatch"):
        return tgt
    try:
        nv = inspect.getclosurevars(tgt).nonlocals
        aname, aidx = nv.get("arg_name"), nv.get("arg_index")
        use_true = False
        if aname and aname in kwargs:
            use_true = bool(kwargs[aname])
        elif isinstance(aidx, int) and 0 <= aidx < len(args):
            use_true = bool(args[aidx])
        inner = nv.get("if_true" if use_true else "if_false")
        return inner if inner is not None else tgt
    except Exception:
        return tgt

def _build_func_table():
    candidates = []
    for fn_name, lowering in _FUNC_SPECS.items():
        if fn_name in ("embedding", "batch_norm"):
            continue  # specs are aten order; python-order names map explicitly below
        for src in (torch, F, torch.special):
            fn = getattr(src, fn_name, None)
            if fn is not None:
                candidates.append((fn, lowering))
    # explicit functional mappings
    explicit = [
        ("torch.nn.functional.linear", _linear),
        ("torch.nn.functional.layer_norm", lambda a, ns, w=None, b=None, eps=1e-5: mx.fast.layer_norm(a, _mx(w) if w is not None else None, _mx(b) if b is not None else None, eps)),
        ("torch.nn.functional.gelu", _FUNC_SPECS["gelu"]),
        ("torch.nn.functional.silu", _FUNC_SPECS["silu"]),
        ("torch.nn.functional.relu", _FUNC_SPECS["relu"]),
        ("torch.nn.functional.softmax", _FUNC_SPECS["softmax"]),
        ("torch.nn.functional.scaled_dot_product_attention", _sdpa),
        ("torch.nn.functional.dropout", _FUNC_SPECS["dropout"]),
        ("torch.nn.functional.batch_norm", lambda x, rm, rv, w=None, b=None, training=False, momentum=0.1, eps=1e-5: _bn3(x, w, b, rm, rv, training, momentum, eps)[0]),
        ("torch.batch_norm", _FUNC_SPECS["batch_norm"]),
        ("torch.nn.functional.embedding", _embedding_any),
        ("torch.embedding", _embedding),
        ("operator.getitem", lambda t, i: t[i]),
        ("operator.matmul", mx.matmul),
        ("operator.add", _FUNC_SPECS["add"]),
        ("operator.sub", _FUNC_SPECS["sub"]),
        ("operator.mul", _FUNC_SPECS["mul"]),
        ("operator.truediv", _FUNC_SPECS["div"]),
        ("operator.floordiv", _FUNC_SPECS["div"]),
        ("operator.neg", _FUNC_SPECS["neg"]),
        ("operator.and_", _FUNC_SPECS["and_"]),
        ("operator.or_", _FUNC_SPECS["or_"]),
        ("operator.xor", _FUNC_SPECS["xor"]),
        ("operator.invert", _FUNC_SPECS["invert"]),
    ]
    for dotted, lowering in explicit:
        try:
            candidates.append((_resolve(dotted), lowering))
        except AttributeError:
            pass
    for fn, lowering in candidates:
        FUNC_TABLE[fn] = lowering

_build_func_table()
# dotted-name resolve only covers torch attrs; operator.* objects register directly
FUNC_TABLE[operator.iadd] = _FUNC_SPECS["add"]
FUNC_TABLE[operator.isub] = _FUNC_SPECS["sub"]
FUNC_TABLE[torch.ops.aten.sub_.Tensor] = _FUNC_SPECS["sub_.Tensor"]
FUNC_TABLE[torch.rsub] = _FUNC_SPECS["rsub"]

# name-keyed fallback: dynamo may reference e.g. torch._C._VariableFunctions.matmul
# which is a different object than torch.matmul
FUNC_TABLE_BY_NAME = {}
for _fn, _low in FUNC_TABLE.items():
    FUNC_TABLE_BY_NAME.setdefault(getattr(_fn, "__name__", ""), _low)

# ops that produce arrays from scalars/shapes — lower even with no mx inputs
_PRODUCERS = {"arange", "zeros", "ones", "full", "empty", "eye", "linspace",
              "tensor", "zeros_like", "ones_like", "full_like", "empty_like",
              "rand", "randn", "randint", "randperm", "new_zeros", "new_ones",
              "new_empty", "new_full", "scalar_tensor"}

# sampling ops: mx.compile would trace an mx.random draw once and replay
# it, and mx.random cannot honor torch's RNG contract anyway — the family
# deliberately falls back to torch (see the note above _FUNC_SPECS); this
# guard keeps a future re-lowering out of the compiled path
_SAMPLERS = {"uniform", "normal", "cauchy", "exponential", "geometric",
             "log_normal"}

# ops that must never CPU-fallback: lower even when upstream nodes produced
# torch tensors (e.g. pool bwd behind an unlowered conv/bn backward), and
# cheap elementwise binops whose sampled forms can arrive with scalar-only
# args (aten.mul.default(2, 3) style — no mx input to trip has_mx)
_ALWAYS_LOWER = {"max_pool2d", "max_pool2d_with_indices",
                 "max_pool2d_with_indices_backward", "avg_pool2d",
                 "avg_pool2d_backward", "adaptive_avg_pool2d",
                 "_adaptive_avg_pool2d", "adaptive_avg_pool2d_backward",
                 "_adaptive_avg_pool2d_backward", "convolution_backward",
                 "threshold_backward", "native_batch_norm_backward",
                 "Float",
                 "add", "sub", "mul", "div", "truediv", "true_divide",
                 "remainder", "xlogy", "special_xlog1py", "relu", "relu_",
                 # comparison ops: the aten .default overloads are
                 # (Scalar, Scalar) -> bool — no mx operand, same gate as
                 # the scalar arith forms above.
                 "eq", "ne", "lt", "le", "gt", "ge"}

# aten.* OpOverload name table (same lowerings, keyed by opname)
ATEN_TABLE = {name: f for name, f in _FUNC_SPECS.items()}
ATEN_TABLE["aten.rsub.Scalar"] = _FUNC_SPECS["rsub"]
ATEN_TABLE["_native_batch_norm_legit"] = _FUNC_SPECS["native_batch_norm"]
# aten.embedding is (weight, indices): F.embedding is (input, weight).
ATEN_TABLE["embedding"] = _embedding_any

# zero-kernel bookkeeping targets: run on python values, logged GLUE
GLUE_FNS = {len, tuple, list, dict, int, float, str, bool, slice, range, zip, torch.finfo, torch.iinfo,
            enumerate, getattr, sorted, max, min, sum, abs, round,
            torch._C._set_grad_enabled, torch._C._log_api_usage_once,
            operator.index, operator.length_hint, operator.not_, operator.truth,
            math.sqrt, math.ceil, math.floor, math.log, math.log2, math.prod}
try:
    GLUE_FNS.add(torch.sym_int)
except AttributeError:
    pass
# symbolic-int bookkeeping over dynamic shapes (e.g. HF DynamicCache
# seq-length sums): concrete ints at call time - pure glue, no kernel.
for _sf in ("sym_sum",):
    _f = getattr(torch, _sf, None)
    if _f is not None:
        GLUE_FNS.add(_f)

_METHOD_DT = {"float": mx.float32, "double": mx.float32, "half": mx.float16,
              "bfloat16": mx.bfloat16, "int": mx.int32, "long": mx.int64,
              "short": mx.int16, "byte": mx.uint8, "char": mx.int8,
              "bool": getattr(mx, "bool_", mx.uint8)}
_MX_DT = {torch.float32: mx.float32, torch.float64: mx.float32, torch.float16: mx.float16,
          torch.bfloat16: mx.bfloat16, torch.int32: mx.int32, torch.int64: mx.int64,
          torch.int16: mx.int16, torch.uint8: mx.uint8, torch.int8: mx.int8,
          torch.bool: getattr(mx, "bool_", mx.uint8)}

_METHODS_ON_MX = set(_FUNC_SPECS) | {"size", "dim", "ndim", "shape", "dtype",
                                   "cpu", "to", "cuda", "numpy", "numel",
                                   "type", "item", "tolist"}

def _run_method(obj, name, args, kwargs):
    """Returns (value, on_mlx). Raises NotImplementedError for unknown methods on mx arrays."""
    if isinstance(obj, mx.array):
        if name in ("size",):
            return tuple(obj.shape)[args[0]] if args else tuple(obj.shape), True
        if name in ("dim", "ndim"):
            return obj.ndim, True
        if name in ("shape",):
            return tuple(obj.shape), True
        if name == "dtype":
            return obj.dtype, True
        if name == "numel":
            return int(np.prod(obj.shape)), True
        if name in ("new_empty", "new_zeros", "new_ones", "new_full"):
            dt = _MX_DT.get(kwargs.get("dtype"), obj.dtype)
            if name == "new_full" and len(args) < 2:
                args = (kwargs["size"], args[0] if args else kwargs["fill_value"])
            shape = tuple(args[0] if args else kwargs["size"])
            if name == "new_ones":
                return mx.ones(shape, dtype=dt), True
            if name == "new_full":
                return mx.full(shape, args[1], dtype=dt), True
            return mx.zeros(shape, dtype=dt), True
        if name in ("cpu", "detach", "type", "cuda"):
            return obj, True
        if name == "to":
            dt = next((a for a in args if isinstance(a, torch.dtype)), kwargs.get("dtype"))
            mdt = _MX_DT.get(dt, obj.dtype)
            return obj if mdt == obj.dtype else obj.astype(mdt), True
        if name in ("float", "half", "bfloat16", "int", "long", "double", "bool", "short", "byte", "char"):
            return obj.astype(_METHOD_DT[name]), True
        if name == "item":
            return float(np.array(obj).item()), True
        if name == "tolist":
            return np.array(obj).tolist(), True
        if name == "numpy":
            return np.array(obj), True
        key = name if name in _FUNC_SPECS else None
        if key is None:
            raise NotImplementedError(f"mx method {name}")
        return _FUNC_SPECS[key](obj, *args, **kwargs), True
    return getattr(obj, name)(*args, **kwargs), False

# ---------------- executor ----------------

def make_executor(gm):
    stats = {"mlx": [], "cpu": [], "glue": [], "crossings": 0}
    strided = _strided_nodes(gm.graph.nodes)
    refused = _refused_nodes(gm.graph.nodes, strided)
    _nprof = os.environ.get("MV_NODE_PROF")
    if _nprof:
        stats["node_ms"] = {}
        import time as _nt

    def resolve(a, env):
        if isinstance(a, torch.fx.Node):
            return env[a]
        if isinstance(a, (tuple, list, torch.fx.immutable_collections.immutable_list)):
            return type(a)(resolve(x, env) for x in a)
        return a

    def _deep_mx(v):
        if isinstance(v, torch.Tensor):
            return mx.array(v.detach().cpu().numpy())
        if isinstance(v, (tuple, list)):
            return type(v)(_deep_mx(x) for x in v)
        return v

    def has_mx(v):
        if isinstance(v, mx.array):
            return True
        return isinstance(v, (tuple, list)) and any(has_mx(x) for x in v)

    def run(*runtime_inputs):
        env = {}
        it = iter(runtime_inputs)
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                env[node] = _mx(next(it))
                continue
            if _nprof:
                _t0 = _nt.perf_counter()

            def _tick(nm):
                if _nprof:
                    stats["node_ms"][nm] = stats["node_ms"].get(nm, 0.0) + (_nt.perf_counter() - _t0) * 1e3

            args = tuple(resolve(a, env) for a in node.args)
            kwargs = {k: resolve(v, env) for k, v in node.kwargs.items()}
            name = _opname(node)

            if node.op == "output":
                out = args[0]
                return type(out)(_torch(x) for x in out) if isinstance(out, (tuple, list)) else _torch(out)
            if node.op == "get_attr":
                t = gm
                for part in str(node.target).split("."):
                    t = getattr(t, part)
                env[node] = _mx(t)
                _tick("get_attr")
                continue
            if (node.target is operator.getitem or name == "getitem") and node not in refused:
                try:
                    if isinstance(args[0], mx.array) and _has_bool_index(args[1]):
                        raise ValueError("a bool index tensor is a mask in aten; the gather path reads it as 0/1")
                    env[node] = args[0][_coerce_index(args[1])]
                    stats["glue"].append("getitem")
                    _tick("getitem")
                    continue
                except Exception as e:
                    # Mixed-domain getitem (torch source, mlx index): coerce the
                    # indexable side to mx when any operand lives there.
                    if has_mx((args[0], args[1])) and not _has_bool_index(args[1]):
                        try:
                            env[node] = _mx(args[0])[_deep_mx(_coerce_index(args[1]))]
                            stats["glue"].append("getitem")
                            _tick("getitem")
                            continue
                        except Exception:
                            pass
                    log.info("LOWER-FAIL	getitem	%s", e)

            val = None
            on_mlx = False
            if node.op == "call_function":
                tgt = _bd_inner(node.target, args, kwargs)
                tn = getattr(tgt, "__name__", "")
                lowering = (FUNC_TABLE.get(tgt)
                            or FUNC_TABLE_BY_NAME.get(tn)
                            or FUNC_TABLE_BY_NAME.get(tn.lstrip("_"))
                            or ATEN_TABLE.get(name))
                is_producer = (getattr(node.target, "__name__", "") in _PRODUCERS
                               or getattr(node.target, "__name__", "") in _ALWAYS_LOWER
                               or name in _PRODUCERS
                               or name in _ALWAYS_LOWER)
                if node in refused:
                    lowering = None
                if lowering is not None and (has_mx(args) or is_producer):
                    try:
                        val = lowering(*_deep_mx(args), **{k: _deep_mx(v) for k, v in kwargs.items()})
                        on_mlx = True
                        # d-review: an in-place ATen op (masked_fill_, add_, ...)
                        # must mutate the graph's input placeholder, so later
                        # nodes - and an output that returns the placeholder -
                        # see the filled values, matching torch.
                        if name.endswith("_") and node.args \
                                and isinstance(node.args[0], torch.fx.Node):
                            env[node.args[0]] = val
                    except Exception as e:
                        def _sh(v):
                            if isinstance(v, (mx.array, torch.Tensor)):
                                return f"{type(v).__name__}{tuple(v.shape)}"
                            if isinstance(v, (tuple, list)):
                                return "(" + ",".join(_sh(x) for x in v) + ")"
                            return repr(v)[:24]
                        log.info("LOWER-FAIL\t%s\t%s\targs=%s", name, e, [_sh(a) for a in args])
                if not on_mlx and node.target in GLUE_FNS:
                    val = node.target(*args, **kwargs)
                    stats["glue"].append(getattr(node.target, "__name__", str(node.target)))
                    env[node] = val
                    _tick(name)
                    continue
            elif node.op == "call_method":
                if node in refused:
                    val, on_mlx = None, False
                else:
                    try:
                        val, on_mlx = _run_method(args[0], str(node.target), args[1:], kwargs)
                    except NotImplementedError:
                        val, on_mlx = None, False
                    except Exception as e:
                        log.info("LOWER-FAIL\t%s\t%s", node.target, e)
                        val, on_mlx = None, False

            if not on_mlx:
                if has_mx(args) or has_mx(kwargs.values()):
                    stats["crossings"] += 1
                targs = type(args)(_torch(a) for a in args)
                tkwargs = {k: _torch(v) for k, v in kwargs.items()}
                if node.op == "call_function":
                    val = node.target(*targs, **tkwargs)
                elif node.op == "call_method":
                    val = getattr(targs[0], str(node.target))(*targs[1:], **tkwargs)
                elif node.op == "call_module":
                    val = gm.get_submodule(str(node.target))(*targs, **tkwargs)
                else:
                    raise RuntimeError(f"unhandled {node.op} {node.target}")
                stats["cpu"].append(name)
                log.info("FALLBACK\t%s\t%s", name, node.target)
            else:
                stats["mlx"].append(name)
                log.info("LOWERED\t%s\t%s", name, node.target)
            env[node] = val
            if (name in ("mul_", "bitwise_and_", "sub_") and node.op in ("call_function", "call_method")
                    and isinstance(node.args[0], torch.fx.Node)):
                env[node.args[0]] = val
            if _DBG_ENV is not None:
                _DBG_ENV[node] = val
            _tick(name)
        return None

    return run, stats


def _eager_run(gm, return_all_outputs=False):
    run, stats = make_executor(gm)
    run._mlx_stats = stats
    return run


def _boxed(runner):
    """aot_autograd's runtime passes the inputs flat (some paths boxed as
    one list). Normalize to *args for the executor, which consumes one
    positional per placeholder."""
    def call(*inputs):
        flat = inputs
        if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                and not isinstance(inputs[0], mx.array)
                and all(isinstance(i, (torch.Tensor, mx.array))
                        for i in inputs[0])):
            flat = tuple(inputs[0])
        return runner(*flat)

    call._boxed_call = True
    return call


@torch._dynamo.register_backend(name="mlx_vulkan")
def mlx_vulkan_backend(gm: torch.fx.GraphModule, example_inputs, _dt=None,
                       return_all_outputs=False):
    if os.environ.get("MLX_VK_EAGER"):
        return _eager_run(gm)
    _record_artifact(gm)
    try:
        run_c = make_compiled_executor(gm, dt=_dt,
                                       return_all_outputs=return_all_outputs)
    except Exception as e:
        log.warning("compiled executor setup failed: %s", e)
        return _eager_run(gm, return_all_outputs)
    holder = {"run": None}

    def guarded(*args):
        if holder["run"] is None:
            try:
                out = run_c(*args)
            except Exception as e:
                log.warning("compiled executor failed (%s); falling back to eager", e,
                            exc_info=True)
                holder["run"] = _eager_run(gm)
            else:
                holder["run"] = run_c
                return out
        return holder["run"](*args)

    guarded._mlx_stats = run_c._mlx_stats
    guarded.return_all_outputs = return_all_outputs
    return guarded


# Training variant: AOTAutograd splits fwd/bwd and compiles both subgraphs
# through the same lowering tables, so *_backward aten nodes lower to mx
# (aten.convolution_backward -> transposed conv + batch->groups conv via
# mx.conv_general; pools fwd+bwd). Use for train graphs:
# backend="mlx_vulkan_train". Kept a separate name: aot functionalizes
# input mutations, which the decode state machinery in
# make_compiled_executor does not model.
def _aot_compiler(gm, example_inputs):
    try:
        from torch._functorch._aot_autograd.utils import make_boxed_func
    except ImportError:
        from functorch.compile import make_boxed_func
    # Training defaults to fp32: bf16 grads drift the loss vs CPU. Explicit
    # MV_DT still overrides. MV_AOT_EAGER=1 forces the per-node interpreter
    # (A/B against the compiled executor).
    run = (_eager_run(gm) if os.environ.get("MV_AOT_EAGER")
           else mlx_vulkan_backend(gm, example_inputs,
                                   _dt=None if os.environ.get("MV_DT") else mx.float32))

    def unboxed(*args):
        # aot's bw runtime may deliver the arg list nested one level deep
        while len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        return run(*args)

    return make_boxed_func(unboxed)


try:
    from torch._dynamo.backends.common import aot_autograd
    torch._dynamo.register_backend(name="mlx_vulkan_train")(
        aot_autograd(fw_compiler=_aot_compiler,
                     bw_compiler=_aot_compiler,
                     inference_compiler=_aot_compiler))
except Exception as e:
    log.warning("mlx_vulkan_train backend unavailable: %s", e)


# ================= compiled executor (v2) =================
# Whole-graph lazy lowering wrapped in mx.compile. Cache mutations
# (index_copy_/add_ on StaticCache buffers) are modeled functionally: updated
# tensors become extra outputs fed back as inputs next call, so HF's buffers
# only seed state once. RMSNorm chains fuse to mx.fast.rms_norm; the sdpa FX
# node maps to mx.fast.scaled_dot_product_attention; GEMM/sdpa inputs run bf16
# while the residual/elementwise spine stays fp32 -> ids match CPU fp32.
#
# bf16 is the right default on Apple/Honeykrisp (measured ~2x decode on
# Qwen3-0.6B) but not on Intel ANV: there fp32 GEMM is faster (7.5 vs 6.9
# tok/s, B60) AND bf16 activation casts flip ids - a 0.024-margin top-2 tie
# at decode step 5 resolved to the wrong index (PT-D5 bisect, greedy ids vs
# CPU diverge at token 5 under bf16, 32/32 match under fp32). Default fp32 on
# Intel parts; MV_DT=bf16|fp32 forces either arm.

def _default_dt():
    v = os.environ.get("MV_DT")
    if v:
        return {"bf16": mx.bfloat16, "fp32": mx.float32}[v]
    try:
        info = mx.device_info()
        if info.get("vendor_id") == 0x8086 or "intel" in str(info.get("driver", "")).lower():
            return mx.float32
    except Exception:
        pass
    return mx.bfloat16

_DT = _default_dt()

# Optional weight quantization (MV_QUANT=4): static params whose every
# consumer chain ends at a matmul-family op are stored packed via mx.quantize
# and matmul lowerings emit mx.quantized_matmul. Transparent ops propagate a
# transpose flag; any other consumer dequantizes on the fly.
_Q_BITS = int(os.environ.get("MV_QUANT", "0") or 0)
_Q_GS = int(os.environ.get("MV_QUANT_GS", "64"))
_Q_MM = {"linear", "matmul", "mm", "bmm", "addmm"}
_Q_TR = {"t", "transpose", "permute"}
_Q_PASS = {"to", "detach", "clone", "contiguous", "float", "half", "type"}


class _QW:
    __slots__ = ("w", "s", "b", "tr")

    def __init__(self, w, s, b, tr=False):
        self.w, self.s, self.b, self.tr = w, s, b, tr

    def dequant(self):
        return mx.dequantize(self.w, self.s, self.b, group_size=_Q_GS, bits=_Q_BITS)


def _dqall(v):
    if isinstance(v, _QW):
        return v.dequant()
    if isinstance(v, (tuple, list)):
        return type(v)(_dqall(x) for x in v)
    return v


def _node_name(n):
    if n.op == "call_method":
        return str(n.target)
    return (getattr(n.target, "__name__", None) or _opname(n)).split(".")[0]


def _quantizable_phs(phs):
    """Static 2-D params used only by matmul-family ops (through transparent ops)."""
    ok = set()
    for p in phs:
        if not str(p.target).startswith(("G_", "L_self_", "l_self_")):
            continue
        good, seen, stack = True, set(), list(p.users)
        while stack and good:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            nm = _node_name(u)
            if nm in _Q_MM:
                continue
            if nm in _Q_TR or nm in _Q_PASS:
                stack.extend(u.users)
                continue
            good = False
        if good:
            ok.add(str(p.target))
    return ok


def _fuse_rmsnorm(nodes):
    """x -> [to] -> pow(2) -> mean -> add(eps) -> rsqrt -> mul(x, r) -> [to] -> mul(w)"""
    fused = {}
    skip = set()
    consumers = {}
    nm = {}
    for n in nodes:
        nm[n] = getattr(n.target, "__name__", None) or _opname(n)
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                consumers.setdefault(a, []).append(n)
    for r in nodes:
        if nm[r] != "rsqrt":
            continue
        a = r.args[0]
        if not (isinstance(a, torch.fx.Node) and nm[a] == "add" and len(a.args) == 2):
            continue
        eps = a.args[1]
        if not isinstance(eps, (int, float)):
            continue
        m, p = a.args[0], None
        if not (isinstance(m, torch.fx.Node) and nm[m] == "mean"):
            continue
        p = m.args[0]
        if not (isinstance(p, torch.fx.Node) and nm[p] == "pow"):
            continue
        x_c = p.args[0]
        mm = next((c for c in consumers.get(r, [])
                   if nm[c] == "mul" and c.args[0] is x_c), None)
        if mm is None:
            continue
        tail = mm
        cand = consumers.get(tail, [])
        if len(cand) == 1 and nm[cand[0]] == "to":
            tail = cand[0]
            cand = consumers.get(tail, [])
        out = next((c for c in cand if nm[c] == "mul"), None)
        if out is None:
            continue
        w_node = out.args[1] if out.args[0] is tail else out.args[0]
        fused[out] = (x_c, w_node, float(eps), {r, a, m, p, mm, tail})
    inputs_used = set()
    for x_c, w_node, _, _c in fused.values():
        inputs_used.update({x_c, w_node})
    final = {}
    for out, (x_c, w_node, eps, chain) in fused.items():
        # skip a member only if every consumer stays inside the fused set and
        # it is not an input to another fused chain
        ok = all(c in chain or c in fused
                 for x in chain - inputs_used
                 for c in consumers.get(x, []))
        if ok:
            final[out] = (x_c, w_node, eps)
            skip.update(chain - inputs_used)
    return skip, final


def _fuse_rope(nodes):
    """HF NeoX-style rotary: add(mul(x, cos), mul(cat(neg(x[h:]), x[:h]), sin))
    -> mx.fast.rope(x, 2h, traditional=False, base=theta, offset=pos0).

    Only fuses when the cos/sin operands provably come from cos()/sin() nodes
    (optionally scaled by a python-float attention scaling). Returns
    (skip, fused) where fused[add_node] = (x_node, half_dim, out_scale).
    """
    nm = {n: getattr(n.target, "__name__", None) or _opname(n) for n in nodes}

    def _slice_get(n):
        # only a slice on the LAST axis (preceded by Ellipsis/full slices)
        if nm.get(n) != "getitem" or len(n.args) != 2:
            return None
        idx = n.args[1]
        if not isinstance(idx, tuple):
            idx = (idx,)
        # last-axis slice is only provable via Ellipsis (x[..., h:]): a bare
        # x[:h] is a one-element index on axis 0, and x[:, :h] slices axis 1;
        # without the tensor's ndim neither can be distinguished.
        if not idx or Ellipsis not in idx or not isinstance(idx[-1], slice):
            return None
        if any(s is not Ellipsis and s != slice(None) for s in idx[:-1]):
            return None
        sl = idx[-1]
        if sl.step is not None:
            return None
        return n.args[0], sl.start, sl.stop

    def _rot_src(cat):
        if nm.get(cat) != "cat":
            return None
        seq = cat.args[0]
        dim = cat.kwargs.get("dim", cat.args[1] if len(cat.args) > 1 else -1)
        if dim != -1 or not (isinstance(seq, (tuple, list)) and len(seq) == 2):
            return None
        nneg, x1 = seq
        if nm.get(nneg) != "neg" or not nneg.args:
            return None
        g2, g1 = _slice_get(nneg.args[0]), _slice_get(x1)
        if not g1 or not g2 or g1[0] is not g2[0]:
            return None
        if (g1[1] in (0, None) and isinstance(g2[1], int) and g2[1] > 0
                and g2[1] == g1[2] and g2[2] is None):
            return g1[0], g2[1], {cat, nneg, x1, nneg.args[0]}
        return None

    _THIN = ("unsqueeze", "to", "contiguous", "detach", "float")

    def _trig_root(a, want):
        # unsqueeze/cast*(trig | mul(trig, scalar)) -> (node, scale)
        while (nm.get(a) in _THIN
               and a.args and isinstance(a.args[0], torch.fx.Node)):
            a = a.args[0]
        if nm.get(a) == want:
            return a, 1.0
        if nm.get(a) == "mul" and len(a.args) == 2:
            for t, sc in ((a.args[0], a.args[1]), (a.args[1], a.args[0])):
                if nm.get(t) == want and isinstance(sc, (int, float)):
                    return t, float(sc)
        return None, None

    def _thin_root(a):
        while (nm.get(a) in _THIN
               and a.args and isinstance(a.args[0], torch.fx.Node)):
            a = a.args[0]
        return a

    fused = {}
    for n in nodes:
        if nm.get(n) != "add" or len(n.args) != 2:
            continue
        for m_cos, m_sin in ((n.args[0], n.args[1]), (n.args[1], n.args[0])):
            if nm.get(m_cos) != "mul" or nm.get(m_sin) != "mul":
                continue
            cat = next((a for a in m_sin.args
                        if isinstance(a, torch.fx.Node) and nm.get(a) == "cat"), None)
            rot = _rot_src(cat) if cat is not None else None
            if not rot:
                continue
            x_rot, half, chain = rot
            # m_cos must multiply the SAME rotated tensor: mul(x, cos).
            if (len(m_cos.args) != 2 or x_rot not in m_cos.args
                    or len(m_sin.args) != 2):
                break
            sin_op = next((a for a in m_sin.args if a is not cat
                           and isinstance(a, torch.fx.Node)), None)
            cos_op = next((a for a in m_cos.args if a is not x_rot
                           and isinstance(a, torch.fx.Node)), None)
            if cos_op is None or sin_op is None:
                break
            cn, csc = _trig_root(cos_op, "cos")
            sn, ssc = _trig_root(sin_op, "sin")
            # cos and sin must share the same angle input (compared after
            # stripping transparent wrappers: dynamo may not CSE them).
            if (cn is None or sn is None or csc != ssc
                    or not cn.args or not sn.args
                    or not isinstance(cn.args[0], torch.fx.Node)
                    or not isinstance(sn.args[0], torch.fx.Node)
                    or _thin_root(cn.args[0]) is not _thin_root(sn.args[0])):
                break
            fused[n] = (x_rot, half, csc, {m_cos, m_sin} | chain)
            break
    consumers = {}
    for n in nodes:
        for a in n.args:
            if isinstance(a, torch.fx.Node):
                consumers.setdefault(a, []).append(n)
            elif isinstance(a, (tuple, list)):
                for x in a:
                    if isinstance(x, torch.fx.Node):
                        consumers.setdefault(x, []).append(n)
    skip = set()
    final = {}
    for out, (x_rot, half, csc, chain) in fused.items():
        ok = all(c in chain or c in fused
                 for x in chain - {x_rot}
                 for c in consumers.get(x, []))
        if ok:
            final[out] = (x_rot, half, csc)
            skip.update(chain)
    return skip, final


def _inplace_writeback(dst, upd):
    # torch in-place contract: the result must fit the destination shape and
    # cast back to the destination dtype (float-family -> integral is
    # rejected, and nothing casts into bool; int64 -> int8 wraps fine).
    if upd.shape != dst.shape:
        raise RuntimeError(
            "output with shape %s doesn't match the broadcast shape %s"
            % (list(dst.shape), list(upd.shape)))
    if upd.dtype != dst.dtype:
        if (dst.dtype == mx.bool_ or
                (_is_floatish_dtype(upd.dtype) and _is_integral_dtype(dst.dtype))):
            raise RuntimeError(
                "result type %s can't be cast to the desired output type %s"
                % (upd.dtype, dst.dtype))
        upd = upd.astype(dst.dtype)
    return upd


def make_compiled_executor(gm, dt=None, return_all_outputs=False):
    nodes = list(gm.graph.nodes)
    phs = [n for n in nodes if n.op == "placeholder"]
    strided = _strided_nodes(nodes)
    skip, fused = _fuse_rmsnorm(nodes)
    rskip, rfused = _fuse_rope(nodes)
    pos_ph = next((n for n in phs if "position_ids" in str(n.target)), None)
    invf_ph = next((n for n in phs if "inv_freq" in str(n.target)), None)
    _halves = {v[1] for v in rfused.values()}
    _n_invf = sum(1 for n in phs if "inv_freq" in str(n.target))
    # fuse only when every site shares one rotary width, exactly one inv_freq
    # placeholder exists (multi-theta graphs need per-site bases), and we can
    # find the position placeholder the emit needs.
    if pos_ph is None or invf_ph is None or len(_halves) != 1 or _n_invf != 1:
        rfused = {}
    else:
        skip |= rskip
    # mx.fast.rope takes base=rope_theta (this fork's freqs= is the wavelength
    # basis, not inv_freq). theta = inv_freq[1]^(-dims/2); recovered once from
    # the runtime tensor so no host sync lands inside the traced graph.
    _rope = {"base": None}
    _invf_i = phs.index(invf_ph) if invf_ph is not None else -1
    _pos_i = phs.index(pos_ph) if pos_ph is not None else -1
    _rope_half = next(iter(rfused.values()))[1] if rfused else 0
    fp32_params = {str(w.target) for _, w, _ in fused.values()}
    # embedding gathers feed the whole residual spine; keep their source weights
    # fp32 (a gather is not a GEMM and gains nothing from the bf16 cast).
    _pass = {"to", "detach", "clone", "contiguous", "view"}
    for n in nodes:
        if n.op == "call_function" and _opname(n) == "embedding" and len(n.args) > 1:
            w = n.args[1]
            while getattr(w, "op", None) == "call_method" and _opname(w) in _pass and w.args:
                w = w.args[0]
            if getattr(w, "op", None) == "placeholder":
                fp32_params.add(str(w.target))
    # same-storage, same-shape aliases only: an in-place op on a derived
    # tensor (neg/matmul/cat output, a dtype-cast result, a reshaped view)
    # does not write back to the placeholder, so it must not mark state and
    # its post-mutation value must not be written back to env[ph] either.
    _alias = {"detach", "alias", "lift_fresh", "lift_fresh_copy"}

    def _ph_of(n):
        while n is not None and n.op != "placeholder":
            if ((getattr(n.target, "__name__", None) or _opname(n)) not in _alias
                    or not getattr(n, "args", None)
                    or not isinstance(n.args[0], torch.fx.Node)):
                return None
            n = n.args[0]
        return n

    def _state_ph(t):
        # torch>=2.15 names StaticCache buffers
        # L_kwargs_past_key_values_layers_N_{keys,values,cumulative_length};
        # older dynamo lifted them as *_cache_layers_*.
        return ("_cache_layers_" in t or
                ("past_key_values" in t and
                 t.endswith(("_keys", "_values", "_cumulative_length"))))

    state_names = {str(n.target) for n in phs if _state_ph(str(n.target))}
    # cache placeholders only count as mutated state when the graph actually
    # mutates them in place; HF concat-style cache (cat -> new outputs) must not
    # replay step-1 state on later calls.
    _mutfns = ("index_copy_", "add_", "sub_", "mul_", "div_", "copy_", "scatter_", "index_put_", "bitwise_and_")
    if not any(
            n.op == "call_method" and _opname(n) in _mutfns
            and getattr(n.args[0], "op", None)
            and (lambda p: p is not None and str(p.target) in state_names)(
                _ph_of(n.args[0]))
            for n in nodes):
        state_names = set()
    # dynamo names: G_* (older torch lifted globals) and L_self_*/l_self_*
    # (torch>=2.4 nn params/buffers) are the static weights; L_kwargs_* are
    # per-call inputs and stay dynamic.
    static_names = ({str(n.target) for n in phs
                     if str(n.target).startswith(("G_", "L_self_", "l_self_"))}
                    - state_names)
    mutated = [n for n in phs if str(n.target) in state_names]
    quant_names = _quantizable_phs(phs) if _Q_BITS else set()
    conv_static = {}
    state = {}
    holder = {}

    _dte = dt or (mx.float32
                  if return_all_outputs and not os.environ.get("MV_DT")
                  else _DT)

    def _cast(x, dt=_dte):
        return x.astype(dt) if isinstance(x, mx.array) and x.dtype != dt else x

    def build(*flat):
        env = _DBG_ENV if _DBG_ENV is not None else {}
        for n, v in zip(phs, flat):
            env[n] = _QW(*v) if str(n.target) in quant_names and isinstance(v, (tuple, list)) else v

        def res(a):
            if isinstance(a, torch.fx.Node):
                if a not in env:
                    order = []
                    for n2 in nodes:
                        order.append("%s/%s" % (n2.op, str(n2.target)[:20]))
                        if n2 is a:
                            break
                    raise KeyError("node %s (%s) target=%s not in env; the "
                                   "graph order up to it: %s"
                                   % (a, a.op, str(a.target)[:40],
                                      order[-8:]))
                return env[a]
            if isinstance(a, (tuple, list)):
                return type(a)(res(x) for x in a)
            return a

        for n in nodes:
            if n.op == "placeholder" or n in skip:
                continue
            name = (getattr(n.target, "__name__", None) or _opname(n)).split(".")[0]
            if n.op == "output":
                out = res(n.args[0])
                holder["n_out"] = len(out) if isinstance(out, (tuple, list)) else 0
                tail = [env[m] for m in mutated]
                return (*out, *tail) if isinstance(out, (tuple, list)) else (out, *tail)
            if n.op == "get_attr":
                t = gm
                for part in str(n.target).split("."):
                    t = getattr(t, part)
                env[n] = _mx(t)
                continue
            if n in fused:
                xn, wn, eps = fused[n]
                env[n] = mx.fast.rms_norm(env[xn], env[wn], eps)
                continue
            if n in rfused:
                xn, half, osc = rfused[n]
                _x, _p = env[xn], env[pos_ph]
                # fused rope only models [B,H,L,D] with the sequence on axis 2
                # and the rotary width covering the head dim; anything else
                # must fail loudly at trace time, not miscompile.
                if (_x.ndim != 4 or _x.shape[-1] != 2 * half
                        or _x.shape[2] != _p.shape[-1]):
                    raise RuntimeError(
                        "fused rope: unsupported layout "
                        f"x={_x.shape} pos={_p.shape} half={half}")
                r = mx.fast.rope(_x, 2 * half, traditional=False,
                                 base=_rope["base"], scale=1.0,
                                 offset=_p.reshape(-1)[0])
                env[n] = r * osc if osc != 1.0 else r
                continue
            args = tuple(res(a) for a in n.args)
            kwargs = {k: res(v) for k, v in n.kwargs.items()}
            if any(isinstance(a, _QW) for a in args) or any(
                    isinstance(v, _QW) for v in kwargs.values()):
                done = False
                if isinstance(args[0], _QW):
                    if name in ("t", "transpose"):
                        env[n] = _QW(args[0].w, args[0].s, args[0].b, not args[0].tr)
                        done = True
                    elif name == "permute":
                        dims = args[1] if len(args) > 1 else tuple(args[1:])
                        if tuple(dims) == (1, 0):
                            env[n] = _QW(args[0].w, args[0].s, args[0].b, not args[0].tr)
                            done = True
                        elif tuple(dims) == (0, 1):
                            env[n] = args[0]
                            done = True
                    elif name in _Q_PASS:
                        env[n] = args[0]
                        done = True
                if not done and len(args) > 1 and isinstance(args[1], _QW):
                    q = args[1]
                    if name == "linear":
                        y = mx.quantized_matmul(_cast(args[0]), q.w, q.s, q.b,
                                                transpose=not q.tr,
                                                group_size=_Q_GS, bits=_Q_BITS)
                        if len(args) > 2 and args[2] is not None:
                            y = y + _cast(args[2])
                        env[n] = y.astype(mx.float32)
                        done = True
                    elif name in ("matmul", "mm", "bmm"):
                        env[n] = mx.quantized_matmul(_cast(args[0]), q.w, q.s, q.b,
                                                     transpose=q.tr,
                                                     group_size=_Q_GS, bits=_Q_BITS).astype(mx.float32)
                        done = True
                if done:
                    continue
                args = _dqall(args)
                kwargs = {k: _dqall(v) for k, v in kwargs.items()}
            if name == "index_copy_":
                t, dim, idx, src = args[0], args[1], args[2], args[3]
                if _has_bool_index(idx):
                    raise RuntimeError(
                        "aten.index_copy_: a bool index tensor is a mask in "
                        "aten; refused to the per-node path for torch semantics")
                if dim < 0:
                    dim += src.ndim
                sh = [1] * src.ndim
                sh[dim] = idx.shape[0] if idx.ndim else 1
                ib = mx.broadcast_to(idx.reshape(sh), src.shape)
                upd = mx.put_along_axis(t, ib, src.astype(t.dtype), axis=dim)
                env[n.args[0]] = upd      # model in-place mutation
                _ph = _ph_of(n.args[0])
                if _ph is not None and _ph is not n.args[0]:
                    env[_ph] = upd        # same-storage alias of a state ph
                env[n] = upd
                continue
            if name == "add_":
                alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                upd = _inplace_writeback(env[n.args[0]],
                                         env[n.args[0]] + args[1] * alpha)
                env[n.args[0]] = upd
                _ph = _ph_of(n.args[0])
                if _ph is not None and _ph is not n.args[0]:
                    env[_ph] = upd        # same-storage alias of a state ph
                env[n] = upd
                continue
            if name in ("sub_", "sub_.Tensor"):
                dst = env[n.args[0]]
                alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                if (dst.dtype == mx.bool_
                        or _dtype_of(args[1]) == mx.bool_):
                    raise RuntimeError(
                        "Subtraction, the `-` operator, with a bool tensor is "
                        "not supported. If you are trying to invert a mask, "
                        "use the `~` or `logical_not()` operator instead.")
                upd = _inplace_writeback(dst, dst - args[1] * alpha)
                env[n.args[0]] = upd
                _ph = _ph_of(n.args[0])
                if _ph is not None and _ph is not n.args[0]:
                    env[_ph] = upd        # same-storage alias of a state ph
                env[n] = upd
                continue
            if name in ("rsub", "rsub.Scalar") and n.op == "call_function":
                alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                env[n] = _rsub(args[0], args[1], alpha)
                continue
            if name == "mul_":
                upd = _inplace_writeback(env[n.args[0]], env[n.args[0]] * args[1])
                env[n.args[0]] = upd
                _ph = _ph_of(n.args[0])
                if _ph is not None and _ph is not n.args[0]:
                    env[_ph] = upd        # same-storage alias of a state ph
                env[n] = upd
                continue
            if name == "bitwise_and_":
                upd = _inplace_result(
                    _bitwise_or_logical(env[n.args[0]], args[1], mx.bitwise_and, mx.logical_and),
                    env[n.args[0]])
                env[n.args[0]] = upd
                _ph = _ph_of(n.args[0])
                if _ph is not None and _ph is not n.args[0]:
                    env[_ph] = upd        # same-storage alias of a state ph
                env[n] = upd
                continue
            # The aot backward graph: these ops must stay on mx or the whole
            # training step crosses to torch per node (PT-T1 speed gate).
            # Unmatched ops fall through to the generic handlers below; a
            # per-op failure raises and lets the guarded wrapper fall back
            # to the per-node eager executor, keeping the run correct.
            if name == "t" and n.op == "call_function":
                env[n] = mx.swapaxes(args[0], -1, -2)
                continue
            if name in ("transpose", "transpose.int") and n.op == "call_function":
                a0 = args[0]
                d0, d1 = int(args[1]), int(args[2])
                env[n] = mx.swapaxes(a0, d0 % a0.ndim, d1 % a0.ndim)
                continue
            if name in _VIEW_OPS and n.op == "call_function":
                if _view_refused(name, n, strided):
                    raise RuntimeError(
                        f"{name} on a possibly non-contiguous input: "
                        "refused to the per-node path for torch stride "
                        "semantics")
                shape = args[1]
                if isinstance(shape, torch.fx.Node):
                    shape = env[shape]
                if isinstance(shape, int):
                    shape = (shape,)
                env[n] = mx.reshape(args[0], tuple(int(s) for s in shape))
                continue
            if name == "detach" and n.op == "call_function":
                env[n] = args[0]

                continue
            if name == "scaled_dot_product_attention":
                # GQA: HF expands KV to q-heads via expand+reshape before sdpa;
                # mx.fast sdpa broadcasts fewer kv heads natively -> look through.
                def _unexpand(narg):
                    if getattr(narg, "op", None) == "call_method" and _opname(narg) in ("reshape", "view"):
                        narg = narg.args[0]
                    if getattr(narg, "op", None) == "call_method" and _opname(narg) == "expand":
                        return env.get(narg.args[0])
                    return None
                uk, uv = _unexpand(n.args[1]), _unexpand(n.args[2])
                if uk is not None and uk.ndim == 5 and uk.shape[2] == 1:
                    uk = uk[:, :, 0]
                if uv is not None and uv.ndim == 5 and uv.shape[2] == 1:
                    uv = uv[:, :, 0]
                q = _cast(args[0])
                k = _cast(uk) if uk is not None else _cast(args[1])
                v = _cast(uv) if uv is not None else _cast(args[2])
                mask = kwargs.get("attn_mask")
                if mask is None and kwargs.get("is_causal"):
                    mask = "causal"  # prefill graphs emit is_causal instead of a mask
                scale = kwargs.get("scale")
                if scale is None:
                    scale = 1.0 / math.sqrt(q.shape[-1])
                if isinstance(mask, mx.array) and mask.dtype != getattr(mx, "bool_", mx.uint8):
                    mask = _cast(mask)
                env[n] = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=scale, mask=mask).astype(mx.float32)
                continue
            if name in ("matmul", "mm", "bmm") and n.op == "call_function":
                env[n] = mx.matmul(_cast(args[0]), _cast(args[1])).astype(mx.float32)
                continue
            if name == "linear" and n.op == "call_function":
                y = mx.matmul(_cast(args[0]), _wt(_cast(args[1])))
                bias = args[2] if len(args) > 2 else kwargs.get("bias")
                if bias is not None:
                    y = y + _cast(bias)
                env[n] = y.astype(mx.float32)
                continue
            if n.op == "call_method":
                if _view_refused(name, n, strided):
                    raise RuntimeError(
                        f"{name} on a possibly non-contiguous input: "
                        "refused to the per-node path for torch stride "
                        "semantics")
                val, on_mx = _run_method(args[0], name, args[1:], kwargs)
                if on_mx:
                    env[n] = val
                    continue
                env[n] = getattr(args[0], name)(*args[1:], **kwargs)
                continue
            if n.target is operator.getitem:
                if isinstance(args[0], mx.array) and _has_bool_index(args[1]):
                    raise RuntimeError(
                        "getitem with a bool mask index: the gather path reads "
                        "bool as 0/1; refused to the per-node path")
                env[n] = args[0][_coerce_index(args[1])]
                continue
            if n.op == "call_function":
                if name in _SAMPLERS:
                    raise RuntimeError(
                        f"unlowerable node {name}: mx.compile freezes "
                        "mx.random draws; sampling stays on the "
                        "per-node path")
                tgt = _bd_inner(n.target, args, kwargs)
                tn = getattr(tgt, "__name__", "")
                low = (FUNC_TABLE.get(tgt)
                       or FUNC_TABLE_BY_NAME.get(tn)
                       or FUNC_TABLE_BY_NAME.get(tn.lstrip("_"))
                       or ATEN_TABLE.get(name))
                if low is not None:
                    kw2 = {k: v for k, v in kwargs.items() if k != "device"}
                    env[n] = low(*args, **kw2)
                    continue
                if n.target in GLUE_FNS:
                    env[n] = n.target(*args, **kwargs)
                    continue
            raise RuntimeError(f"unlowerable node {name} {n.target}")
        raise RuntimeError("no output node")

    cfn = mx.compile(build)
    stats = {"fused_rms": len(fused), "fused_rope": len(rfused),
             "state": len(state_names), "static": len(static_names),
             "quant": len(quant_names)}
    log.warning("compiled executor: fused_rms=%d fused_rope=%d state=%d static=%d quant=%d",
                stats["fused_rms"], stats["fused_rope"], stats["state"],
                stats["static"], stats["quant"])

    import time as _t
    _profile = os.environ.get("MV_PROF")
    _acc = {"prep": 0.0, "cfn": 0.0, "conv": 0.0, "n": 0}

    ph_names = [str(n.target) for n in phs]
    _slots = {"flat": None, "dyn": [], "st": []}
    pcache = {}

    def run(*runtime_inputs):
        t0 = _t.perf_counter()
        flat = _slots["flat"]
        if flat is not None:
            # steady state: statics are resident; refresh only state slots
            # (rebound by the mutated-tail writeback) and per-call inputs.
            _st, _dyn = _slots["st"], _slots["dyn"]
            for i in _st:
                flat[i] = state[ph_names[i]]
            for i, n in _dyn:
                v = runtime_inputs[i]
                # aot partitions re-feed the same param tensors each step:
                # reuse the mx array unless the torch tensor was mutated
                ent = pcache.get(n)
                if (ent is not None and ent[0] is v
                        and ent[1] == getattr(v, "_version", None)):
                    flat[i] = ent[2]
                else:
                    arr = _mx(v)
                    if isinstance(v, torch.Tensor):
                        pcache[n] = (v, getattr(v, "_version", None), arr)
                    flat[i] = arr
        else:
            flat, st, dyn = [], [], []
            for i, (n, v) in enumerate(zip(phs, runtime_inputs)):
                name = ph_names[i]
                if name in state:
                    flat.append(state[name])
                    st.append(i)   # retry path: keep the slot tracked
                elif name in conv_static:
                    flat.append(conv_static[name])
                elif name in state_names:
                    arr = _mx(v)
                    if not isinstance(arr, mx.array):
                        arr = mx.array(arr)   # scalars must not be signature constants
                    elif arr.ndim >= 2:
                        arr = _cast(arr)
                    state[name] = arr
                    flat.append(arr)
                    st.append(i)
                else:
                    arr = _mx(v)
                    if name in static_names:
                        if name in quant_names:
                            try:
                                arr = mx.quantize(arr, group_size=_Q_GS, bits=_Q_BITS)
                                mx.eval(*arr)  # force now: lazy graphs would hold every bf16 source
                            except Exception:
                                quant_names.discard(name)
                                arr = _cast(arr)
                        elif name not in fp32_params:
                            arr = _cast(arr)
                        conv_static[name] = arr
                    else:
                        dyn.append((i, n))
                        if isinstance(v, torch.Tensor):
                            pcache[n] = (v, getattr(v, "_version", None), arr)
                    flat.append(arr)
            # publish only after a clean build so a mid-setup exception
            # cannot leave a partially populated slot cache behind.
            _slots["st"], _slots["dyn"], _slots["flat"] = st, dyn, flat
        if rfused:
            # positions are gated on EVERY call (a caller may retry after a
            # call-1 raise or pass divergent rows later); the scalar-offset
            # emit only models per-row-uniform, unit-contiguous positions.
            pt = runtime_inputs[_pos_i]
            pt = pt.reshape(-1, pt.shape[-1])
            if not bool((pt == pt[:1]).all()):
                raise RuntimeError("fused rope: per-row positions unsupported")
            row = pt[0]
            if row.numel() > 1 and not bool((row[1:] - row[:-1] == 1).all()):
                raise RuntimeError("fused rope: non-contiguous positions")
            if _rope["base"] is None:
                # theta = inv_freq[1]^(-half) only for a geometric inv_freq
                # anchored at 1.0; scaled (linear rope: f[0] != 1), YaRN and
                # otherwise non-uniform spacing raise instead of miscomputing.
                f = runtime_inputs[_invf_i].reshape(-1)
                if f.numel() != _rope_half or f.numel() < 2:
                    raise RuntimeError(
                        f"fused rope: inv_freq numel {f.numel()} != half {_rope_half}")
                if abs(float(f[0]) - 1.0) > 1e-4:
                    raise RuntimeError("fused rope: scaled inv_freq")
                if not (0.0 < float(f[1]) < 1.0):
                    raise RuntimeError("fused rope: inv_freq out of range")
                r = f[:-1].double() / f[1:].double()
                if not bool(((r - r[0]).abs() <= 1e-4 * r[0].abs()).all()):
                    raise RuntimeError("fused rope: non-geometric inv_freq")
                _rope["base"] = float(f[1]) ** (-_rope_half)
        t1 = _t.perf_counter()
        outs = (build if _DBG_ENV is not None else cfn)(*flat)
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        mx.eval(*[o for o in outs if isinstance(o, mx.array)])
        t2 = _t.perf_counter()
        # outs = real outputs + len(mutated) state tail. aot graphs have no
        # mutated placeholders, so all outputs are real (and include the
        # saved-for-backward tensors aot needs back). holder["n_out"] marks
        # the boundary when the output node ran multi-output.
        n_out = holder.get("n_out", 1)
        real = outs[:n_out] if n_out else outs[:1]
        extra = outs[n_out:] if n_out else outs[1:]
        for m, arr in zip(mutated, extra):
            state[str(m.target)] = arr
        out = tuple(_torch(x) for x in real)
        t3 = _t.perf_counter()
        if _profile:
            _acc["prep"] += t1 - t0; _acc["cfn"] += t2 - t1; _acc["conv"] += t3 - t2; _acc["n"] += 1
            if _acc["n"] % 10 == 0:
                print(f"MV_PROF n={_acc['n']} prep={_acc['prep']/_acc['n']*1e3:.1f}ms cfn={_acc['cfn']/_acc['n']*1e3:.1f}ms conv={_acc['conv']/_acc['n']*1e3:.1f}ms", flush=True)
        if return_all_outputs:
            return tuple(_torch(t) for t in outs)
        return out

    run._mlx_stats = stats
    return run
